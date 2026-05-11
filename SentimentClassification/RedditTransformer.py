from transformers import AutoModel, AutoConfig
import torch
import copy


class RedditTransformer(torch.nn.Module):
    def __init__(self, model_name, num_classes, extra_dropout, num_groups):
        super(RedditTransformer, self).__init__()

        config = AutoConfig.from_pretrained(
            model_name,
            output_hidden_states=True,
            use_cache=False,
            attn_implementation="eager"
        )
        self.encoder = AutoModel.from_pretrained(model_name, config=config)

        # Add extra dropout
        for layer in self.encoder.encoder.layer:
            layer.attention.self.dropout = torch.nn.Dropout(
                self.encoder.config.attention_probs_dropout_prob + extra_dropout)
            layer.output.dropout = torch.nn.Dropout(self.encoder.config.hidden_dropout_prob + extra_dropout)

        print(self.encoder.config)

        self.classification_head = torch.nn.Sequential(
            torch.nn.Dropout(config.hidden_dropout_prob + extra_dropout),
            torch.nn.Linear(config.hidden_size, num_classes),
        )

        if num_groups is not None:
            # Overwrite the default encoder and pooler with our custom multi-task ones
            self.encoder.encoder = BertEncoder(config, self.encoder.encoder.layer)
            self.encoder.pooler = BertPooler(config, self.encoder.pooler.dense)
            self.aux = True

            self.classification_head_aux = torch.nn.Sequential(
                torch.nn.Dropout(config.hidden_dropout_prob + extra_dropout),
                torch.nn.Linear(config.hidden_size, num_groups),
            )
        else:
            self.aux = False

    def forward(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        # Restore the batch dimension if the dataloader dropped it
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        # -------------------------------------------------------------------
        # KEY FIX: Run BertModel's embedding + mask-extension logic manually,
        # then call our custom BertEncoder directly with the already-extended
        # 4-D mask.  This bypasses the BertModel.forward() path that was
        # re-extending (or not extending) the mask before reaching our encoder.
        # -------------------------------------------------------------------
        bert = self.encoder  # BertModel instance

        # 1. Embed the tokens  →  [B, S, H]
        embedding_output = bert.embeddings(
            input_ids=input_ids,
            position_ids=None,
            token_type_ids=torch.zeros_like(input_ids),
            inputs_embeds=None,
            past_key_values_length=0,
        )

        # 2. Extend mask  [B, S]  →  [B, 1, 1, S]  with additive -inf/0 values
        extended_attention_mask = bert.get_extended_attention_mask(
            attention_mask, input_ids.shape
        )

        # 3. Run our custom encoder directly (already receives 4-D mask)
        encoder_outputs = bert.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )

        # 4. Pool
        pooled_output = bert.pooler(encoder_outputs[0])

        # encoder_outputs[0] = hidden states tuple (main, aux)  [from BertEncoder]
        # pooled_output       = pooled tuple (main, aux)        [from BertPooler]

        if self.aux:
            features_main = pooled_output[0]
            features_aux  = pooled_output[1]

            logits_main = self.classification_head(features_main)
            logits_aux  = self.classification_head_aux(features_aux)

            return logits_main, logits_aux, encoder_outputs[0]
        else:
            features    = pooled_output
            logits_main = self.classification_head(features)

            return logits_main, None, encoder_outputs[0]


class BertEncoder(torch.nn.Module):
    def __init__(self, config, layers):
        super().__init__()
        self.config = config

        # Must wrap the sliced layers in ModuleList, or PyTorch won't train them!
        self.layer = torch.nn.ModuleList(layers[:-1])

        # Split the very last layer for the two different tasks
        self.layer_main = copy.deepcopy(layers[-1])
        self.layer_aux  = copy.deepcopy(layers[-1])

    def forward(
            self,
            hidden_states,
            attention_mask=None,        # arrives as 4-D extended mask from BertModel
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
            **kwargs,                   # absorb any future HuggingFace kwargs
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions    = () if output_attentions    else None

        layer_kwargs = {
            "attention_mask":    attention_mask,   # already 4-D — no extension needed
            "output_attentions": output_attentions,
        }

        # 1. Pass through the first N-1 shared layers
        for i, layer_module in enumerate(self.layer):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value  = past_key_values[i] if past_key_values is not None else None

            hidden_states, all_attentions, all_hidden_states = self._layer_loop(
                hidden_states, layer_module, layer_head_mask,
                all_hidden_states, all_attentions,
                past_key_value=past_key_value,
                **layer_kwargs,
            )

        # 2. Pass through the two split 12th layers (main task and aux task)
        layer_head_mask     = head_mask[len(self.layer)] if head_mask is not None else None
        past_key_value_last = past_key_values[len(self.layer)] if past_key_values is not None else None

        hidden_states_main, _, _ = self._layer_loop(
            hidden_states, self.layer_main, layer_head_mask,
            all_hidden_states, all_attentions,
            past_key_value=past_key_value_last,
            **layer_kwargs,
        )

        hidden_states_aux, _, _ = self._layer_loop(
            hidden_states, self.layer_aux, layer_head_mask,
            all_hidden_states, all_attentions,
            past_key_value=past_key_value_last,
            **layer_kwargs,
        )

        # 3. Return the split hidden states as a tuple in position [0]
        #    BertPooler expects encoder_outputs[0] to be (main, aux)
        return ((hidden_states_main, hidden_states_aux),)

    def _layer_loop(
            self, hidden_states, layer_module, layer_head_mask,
            all_hidden_states, all_attentions,
            attention_mask, output_attentions,
            past_key_value=None,
    ):
        """Process a single transformer layer, handling gradient checkpointing."""
        if all_hidden_states is not None:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if getattr(self.config, "gradient_checkpointing", False):
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, output_attentions=output_attentions)
                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(layer_module),
                hidden_states,
                attention_mask,
                layer_head_mask,
            )
        else:
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=layer_head_mask,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
            )

        if isinstance(layer_outputs, tuple):
            hidden_states = layer_outputs[0]
            if output_attentions and len(layer_outputs) > 1:
                all_attentions = all_attentions + (layer_outputs[1],)
        else:
            hidden_states = layer_outputs

        return hidden_states, all_attentions, all_hidden_states


class BertPooler(torch.nn.Module):
    def __init__(self, config, dense):
        super().__init__()
        # Split the dense layer for the two tasks
        self.dense_main = copy.deepcopy(dense)
        self.dense_aux  = copy.deepcopy(dense)
        self.activation = torch.nn.Tanh()

    def forward(self, hidden_states):
        # hidden_states is (hidden_states_main, hidden_states_aux)
        # each of shape [B, S, H] — grab the [CLS] token (position 0)
        first_token_tensor_main = hidden_states[0][:, 0]
        first_token_tensor_aux  = hidden_states[1][:, 0]

        pooled_output_main = self.activation(self.dense_main(first_token_tensor_main))
        pooled_output_aux  = self.activation(self.dense_aux(first_token_tensor_aux))

        return (pooled_output_main, pooled_output_aux)
