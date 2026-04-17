"""
Module A: Sentence-Level Bias & Emotion Engine
Multi-Task Learning (MTL) model with RoBERTa-base.

Architecture:
- Base Model: Hugging Face roberta-base
- Shared Representation: Layers 1-11 shared for general linguistic context
- MTL Heads (Layer 12 Split): 3 distinct predictive heads
  1. Primary Head: "Us vs. Them" continuous bias score (Regression, 0.0-1.0)
  2. Auxiliary Head 1: Emotion prediction (Multi-label classification)
  3. Auxiliary Head 2: Targeted social group prediction (Classification)
"""

from transformers import AutoModel, AutoConfig, AutoTokenizer
import torch
import torch.nn as nn
import copy
from typing import Dict, Tuple, Optional, List


class RoBERTaMTL(nn.Module):
    """
    Multi-Task Learning RoBERTa model with shared encoder layers 1-11
    and 3 task-specific heads at layer 12.
    """
    
    def __init__(
        self,
        model_name: str = "roberta-base",
        num_emotions: int = 13,
        num_social_groups: int = 10,
        extra_dropout: float = 0.0,
    ):
        """
        Initialize the MTL RoBERTa model.
        
        Args:
            model_name: HuggingFace model identifier (default: roberta-base)
            num_emotions: Number of emotion labels for multi-label classification
            num_social_groups: Number of targeted social group classes
            extra_dropout: Additional dropout to apply on top of base model dropout
        """
        super(RoBERTaMTL, self).__init__()
        
        # Load RoBERTa configuration with hidden states output enabled
        config = AutoConfig.from_pretrained(
            model_name,
            output_hidden_states=True,
            use_cache=False,
        )
        self.config = config
        
        # Load pre-trained RoBERTa encoder
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        
        # Apply extra dropout if specified
        if extra_dropout > 0:
            for layer in self.encoder.encoder.layer:
                layer.attention.self.dropout = nn.Dropout(
                    config.attention_probs_dropout_prob + extra_dropout
                )
                layer.output.dropout = nn.Dropout(
                    config.hidden_dropout_prob + extra_dropout
                )
        
        # Replace standard encoder with our custom MTL encoder that splits at layer 12
        # Save reference to original layers before replacement
        original_layers = self.encoder.encoder.layer
        self.encoder.encoder = RoBERTaMTLEncoder(config, original_layers)
        
        # Replace pooler with MTL-aware pooler that handles 3 task streams
        original_dense = self.encoder.pooler.dense if hasattr(self.encoder.pooler, 'dense') else None
        if original_dense is not None:
            self.encoder.pooler = RoBERTaMTLPooler(config, original_dense)
        else:
            # RoBERTa doesn't have a pooler by default, create one
            self.encoder.pooler = RoBERTaMTLPooler(config, None)
        
        # Task-specific classification heads
        hidden_size = config.hidden_size
        dropout_prob = config.hidden_dropout_prob + extra_dropout
        
        # Primary Head: Bias Score Regression (continuous 0.0-1.0)
        self.bias_head = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout_prob / 2),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),  # Ensure output is between 0 and 1
        )
        
        # Auxiliary Head 1: Emotion Prediction (Multi-label classification)
        self.emotion_head = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout_prob / 2),
            nn.Linear(hidden_size // 2, num_emotions),
            # No activation here - will use BCEWithLogitsLoss
        )
        
        # Auxiliary Head 2: Social Group Prediction (Classification)
        self.social_group_head = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout_prob / 2),
            nn.Linear(hidden_size // 2, num_social_groups),
            # No activation here - will use CrossEntropyLoss
        )
        
        # Ablation flag for phase 2 training (disables social group head)
        self.ablate_social_group = False
        
        # Mapping labels to heads for dynamic task management
        self.num_emotions = num_emotions
        self.num_social_groups = num_social_groups

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        batch: Optional[Dict[str, torch.Tensor]] = None,
        return_all_outputs: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the MTL model.
        
        Args:
           input_ids: Token IDs of shape (batch_size, seq_len)
           attention_mask: Attention mask of shape (batch_size, seq_len)
           batch: Alternative way to pass inputs as a dictionary (contains 'input_ids', 'attention_mask')
           return_all_outputs: If True, return hidden states for analysis
           
        Returns:
           Dictionary containing predictions for all 3 tasks.
        """
        # Handle batch dictionary if provided
        if batch is not None:
            input_ids = batch.get("input_ids", input_ids)
            attention_mask = batch.get("attention_mask", attention_mask)

        if input_ids is None or attention_mask is None:
            raise ValueError("Must provide either input_ids/attention_mask or a batch dictionary.")
        # Restore batch dimension if dropped
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        
        # Get embeddings
        embedding_output = self.encoder.embeddings(
            input_ids=input_ids,
            position_ids=None,
            token_type_ids=None,  # RoBERTa doesn't use token type embeddings
            inputs_embeds=None,
            past_key_values=None,
        )
        
        # Extend attention mask for transformer
        extended_attention_mask = self.encoder.get_extended_attention_mask(
            attention_mask, input_ids.shape
        )
        
        # Pass through custom MTL encoder (layers 1-11 shared, layer 12 split)
        encoder_outputs = self.encoder.encoder(
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
        
        # Extract hidden states for each task from the last layer
        # encoder_outputs[0] is ((hidden_bias, hidden_emotion, hidden_social),)
        hidden_states_tuple = encoder_outputs[0]
        hidden_bias = hidden_states_tuple[0]  # (batch, seq_len, hidden)
        hidden_emotion = hidden_states_tuple[1]
        hidden_social = hidden_states_tuple[2]
        
        # Pool to get [CLS] token representation for each task
        pooled_bias = self.encoder.pooler(hidden_bias, task="bias")
        pooled_emotion = self.encoder.pooler(hidden_emotion, task="emotion")
        
        if not self.ablate_social_group:
            pooled_social = self.encoder.pooler(hidden_social, task="social")
        else:
            # During ablation, use bias representation for social head
            # but the head will be disabled in loss computation
            pooled_social = None
        
        # Task-specific predictions
        bias_score = self.bias_head(pooled_bias)  # (batch, 1)
        emotion_logits = self.emotion_head(pooled_emotion)  # (batch, num_emotions)
        
        if not self.ablate_social_group:
            social_logits = self.social_group_head(pooled_social)  # (batch, num_social_groups)
        else:
            # Return zeros during ablation
            social_logits = torch.zeros(
                input_ids.size(0), 
                self.social_group_head[-1].out_features,
                device=input_ids.device
            )
        
        outputs = {
            "bias_score": bias_score,
            "emotions": emotion_logits,
            "social_group": social_logits,
        }
        
        if return_all_outputs:
            outputs["hidden_states"] = hidden_states_tuple
            outputs["pooled_bias"] = pooled_bias
            outputs["pooled_emotion"] = pooled_emotion
            if pooled_social is not None:
                outputs["pooled_social"] = pooled_social
        
        return outputs
    
    def set_ablation_mode(self, ablate: bool = True):
        """
        Enable/disable ablation mode for the social group head.
        
        Args:
            ablate: If True, disables the social group head for phase 2 training
        """
        self.ablate_social_group = ablate
        # Freeze/unfreeze social group head parameters
        for param in self.social_group_head.parameters():
            param.requires_grad = not ablate
    
    def get_shared_encoder_params(self) -> List[nn.Parameter]:
        """Get parameters for shared layers 1-11 (for differential learning rates)."""
        return list(self.encoder.encoder.layer.parameters())
    
    def get_task_specific_params(self, task: str = "all") -> List[nn.Parameter]:
        """Get task-specific parameters."""
        params = []
        if task in ["bias", "all"]:
            params.extend(self.bias_head.parameters())
        if task in ["emotion", "all"]:
            params.extend(self.emotion_head.parameters())
        if task in ["social", "all"] and not self.ablate_social_group:
            params.extend(self.social_group_head.parameters())
        # Include layer 12 split parameters
        params.extend(self.encoder.encoder.get_split_layer_params(task))
        return params


class RoBERTaMTLEncoder(nn.Module):
    """
    Custom RoBERTa encoder that splits at layer 12 into 3 task-specific paths.
    Layers 1-11 are shared, layer 12 is split for bias, emotion, and social group tasks.
    """
    
    def __init__(self, config, layers):
        super().__init__()
        self.config = config
        
        # Shared layers 1-11 (all but the last layer)
        # RoBERTa-base has 12 layers (0-11), so we take first 11 (0-10)
        self.shared_layers = nn.ModuleList(layers[:-1])
        
        # Split the 12th layer (index 11) into 3 task-specific copies
        last_layer = layers[-1]
        self.layer_bias = copy.deepcopy(last_layer)
        self.layer_emotion = copy.deepcopy(last_layer)
        self.layer_social = copy.deepcopy(last_layer)
        
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        # Pass through shared layers 1-11
        for i, layer_module in enumerate(self.shared_layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            hidden_states = self._layer_forward(
                hidden_states, layer_module, layer_head_mask,
                attention_mask, past_key_value
            )
        
        # Split at layer 12 into 3 task-specific paths
        last_idx = len(self.shared_layers)
        layer_head_mask = head_mask[last_idx] if head_mask is not None else None
        past_key_value_last = past_key_values[last_idx] if past_key_values is not None else None
        
        # Task-specific transformations at layer 12
        hidden_bias = self._layer_forward(
            hidden_states, self.layer_bias, layer_head_mask,
            attention_mask, past_key_value_last
        )
        hidden_emotion = self._layer_forward(
            hidden_states, self.layer_emotion, layer_head_mask,
            attention_mask, past_key_value_last
        )
        hidden_social = self._layer_forward(
            hidden_states, self.layer_social, layer_head_mask,
            attention_mask, past_key_value_last
        )
        
        # Return tuple of task-specific hidden states
        return ((hidden_bias, hidden_emotion, hidden_social),)
    
    def _layer_forward(self, hidden_states, layer_module, layer_head_mask,
                       attention_mask, past_key_value):
        """Process a single transformer layer."""
        layer_outputs = layer_module(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=layer_head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=past_key_value,
            output_attentions=False,
        )
        
        # Extract hidden states from layer output (first element of tuple)
        if isinstance(layer_outputs, tuple):
            return layer_outputs[0]
        return layer_outputs
    
    def get_split_layer_params(self, task: str = "all") -> List[nn.Parameter]:
        """Get parameters for the split layer 12."""
        params = []
        if task in ["bias", "all"]:
            params.extend(self.layer_bias.parameters())
        if task in ["emotion", "all"]:
            params.extend(self.layer_emotion.parameters())
        if task in ["social", "all"]:
            params.extend(self.layer_social.parameters())
        return params


class RoBERTaMTLPooler(nn.Module):
    """
    Pooler that handles task-specific [CLS] token representations.
    """
    
    def __init__(self, config, dense=None):
        super().__init__()
        self.config = config
        
        if dense is not None:
            # Use provided dense layer as template
            self.dense_bias = copy.deepcopy(dense)
            self.dense_emotion = copy.deepcopy(dense)
            self.dense_social = copy.deepcopy(dense)
        else:
            # Create new dense layers
            self.dense_bias = nn.Linear(config.hidden_size, config.hidden_size)
            self.dense_emotion = nn.Linear(config.hidden_size, config.hidden_size)
            self.dense_social = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.activation = nn.Tanh()
    
    def forward(self, hidden_states, task: str = "bias"):
        """
        Pool the [CLS] token representation.
        
        Args:
            hidden_states: Hidden states tensor (batch, seq_len, hidden)
            task: Which task's pooler to use ('bias', 'emotion', or 'social')
        """
        # Take [CLS] token (first position)
        first_token_tensor = hidden_states[:, 0]
        
        # Select appropriate dense layer
        if task == "bias":
            pooled = self.dense_bias(first_token_tensor)
        elif task == "emotion":
            pooled = self.dense_emotion(first_token_tensor)
        elif task == "social":
            pooled = self.dense_social(first_token_tensor)
        else:
            raise ValueError(f"Unknown task: {task}")
        
        return self.activation(pooled)


if __name__ == "__main__":
    # Quick test of the model
    print("Testing RoBERTaMTL model...")
    
    model = RoBERTaMTL(
        model_name="roberta-base",
        num_emotions=13,
        num_social_groups=10,
        extra_dropout=0.1,
    )
    
    # Test forward pass
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    text = ["This is a test sentence.", "Another test sentence."]
    inputs = tokenizer(text, padding=True, return_tensors="pt")
    
    outputs = model(inputs["input_ids"], inputs["attention_mask"])
    
    print(f"Bias score shape: {outputs['bias_score'].shape}")
    print(f"Emotion logits shape: {outputs['emotions'].shape}")
    print(f"Social group logits shape: {outputs['social_group'].shape}")
    print(f"Bias scores: {outputs['bias_score']}")
    
    # Test ablation mode
    model.set_ablation_mode(True)
    outputs_ablated = model(inputs["input_ids"], inputs["attention_mask"])
    print(f"\nWith ablation - Social group zeros: {outputs_ablated['social_group'].sum().item() == 0}")
    
    print("\nModel test passed!")
