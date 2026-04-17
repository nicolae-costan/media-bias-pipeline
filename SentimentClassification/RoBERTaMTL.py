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
    and 3 task-specific branches at layer 12.
    """
    
    def __init__(
        self,
        model_name: str = "roberta-base",
        num_emotions: int = 13,
        num_social_groups: int = 10,
        extra_dropout: float = 0.0,
        loss_weights: Optional[Dict[str, float]] = None,
    ):
        super(RoBERTaMTL, self).__init__()
        
        # 1. Configuration & Dropout Modification
        config = AutoConfig.from_pretrained(model_name)
        if extra_dropout > 0:
            config.hidden_dropout_prob += extra_dropout
            config.attention_probs_dropout_prob += extra_dropout
        
        self.config = config
        self.loss_weights = loss_weights or {"bias": 1.0, "emotion": 1.0, "social": 1.0}
        
        # 2. Native Encoder Initialization (Layers 1-12)
        base_model = AutoModel.from_pretrained(model_name, config=config)
        self.encoder = base_model
        
        # 3. Branching Layer 12
        # Extract the 12th layer (index 11) and duplicate it for each task head
        layer12 = self.encoder.encoder.layer[11]
        self.layer_bias = copy.deepcopy(layer12)
        self.layer_emotion = copy.deepcopy(layer12)
        self.layer_social = copy.deepcopy(last_layer := layer12) # syntactic sugar for readability
        
        # 4. Truncate Shared Encoder (Layers 1-11)
        # This allows calling self.encoder() to process the first 11 layers natively
        self.encoder.encoder.layer = nn.ModuleList(self.encoder.encoder.layer[:11])
        
        # 5. Task-specific Classification Heads
        hidden_size = config.hidden_size
        
        # Primary Head: Bias Score Regression
        self.bias_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )
        
        # Auxiliary Head 1: Emotion Prediction (Multi-label)
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(hidden_size // 2, num_emotions),
        )
        
        # Auxiliary Head 2: Social Group Prediction (Classification)
        self.social_group_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(hidden_size // 2, num_social_groups),
        )
        
        # Initialization
        self._init_custom_weights()
        
        self.ablate_social_group = False
        self.num_emotions = num_emotions
        self.num_social_groups = num_social_groups

    def _init_custom_weights(self):
        """Apply RoBERTa's native weight initialization to new components."""
        for module in [self.layer_bias, self.layer_emotion, self.layer_social]:
            self.encoder._init_weights(module)
        
        for head in [self.bias_head, self.emotion_head, self.social_group_head]:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    self.encoder._init_weights(m)

    def _mean_pooling(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Correctly average embeddings while ignoring padding tokens."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels_bias: Optional[torch.Tensor] = None,
        labels_emotion: Optional[torch.Tensor] = None,
        labels_social: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        
        # Pass through shared layers 1-11
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        shared_hidden = outputs.last_hidden_state
        
        # Extend attention mask for the 12th layer manually if needed
        # (AutoModel usually handles this, but since we are calling the layer directly...)
        extended_mask = self.encoder.get_extended_attention_mask(attention_mask, input_ids.shape)
        
        # Process task-specific 12th layers
        bias_out = self.layer_bias(shared_hidden, attention_mask=extended_mask)
        emotion_out = self.layer_emotion(shared_hidden, attention_mask=extended_mask)
        social_out = self.layer_social(shared_hidden, attention_mask=extended_mask)
        
        # Handle cases where the layer might return a tuple or a raw tensor
        hidden_bias = bias_out[0] if isinstance(bias_out, tuple) else bias_out
        hidden_emotion = emotion_out[0] if isinstance(emotion_out, tuple) else emotion_out
        hidden_social = social_out[0] if isinstance(social_out, tuple) else social_out
        
        # Mean Pooling
        pooled_bias = self._mean_pooling(hidden_bias, attention_mask)
        pooled_emotion = self._mean_pooling(hidden_emotion, attention_mask)
        pooled_social = self._mean_pooling(hidden_social, attention_mask)
        
        # Heads
        bias_score = self.bias_head(pooled_bias)
        emotion_logits = self.emotion_head(pooled_emotion)
        social_logits = self.social_group_head(pooled_social)
        
        results = {
            "bias_score": bias_score,
            "emotions": emotion_logits,
            "social_group": social_logits,
        }
        
        # Internal Loss Computation
        loss = 0
        if labels_bias is not None:
            loss_bias = nn.MSELoss()(bias_score.squeeze(), labels_bias)
            loss += self.loss_weights["bias"] * loss_bias
            results["loss_bias"] = loss_bias
            
        if labels_emotion is not None:
            loss_emotion = nn.BCEWithLogitsLoss()(emotion_logits, labels_emotion)
            loss += self.loss_weights["emotion"] * loss_emotion
            results["loss_emotion"] = loss_emotion
            
        if labels_social is not None and not self.ablate_social_group:
            loss_social = nn.CrossEntropyLoss()(social_logits, labels_social.long())
            loss += self.loss_weights["social"] * loss_social
            results["loss_social"] = loss_social
        elif self.ablate_social_group:
            # During ablation, zero out social logits to prevent accidental training
            results["social_group"] = torch.zeros_like(social_logits)
            
        if labels_bias is not None or labels_emotion is not None:
            results["loss"] = loss
            
        return results

    def set_ablation_mode(self, ablate: bool = True):
        self.ablate_social_group = ablate
        for param in self.social_group_head.parameters():
            param.requires_grad = not ablate
        for param in self.layer_social.parameters():
            param.requires_grad = not ablate

    
    def get_shared_encoder_params(self) -> List[nn.Parameter]:
        """Get parameters for shared layers 1-11 (for differential learning rates)."""
        return list(self.encoder.encoder.layer.parameters())
    
    def get_task_specific_params(self, task: str = "all") -> List[nn.Parameter]:
        """Get task-specific parameters."""
        params = []
        if task in ["bias", "all"]:
            params.extend(self.bias_head.parameters())
            params.extend(self.layer_bias.parameters())
        if task in ["emotion", "all"]:
            params.extend(self.emotion_head.parameters())
            params.extend(self.layer_emotion.parameters())
        if task in ["social", "all"] and not self.ablate_social_group:
            params.extend(self.social_group_head.parameters())
            params.extend(self.layer_social.parameters())
        return params


if __name__ == "__main__":
    # Quick test of the model
    print("Testing refactored RoBERTaMTL model...")
    
    model = RoBERTaMTL(
        model_name="roberta-base",
        num_emotions=13,
        num_social_groups=10,
        extra_dropout=0.1,
    )
    
    # Test forward pass with dummy data
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    text = ["This is a test sentence.", "Another test sentence."]
    inputs = tokenizer(text, padding=True, return_tensors="pt")
    
    # Create dummy labels
    dummy_labels = {
        "labels_bias": torch.tensor([0.5, 0.8]),
        "labels_emotion": torch.zeros((2, 13)),
        "labels_social": torch.tensor([1, 2]),
    }
    
    outputs = model(inputs["input_ids"], inputs["attention_mask"], **dummy_labels)
    
    print(f"Bias score shape: {outputs['bias_score'].shape}")
    print(f"Emotion logits shape: {outputs['emotions'].shape}")
    print(f"Social group logits shape: {outputs['social_group'].shape}")
    print(f"Total Loss: {outputs.get('loss', 'N/A')}")
    
    # Test ablation mode
    print("\nTesting Ablation Mode...")
    model.set_ablation_mode(True)
    outputs_ablated = model(inputs["input_ids"], inputs["attention_mask"], **dummy_labels)
    print(f"Social group logits zeroed: {torch.all(outputs_ablated['social_group'] == 0)}")
    print(f"Loss social present: {'loss_social' in outputs_ablated}")
    
    print("\nModel test passed!")
