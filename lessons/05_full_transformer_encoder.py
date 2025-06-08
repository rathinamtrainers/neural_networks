"""
Lesson 5: Full Transformer Encoder
==================================

Building a complete transformer encoder by stacking multiple encoder blocks.
This creates the encoder used in models like BERT for tasks like:
- Text classification
- Named entity recognition  
- Question answering
- Feature extraction

Key concepts:
- Stacking multiple transformer blocks
- Different output strategies (CLS token, pooling, etc.)
- Practical applications and use cases
- Memory and computational considerations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from lessons.transformer_encoder_block import TransformerEncoderBlock, PreNormTransformerEncoderBlock
from lessons.positional_encoding import SinusoidalPositionalEncoding

class TransformerEncoder(nn.Module):
    """
    Complete Transformer Encoder (like BERT encoder)
    """
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff=None,
                 max_seq_len=512, dropout=0.1, pre_norm=False, pad_token_id=0):
        super().__init__()
        
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len)
        
        # Transformer blocks
        block_class = PreNormTransformerEncoderBlock if pre_norm else TransformerEncoderBlock
        self.blocks = nn.ModuleList([
            block_class(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm (especially important for pre-norm)
        self.final_norm = nn.LayerNorm(d_model) if pre_norm else None
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights following best practices"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)
    
    def create_padding_mask(self, input_ids):
        """Create mask to ignore padding tokens"""
        return (input_ids != self.pad_token_id).unsqueeze(1).unsqueeze(2)
    
    def forward(self, input_ids, attention_mask=None, return_attention_weights=False):
        """
        Args:
            input_ids: (batch_size, seq_len) - token indices
            attention_mask: (batch_size, seq_len) - 1 for real tokens, 0 for padding
            return_attention_weights: whether to return attention weights
        """
        batch_size, seq_len = input_ids.shape
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = self.create_padding_mask(input_ids)
        else:
            # Reshape provided mask for multi-head attention
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Token embeddings (scaled)
        embeddings = self.token_embedding(input_ids) * (self.d_model ** 0.5)
        
        # Add positional encoding
        x = self.positional_encoding(embeddings)
        x = self.dropout(x)
        
        # Pass through transformer blocks
        all_attention_weights = []
        for block in self.blocks:
            x, attention_weights = block(x, attention_mask)
            if return_attention_weights:
                all_attention_weights.append(attention_weights)
        
        # Final normalization
        if self.final_norm is not None:
            x = self.final_norm(x)
        
        if return_attention_weights:
            return x, all_attention_weights
        return x

class BERTLikeModel(nn.Module):
    """
    BERT-like model with classification head
    """
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, num_classes,
                 d_ff=None, max_seq_len=512, dropout=0.1, cls_token_id=101):
        super().__init__()
        
        self.cls_token_id = cls_token_id
        
        # Transformer encoder
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            pre_norm=True  # Use pre-norm for better training
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
    def forward(self, input_ids, attention_mask=None):
        """
        For classification, use [CLS] token representation
        """
        # Get encoder output
        encoder_output = self.encoder(input_ids, attention_mask)
        
        # Use [CLS] token (first token) for classification
        cls_representation = encoder_output[:, 0, :]  # (batch_size, d_model)
        
        # Classification
        logits = self.classifier(cls_representation)
        
        return logits

class SequenceTaggingModel(nn.Module):
    """
    Model for sequence tagging (like NER)
    """
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, num_tags,
                 d_ff=None, max_seq_len=512, dropout=0.1):
        super().__init__()
        
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        # Tag classifier for each token
        self.tag_classifier = nn.Linear(d_model, num_tags)
        
    def forward(self, input_ids, attention_mask=None):
        """
        Predict tags for each token
        """
        # Get encoder output
        encoder_output = self.encoder(input_ids, attention_mask)
        
        # Predict tag for each token
        tag_logits = self.tag_classifier(encoder_output)
        
        return tag_logits

def analyze_model_complexity(model):
    """Analyze model complexity and memory usage"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate memory usage (rough)
    param_memory = total_params * 4 / (1024**2)  # 4 bytes per float32, convert to MB
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Approximate parameter memory: {param_memory:.1f} MB")
    
    # Breakdown by module type
    embedding_params = sum(p.numel() for n, p in model.named_parameters() if 'embedding' in n)
    attention_params = sum(p.numel() for n, p in model.named_parameters() if 'attention' in n)
    ff_params = sum(p.numel() for n, p in model.named_parameters() if 'feed_forward' in n)
    
    print(f"Embedding parameters: {embedding_params:,} ({embedding_params/total_params*100:.1f}%)")
    print(f"Attention parameters: {attention_params:,} ({attention_params/total_params*100:.1f}%)")
    print(f"Feed-forward parameters: {ff_params:,} ({ff_params/total_params*100:.1f}%)")

def visualize_attention_heads(attention_weights, layer_idx=0, tokens=None):
    """Visualize attention patterns across different heads"""
    # Take specific layer
    layer_attention = attention_weights[layer_idx][0]  # First batch
    num_heads = layer_attention.shape[0]
    
    # Create subplot for each head
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for head_idx in range(min(8, num_heads)):
        attn = layer_attention[head_idx].detach().numpy()
        
        im = axes[head_idx].imshow(attn, cmap='Blues', aspect='auto')
        axes[head_idx].set_title(f'Head {head_idx + 1}')
        
        if tokens:
            axes[head_idx].set_xticks(range(len(tokens)))
            axes[head_idx].set_yticks(range(len(tokens)))
            axes[head_idx].set_xticklabels(tokens, rotation=45)
            axes[head_idx].set_yticklabels(tokens)
        
        plt.colorbar(im, ax=axes[head_idx])
    
    plt.suptitle(f'Attention Patterns - Layer {layer_idx + 1}')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Example 1: Basic Transformer Encoder
    print("=== Example 1: Basic Transformer Encoder ===")
    
    vocab_size = 5000
    d_model = 256
    num_heads = 8
    num_layers = 6
    seq_len = 32
    batch_size = 4
    
    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        pre_norm=True
    )
    
    # Create sample input
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))  # Avoid pad token (0)
    
    # Forward pass
    output = encoder(input_ids)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {output.shape}")
    
    analyze_model_complexity(encoder)
    
    # Example 2: BERT-like Classification Model
    print("\n=== Example 2: BERT-like Classification ===")
    
    num_classes = 3  # e.g., sentiment analysis
    bert_model = BERTLikeModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        num_classes=num_classes
    )
    
    # Add [CLS] token to beginning of sequences
    cls_token_id = 101
    input_with_cls = torch.cat([
        torch.full((batch_size, 1), cls_token_id),
        input_ids[:, :-1]  # Remove last token to keep same length
    ], dim=1)
    
    # Classification
    logits = bert_model(input_with_cls)
    
    print(f"Classification logits shape: {logits.shape}")
    print(f"Predicted classes: {torch.argmax(logits, dim=-1)}")
    
    # Example 3: Sequence Tagging Model
    print("\n=== Example 3: Sequence Tagging (NER) ===")
    
    num_tags = 9  # e.g., BIO tags for NER
    tagging_model = SequenceTaggingModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        num_tags=num_tags
    )
    
    # Tag prediction
    tag_logits = tagging_model(input_ids)
    predicted_tags = torch.argmax(tag_logits, dim=-1)
    
    print(f"Tag logits shape: {tag_logits.shape}")
    print(f"Predicted tags shape: {predicted_tags.shape}")
    print(f"Sample predicted tags: {predicted_tags[0]}")
    
    # Example 4: Attention Analysis
    print("\n=== Example 4: Attention Pattern Analysis ===")
    
    # Get attention weights
    sample_input = input_ids[:1]  # Single sequence
    output_with_attn, attention_weights = encoder(sample_input, return_attention_weights=True)
    
    print(f"Number of layers: {len(attention_weights)}")
    print(f"Attention shape per layer: {attention_weights[0].shape}")
    
    # Analyze attention patterns
    tokens = [f"token_{i}" for i in range(seq_len)]
    visualize_attention_heads(attention_weights, layer_idx=0, tokens=tokens[:8])
    
    # Example 5: Different Pooling Strategies
    print("\n=== Example 5: Different Output Pooling Strategies ===")
    
    encoder_output = encoder(input_ids)
    
    # 1. Mean pooling (average all tokens)
    mean_pooled = torch.mean(encoder_output, dim=1)
    
    # 2. Max pooling 
    max_pooled, _ = torch.max(encoder_output, dim=1)
    
    # 3. CLS token (first token)
    cls_pooled = encoder_output[:, 0, :]
    
    # 4. Attention-weighted pooling
    attention_weights_last = attention_weights[-1]  # Last layer
    avg_attention = torch.mean(attention_weights_last, dim=1)  # Average across heads
    attention_scores = torch.mean(avg_attention, dim=1)  # Average attention from all positions
    attention_weights_norm = F.softmax(attention_scores, dim=-1).unsqueeze(-1)
    attention_pooled = torch.sum(encoder_output * attention_weights_norm, dim=1)
    
    print(f"Mean pooled shape: {mean_pooled.shape}")
    print(f"Max pooled shape: {max_pooled.shape}")
    print(f"CLS pooled shape: {cls_pooled.shape}")
    print(f"Attention pooled shape: {attention_pooled.shape}")
    
    # Example 6: Handling Variable Length Sequences
    print("\n=== Example 6: Variable Length Sequences ===")
    
    # Create sequences with different lengths
    seq_lengths = [10, 15, 8, 12]
    max_len = max(seq_lengths)
    
    # Create padded input
    padded_input = torch.zeros(len(seq_lengths), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(seq_lengths), max_len)
    
    for i, length in enumerate(seq_lengths):
        padded_input[i, :length] = torch.randint(1, vocab_size, (length,))
        attention_mask[i, :length] = 1
    
    # Process with attention mask
    masked_output = encoder(padded_input, attention_mask)
    
    print(f"Padded input shape: {padded_input.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    print(f"Masked output shape: {masked_output.shape}")
    print(f"Sequence lengths: {seq_lengths}")
    
    # Show that padding positions are properly masked
    print("First sequence (length 10):")
    print(f"Real tokens: {padded_input[0, :seq_lengths[0]]}")
    print(f"Padding: {padded_input[0, seq_lengths[0]:]}")
    
    print("\nTransformer encoder is complete! It can now handle various NLP tasks.")
    print("Next: We'll build the decoder for sequence generation tasks.")