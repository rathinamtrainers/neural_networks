"""
Lesson 4: Transformer Encoder Block
===================================

The transformer encoder block combines multiple components:
1. Multi-head self-attention
2. Add & Norm (residual connection + layer normalization)
3. Feed-forward network (MLP)
4. Another Add & Norm

Key concepts:
- Residual connections for gradient flow
- Layer normalization for training stability
- Feed-forward network for non-linear transformations
- Pre-norm vs post-norm arrangements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from lessons.multi_head_attention import MultiHeadAttention
from lessons.positional_encoding import SinusoidalPositionalEncoding

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network (MLP)
    Typically: Linear -> ReLU -> Linear with expansion factor of 4
    """
    
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model  # Standard expansion factor
            
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerEncoderBlock(nn.Module):
    """
    Single transformer encoder block with post-norm arrangement
    (original Transformer paper arrangement)
    """
    
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1):
        super().__init__()
        
        # Multi-head attention
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: attention mask
        """
        # Self-attention with residual connection and layer norm
        attn_output, attn_weights = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attn_weights

class PreNormTransformerEncoderBlock(nn.Module):
    """
    Transformer encoder block with pre-norm arrangement
    (often used in modern implementations for better training)
    """
    
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """Pre-norm: normalize before sub-layers"""
        # Pre-norm self-attention
        normed_x = self.norm1(x)
        attn_output, attn_weights = self.self_attention(normed_x, normed_x, normed_x, mask)
        x = x + self.dropout(attn_output)
        
        # Pre-norm feed-forward
        normed_x = self.norm2(x)
        ff_output = self.feed_forward(normed_x)
        x = x + self.dropout(ff_output)
        
        return x, attn_weights

class SimpleTransformerEncoder(nn.Module):
    """
    Simple transformer encoder with multiple blocks
    """
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff=None, 
                 max_seq_len=5000, dropout=0.1, pre_norm=False):
        super().__init__()
        
        self.d_model = d_model
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len)
        
        # Transformer blocks
        block_class = PreNormTransformerEncoderBlock if pre_norm else TransformerEncoderBlock
        self.blocks = nn.ModuleList([
            block_class(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm (for pre-norm arrangement)
        self.final_norm = nn.LayerNorm(d_model) if pre_norm else None
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len) - token indices
            mask: attention mask
        """
        # Token embeddings
        x = self.token_embedding(x) * (self.d_model ** 0.5)  # Scale embeddings
        
        # Add positional encoding
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Pass through transformer blocks
        attention_weights = []
        for block in self.blocks:
            x, attn_weights = block(x, mask)
            attention_weights.append(attn_weights)
        
        # Final normalization (for pre-norm)
        if self.final_norm is not None:
            x = self.final_norm(x)
            
        return x, attention_weights

def create_causal_mask(seq_len):
    """Create causal mask to prevent looking at future tokens"""
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions

def analyze_attention_patterns(attention_weights, tokens=None):
    """Analyze attention patterns across layers"""
    num_layers = len(attention_weights)
    num_heads = attention_weights[0].shape[1]
    
    print(f"Analyzing attention across {num_layers} layers, {num_heads} heads each")
    
    # Average attention entropy per layer (measure of attention distribution)
    entropies = []
    for layer_idx, attn in enumerate(attention_weights):
        # Calculate entropy for each attention head
        attn_probs = attn[0]  # First batch
        layer_entropies = []
        
        for head in range(num_heads):
            head_attn = attn_probs[head]
            # Calculate entropy for each query position
            head_entropy = -torch.sum(head_attn * torch.log(head_attn + 1e-9), dim=-1)
            layer_entropies.append(head_entropy.mean().item())
        
        avg_entropy = sum(layer_entropies) / len(layer_entropies)
        entropies.append(avg_entropy)
        print(f"Layer {layer_idx + 1} average attention entropy: {avg_entropy:.3f}")
    
    return entropies

if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Example 1: Single Encoder Block
    print("=== Example 1: Single Encoder Block ===")
    
    d_model = 256
    num_heads = 8
    seq_len = 10
    batch_size = 2
    
    # Create input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Create encoder block
    encoder_block = TransformerEncoderBlock(d_model, num_heads)
    
    # Forward pass
    output, attn_weights = encoder_block(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in encoder_block.parameters())}")
    
    # Example 2: Compare Post-Norm vs Pre-Norm
    print("\n=== Example 2: Post-Norm vs Pre-Norm ===")
    
    post_norm_block = TransformerEncoderBlock(d_model, num_heads)
    pre_norm_block = PreNormTransformerEncoderBlock(d_model, num_heads)
    
    # Same input
    x_test = torch.randn(1, seq_len, d_model)
    
    out_post, _ = post_norm_block(x_test)
    out_pre, _ = pre_norm_block(x_test)
    
    print(f"Post-norm output range: [{out_post.min():.3f}, {out_post.max():.3f}]")
    print(f"Pre-norm output range: [{out_pre.min():.3f}, {out_pre.max():.3f}]")
    print(f"Post-norm output std: {out_post.std():.3f}")
    print(f"Pre-norm output std: {out_pre.std():.3f}")
    
    # Example 3: Complete Transformer Encoder
    print("\n=== Example 3: Complete Transformer Encoder ===")
    
    vocab_size = 1000
    d_model = 128
    num_heads = 4
    num_layers = 3
    seq_len = 8
    
    # Create encoder
    encoder = SimpleTransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model, 
        num_heads=num_heads,
        num_layers=num_layers
    )
    
    # Create input tokens
    input_tokens = torch.randint(0, vocab_size, (2, seq_len))
    
    # Forward pass
    encoded, all_attention_weights = encoder(input_tokens)
    
    print(f"Input tokens shape: {input_tokens.shape}")
    print(f"Encoded output shape: {encoded.shape}")
    print(f"Number of attention weight tensors: {len(all_attention_weights)}")
    print(f"Total parameters: {sum(p.numel() for p in encoder.parameters())}")
    
    # Example 4: Attention with Masking
    print("\n=== Example 4: Attention with Masking ===")
    
    # Create padding mask (simulate variable length sequences)
    seq_len = 6
    actual_lengths = [4, 3]  # Actual sequence lengths
    batch_size = len(actual_lengths)
    
    # Create mask
    mask = torch.zeros(batch_size, seq_len, seq_len)
    for i, length in enumerate(actual_lengths):
        mask[i, :length, :length] = 1
    
    # Create input with padding
    input_with_padding = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward with mask
    encoded_masked, attn_masked = encoder(input_with_padding, mask=mask)
    
    print("Attention weights with masking (first sequence, first head, layer 1):")
    print(attn_masked[0][0, 0])  # First layer, first batch, first head
    
    # Example 5: Analyze Attention Patterns
    print("\n=== Example 5: Attention Pattern Analysis ===")
    
    # Create simple sequence for analysis
    simple_tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])  # Single sequence
    encoded_simple, attn_simple = encoder(simple_tokens)
    
    entropies = analyze_attention_patterns(attn_simple)
    
    # Visualize attention evolution across layers
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(entropies) + 1), entropies, 'o-')
    plt.xlabel('Layer')
    plt.ylabel('Average Attention Entropy')
    plt.title('Attention Entropy Across Transformer Layers')
    plt.grid(True)
    plt.show()
    
    # Example 6: Causal Masking (for comparison with decoder)
    print("\n=== Example 6: Causal Masking Example ===")
    
    causal_mask = create_causal_mask(seq_len)
    print(f"Causal mask shape: {causal_mask.shape}")
    print("Causal mask (prevents looking at future positions):")
    print(causal_mask[0, 0])
    
    # Apply causal mask
    encoded_causal, attn_causal = encoder(simple_tokens, mask=causal_mask)
    
    print("Attention with causal mask (first head, layer 1):")
    print(attn_causal[0][0, 0])
    
    print("\nTransformer encoder block combines attention, normalization, and feed-forward layers!")
    print("Next: We'll stack multiple blocks to create a full transformer encoder.")