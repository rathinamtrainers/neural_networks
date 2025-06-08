"""
Lesson 2: Multi-Head Attention
==============================

Multi-head attention allows the model to attend to information from different 
representation subspaces at different positions simultaneously.

Key concepts:
- Multiple attention heads running in parallel
- Each head learns different types of relationships
- Concatenate and project the outputs
- Scaled dot-product attention with efficiency improvements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention implementation"""
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, query, key, value, mask=None):
        """
        Scaled dot-product attention
        
        Args:
            query: (batch_size, num_heads, seq_len, d_k)
            key: (batch_size, num_heads, seq_len, d_k)
            value: (batch_size, num_heads, seq_len, d_k)
            mask: optional mask to prevent attention to certain positions
        """
        d_k = query.size(-1)
        
        # Calculate attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query, key, value: (batch_size, seq_len, d_model)
            mask: optional attention mask
        """
        batch_size, seq_len, d_model = query.size()
        
        # 1. Linear projections and reshape for multi-head
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 3. Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # 4. Final linear projection
        output = self.w_o(attention_output)
        
        return output, attention_weights

class SelfAttention(nn.Module):
    """Self-attention: Q, K, V all come from the same input"""
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        return self.multi_head_attention(x, x, x, mask)

def visualize_multi_head_attention(attention_weights, tokens=None, num_heads_to_show=4):
    """Visualize attention weights for multiple heads"""
    batch_size, num_heads, seq_len, _ = attention_weights.shape
    
    # Show first few heads
    heads_to_show = min(num_heads_to_show, num_heads)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i in range(heads_to_show):
        attn = attention_weights[0, i].detach().numpy()
        
        im = axes[i].imshow(attn, cmap='Blues', aspect='auto')
        axes[i].set_title(f'Head {i+1}')
        axes[i].set_xlabel('Key Position')
        axes[i].set_ylabel('Query Position')
        
        if tokens:
            axes[i].set_xticks(range(len(tokens)))
            axes[i].set_yticks(range(len(tokens)))
            axes[i].set_xticklabels(tokens, rotation=45)
            axes[i].set_yticklabels(tokens)
        
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.show()

def create_padding_mask(seq_len, actual_lengths):
    """Create mask to ignore padding tokens"""
    batch_size = len(actual_lengths)
    mask = torch.zeros(batch_size, seq_len, seq_len)
    
    for i, length in enumerate(actual_lengths):
        mask[i, :length, :length] = 1
    
    return mask

if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Example 1: Basic Multi-Head Attention
    print("=== Example 1: Multi-Head Attention ===")
    
    d_model = 512
    num_heads = 8
    seq_len = 6
    batch_size = 2
    
    # Create input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Create multi-head attention
    mha = MultiHeadAttention(d_model, num_heads)
    
    # Forward pass
    output, attention_weights = mha(x, x, x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in mha.parameters())}")
    
    # Example 2: Self-Attention with different inputs
    print("\n=== Example 2: Self-Attention ===")
    
    self_attention = SelfAttention(d_model=64, num_heads=4)
    
    # Create a simple sequence
    seq_len = 5
    x_simple = torch.randn(1, seq_len, 64)
    
    output_self, attn_weights_self = self_attention(x_simple)
    print(f"Self-attention output shape: {output_self.shape}")
    print(f"Self-attention weights shape: {attn_weights_self.shape}")
    
    # Example 3: Attention with masking
    print("\n=== Example 3: Attention with Padding Mask ===")
    
    # Simulate sequences of different lengths
    seq_len = 4
    actual_lengths = [3, 2]  # Actual lengths without padding
    
    x_padded = torch.randn(2, seq_len, 64)
    padding_mask = create_padding_mask(seq_len, actual_lengths)
    
    output_masked, attn_weights_masked = self_attention(x_padded, mask=padding_mask)
    
    print("Attention weights with padding mask (first batch):")
    print(attn_weights_masked[0, 0])  # First head of first batch
    
    # Example 4: Visualize different attention heads
    print("\n=== Example 4: Attention Head Visualization ===")
    
    # Create a smaller example for visualization
    tokens = ["The", "cat", "sat", "mat"]
    d_model_vis = 32
    mha_vis = MultiHeadAttention(d_model_vis, num_heads=4)
    
    x_vis = torch.randn(1, len(tokens), d_model_vis)
    _, attn_vis = mha_vis(x_vis, x_vis, x_vis)
    
    visualize_multi_head_attention(attn_vis, tokens)
    
    # Example 5: Compare single-head vs multi-head
    print("\n=== Example 5: Single-head vs Multi-head Comparison ===")
    
    # Single head (essentially regular attention)
    single_head = MultiHeadAttention(d_model=64, num_heads=1)
    multi_head = MultiHeadAttention(d_model=64, num_heads=8)
    
    x_compare = torch.randn(1, 4, 64)
    
    out_single, attn_single = single_head(x_compare, x_compare, x_compare)
    out_multi, attn_multi = multi_head(x_compare, x_compare, x_compare)
    
    print(f"Single-head attention weights shape: {attn_single.shape}")
    print(f"Multi-head attention weights shape: {attn_multi.shape}")
    
    # Show that multi-head can capture different patterns
    print("\nAttention diversity across heads:")
    for i in range(min(4, attn_multi.shape[1])):
        head_attn = attn_multi[0, i, 0, :]  # First query position
        print(f"Head {i+1}: {head_attn.detach().numpy()}")