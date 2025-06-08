"""
Multi-Head Attention Implementation

This module implements multi-head attention, which allows the model to jointly
attend to information from different representation subspaces at different positions.
Multi-head attention is a key component of the Transformer architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class MultiHeadAttention(nn.Module):
    """
    Implements Multi-Head Attention mechanism.
    
    Instead of performing a single attention function, multi-head attention
    linearly projects the queries, keys and values h times with different,
    learned linear projections. This allows the model to jointly attend to
    information from different representation subspaces.
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Args:
            embed_dim: Total dimension of the model
            num_heads: Number of parallel attention heads
            dropout: Dropout probability
        """
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        # Linear projections for Q, K, V
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len_q, embed_dim)
            key: Key tensor of shape (batch_size, seq_len_k, embed_dim)
            value: Value tensor of shape (batch_size, seq_len_v, embed_dim)
            mask: Optional mask tensor
            
        Returns:
            output: Output tensor of shape (batch_size, seq_len_q, embed_dim)
            attention_weights: Attention weights averaged across heads
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        
        # Linear projections in batch from embed_dim => num_heads x head_dim
        Q = self.query_proj(query)
        K = self.key_proj(key)
        V = self.value_proj(value)
        
        # Reshape to (batch_size, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for num_heads
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.embed_dim
        )
        
        # Final linear projection
        output = self.out_proj(attention_output)
        
        # Average attention weights across heads for visualization
        attention_weights_avg = attention_weights.mean(dim=1)
        
        return output, attention_weights_avg


def visualize_multi_head_attention():
    """Visualizes how multi-head attention works with different heads."""
    
    torch.manual_seed(42)
    
    # Parameters
    batch_size = 1
    seq_len = 8
    embed_dim = 64
    num_heads = 4
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Initialize multi-head attention
    mha = MultiHeadAttention(embed_dim, num_heads)
    
    # Apply multi-head attention
    output, attention_weights_avg = mha(x, x, x)
    
    print("Multi-Head Attention Demonstration")
    print("=" * 50)
    print(f"Input shape: {x.shape}")
    print(f"Number of heads: {num_heads}")
    print(f"Head dimension: {embed_dim // num_heads}")
    print(f"Output shape: {output.shape}")
    print()
    
    # Visualize attention patterns for each head
    # We need to access internal attention weights per head
    with torch.no_grad():
        Q = mha.query_proj(x).view(batch_size, seq_len, num_heads, mha.head_dim).transpose(1, 2)
        K = mha.key_proj(x).view(batch_size, seq_len, num_heads, mha.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / mha.scale
        attention_per_head = F.softmax(scores, dim=-1)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot attention weights for each head
    for head_idx in range(num_heads):
        ax = axes[head_idx]
        im = ax.imshow(attention_per_head[0, head_idx].numpy(), cmap='Blues', aspect='auto')
        ax.set_title(f'Head {head_idx + 1} Attention Weights')
        ax.set_xlabel('Key Positions')
        ax.set_ylabel('Query Positions')
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Plot averaged attention weights
    ax = axes[4]
    im = ax.imshow(attention_weights_avg[0].numpy(), cmap='Greens', aspect='auto')
    ax.set_title('Averaged Attention Weights')
    ax.set_xlabel('Key Positions')
    ax.set_ylabel('Query Positions')
    ax.set_xticks(range(seq_len))
    ax.set_yticks(range(seq_len))
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Hide the last subplot
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Demonstrate cross-attention
    print("\nCross-Attention Example:")
    print("-" * 30)
    
    # Different sequence lengths for encoder and decoder
    encoder_seq_len = 10
    decoder_seq_len = 6
    
    # Encoder and decoder inputs
    encoder_output = torch.randn(batch_size, encoder_seq_len, embed_dim)
    decoder_input = torch.randn(batch_size, decoder_seq_len, embed_dim)
    
    # Apply cross-attention (decoder attends to encoder)
    cross_output, cross_attention = mha(
        query=decoder_input,
        key=encoder_output,
        value=encoder_output
    )
    
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Decoder input shape: {decoder_input.shape}")
    print(f"Cross-attention output shape: {cross_output.shape}")
    print(f"Cross-attention weights shape: {cross_attention.shape}")
    
    # Visualize cross-attention
    plt.figure(figsize=(8, 6))
    plt.imshow(cross_attention[0].numpy(), cmap='Purples', aspect='auto')
    plt.title('Cross-Attention Weights\n(Decoder attending to Encoder)')
    plt.xlabel('Encoder Positions')
    plt.ylabel('Decoder Positions')
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def test_multi_head_attention():
    """Test multi-head attention with sample data."""
    
    print("\nTesting Multi-Head Attention with Sample Data")
    print("=" * 50)
    
    # Create a simple test case
    batch_size = 2
    seq_len = 4
    embed_dim = 32
    num_heads = 4
    
    # Create input with specific patterns
    # Pattern 1: Repeating pattern
    pattern1 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    # Pattern 2: Gradient pattern
    pattern2 = torch.tensor([[0.0, 0.0], [0.33, 0.33], [0.66, 0.66], [1.0, 1.0]])
    
    # Expand patterns to full embedding dimension
    x = torch.zeros(batch_size, seq_len, embed_dim)
    x[0, :, :2] = pattern1
    x[1, :, :2] = pattern2
    
    # Add some noise to other dimensions
    x[:, :, 2:] = torch.randn(batch_size, seq_len, embed_dim - 2) * 0.1
    
    # Initialize and apply multi-head attention
    mha = MultiHeadAttention(embed_dim, num_heads, dropout=0.0)
    output, attention_weights = mha(x, x, x)
    
    print(f"Input patterns (first 2 dims):")
    print(f"Batch 1: {x[0, :, :2].tolist()}")
    print(f"Batch 2: {x[1, :, :2].tolist()}")
    print()
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Check that output maintains batch and sequence dimensions
    assert output.shape == x.shape
    assert attention_weights.shape == (batch_size, seq_len, seq_len)
    
    print("\nTest passed! Multi-head attention maintains correct dimensions.")


if __name__ == "__main__":
    visualize_multi_head_attention()
    test_multi_head_attention()