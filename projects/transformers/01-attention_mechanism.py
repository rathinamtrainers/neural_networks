"""
Basic Self-Attention Mechanism Implementation

This module demonstrates the fundamental self-attention mechanism used in transformers.
Self-attention allows each position in a sequence to attend to all positions in the 
same sequence, learning relationships between different parts of the input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns


class SelfAttention(nn.Module):
    """
    Implements the self-attention mechanism.
    
    Self-attention computes attention weights between all positions in a sequence,
    allowing the model to focus on relevant parts of the input when processing
    each position.
    """
    
    def __init__(self, embed_dim):
        """
        Args:
            embed_dim: Dimension of the input embeddings
        """
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        
        # Linear transformations for queries, keys, and values
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        # Scaling factor for dot product attention
        self.scale = torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))
        
    def forward(self, x, mask=None):
        """
        Forward pass of self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional mask tensor to prevent attention to certain positions
            
        Returns:
            output: Attention output of shape (batch_size, seq_len, embed_dim)
            attention_weights: Attention weights of shape (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute queries, keys, and values
        Q = self.query(x)  # (batch_size, seq_len, embed_dim)
        K = self.key(x)    # (batch_size, seq_len, embed_dim)
        V = self.value(x)  # (batch_size, seq_len, embed_dim)
        
        # Compute attention scores
        # Q @ K^T gives us (batch_size, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


def demonstrate_self_attention():
    """Demonstrates self-attention with a simple example."""
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create a simple example: 3 sequences of length 5 with embedding dimension 8
    batch_size = 3
    seq_len = 5
    embed_dim = 8
    
    # Create random input embeddings
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Initialize self-attention layer
    attention = SelfAttention(embed_dim)
    
    # Apply self-attention
    output, attention_weights = attention(x)
    
    print("Self-Attention Demonstration")
    print("=" * 50)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print()
    
    # Visualize attention weights for the first sequence
    plt.figure(figsize=(10, 8))
    
    # Plot attention weights heatmap
    plt.subplot(2, 2, 1)
    sns.heatmap(attention_weights[0].detach().numpy(), 
                annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=range(seq_len), 
                yticklabels=range(seq_len))
    plt.title('Attention Weights Heatmap')
    plt.xlabel('Keys (positions)')
    plt.ylabel('Queries (positions)')
    
    # Plot attention weights as bars for position 0
    plt.subplot(2, 2, 2)
    attention_pos0 = attention_weights[0, 0].detach().numpy()
    plt.bar(range(seq_len), attention_pos0)
    plt.title('Attention from Position 0')
    plt.xlabel('Position')
    plt.ylabel('Attention Weight')
    plt.ylim(0, 1)
    
    # Demonstrate masked self-attention
    plt.subplot(2, 2, 3)
    
    # Create a causal mask (lower triangular)
    mask = torch.tril(torch.ones(seq_len, seq_len))
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    
    # Apply masked self-attention
    output_masked, attention_weights_masked = attention(x, mask)
    
    # Visualize masked attention weights
    sns.heatmap(attention_weights_masked[0].detach().numpy(), 
                annot=True, fmt='.2f', cmap='Reds', 
                xticklabels=range(seq_len), 
                yticklabels=range(seq_len))
    plt.title('Masked Attention Weights (Causal)')
    plt.xlabel('Keys (positions)')
    plt.ylabel('Queries (positions)')
    
    # Show input and output comparison
    plt.subplot(2, 2, 4)
    plt.plot(x[0, :, 0].detach().numpy(), 'b-', label='Input (dim 0)', marker='o')
    plt.plot(output[0, :, 0].detach().numpy(), 'r--', label='Output (dim 0)', marker='s')
    plt.title('Input vs Output (First Dimension)')
    plt.xlabel('Position')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Demonstrate attention with different patterns
    print("\nAttention Pattern Examples:")
    print("-" * 30)
    
    # Create specific input patterns
    # Pattern 1: Increasing values
    pattern1 = torch.arange(seq_len).float().unsqueeze(0).unsqueeze(-1)
    pattern1 = pattern1.expand(1, -1, embed_dim)
    
    output1, weights1 = attention(pattern1)
    print("Pattern 1 - Increasing values:")
    print(f"Input: {pattern1[0, :, 0].tolist()}")
    print(f"Attention to last position: {weights1[0, :, -1].tolist()}")
    

if __name__ == "__main__":
    demonstrate_self_attention()