"""
Lesson 3: Positional Encoding
=============================

Transformers process all positions in parallel, unlike RNNs which process sequentially.
This means they need explicit positional information to understand word order.

Key concepts:
- Why positional encoding is needed
- Sinusoidal positional encoding (original Transformer)
- Learned positional embeddings
- Relative positional encoding
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from 'Attention Is All You Need'
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model, max_seq_len=5000):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        
        # Create the division term for the encoding
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(np.log(10000.0) / d_model))
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class LearnedPositionalEncoding(nn.Module):
    """Learned positional embeddings"""
    
    def __init__(self, d_model, max_seq_len=5000):
        super().__init__()
        self.positional_embeddings = nn.Embedding(max_seq_len, d_model)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        pos_embeddings = self.positional_embeddings(positions)
        return x + pos_embeddings

class RelativePositionalEncoding(nn.Module):
    """
    Simplified relative positional encoding
    Encodes relative distances between positions
    """
    
    def __init__(self, d_model, max_relative_distance=32):
        super().__init__()
        self.d_model = d_model
        self.max_relative_distance = max_relative_distance
        
        # Create relative position embeddings
        # Vocabulary size: 2 * max_relative_distance + 1 (for distances from -max to +max)
        vocab_size = 2 * max_relative_distance + 1
        self.relative_embeddings = nn.Embedding(vocab_size, d_model)
        
    def forward(self, seq_len):
        """
        Create relative positional encoding matrix
        
        Returns:
            relative_pos: (seq_len, seq_len, d_model)
        """
        # Create matrix of relative positions
        positions = torch.arange(seq_len)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        
        # Clip to max distance and shift to positive indices
        relative_positions = torch.clamp(
            relative_positions, 
            -self.max_relative_distance, 
            self.max_relative_distance
        ) + self.max_relative_distance
        
        # Get embeddings
        relative_embeddings = self.relative_embeddings(relative_positions)
        
        return relative_embeddings

def visualize_positional_encoding(pe_matrix, title="Positional Encoding"):
    """Visualize positional encoding as a heatmap"""
    plt.figure(figsize=(12, 8))
    
    # Take first batch if it exists
    if len(pe_matrix.shape) == 3:
        pe_matrix = pe_matrix[0]
    
    pe_np = pe_matrix.detach().numpy()
    
    plt.imshow(pe_np.T, cmap='RdYlBu', aspect='auto')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Position')
    plt.ylabel('Dimension')
    plt.tight_layout()
    plt.show()

def compare_positional_encodings():
    """Compare different positional encoding methods"""
    d_model = 64
    seq_len = 50
    
    # Create dummy input
    x = torch.zeros(1, seq_len, d_model)
    
    # 1. Sinusoidal encoding
    sin_pe = SinusoidalPositionalEncoding(d_model)
    x_sin = sin_pe(x)
    
    # 2. Learned encoding
    learned_pe = LearnedPositionalEncoding(d_model)
    x_learned = learned_pe(x)
    
    # 3. Relative encoding (just the encoding matrix)
    rel_pe = RelativePositionalEncoding(d_model)
    rel_encoding = rel_pe(seq_len)
    
    return x_sin, x_learned, rel_encoding

def demonstrate_positional_similarity():
    """Show how positional encodings maintain similarity patterns"""
    d_model = 128
    pe = SinusoidalPositionalEncoding(d_model)
    
    # Get positional encodings for different positions
    dummy_input = torch.zeros(1, 100, d_model)
    encoded = pe(dummy_input)
    
    # Extract just the positional encodings (subtract the zero input)
    pos_encodings = encoded[0]  # Shape: (100, 128)
    
    # Calculate cosine similarity between different positions
    similarities = torch.cosine_similarity(
        pos_encodings.unsqueeze(1), 
        pos_encodings.unsqueeze(0), 
        dim=-1
    )
    
    return similarities

if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Example 1: Basic sinusoidal positional encoding
    print("=== Example 1: Sinusoidal Positional Encoding ===")
    
    d_model = 128
    seq_len = 20
    
    pe = SinusoidalPositionalEncoding(d_model)
    
    # Create dummy input (all zeros to see pure positional encoding)
    x = torch.zeros(1, seq_len, d_model)
    x_with_pe = pe(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_with_pe.shape}")
    print(f"First position encoding (first 10 dims): {x_with_pe[0, 0, :10]}")
    print(f"Second position encoding (first 10 dims): {x_with_pe[0, 1, :10]}")
    
    # Visualize the positional encoding
    visualize_positional_encoding(x_with_pe, "Sinusoidal Positional Encoding")
    
    # Example 2: Compare different encoding types
    print("\n=== Example 2: Comparing Encoding Types ===")
    
    x_sin, x_learned, rel_encoding = compare_positional_encodings()
    
    print(f"Sinusoidal encoding range: [{x_sin.min():.3f}, {x_sin.max():.3f}]")
    print(f"Learned encoding range: [{x_learned.min():.3f}, {x_learned.max():.3f}]")
    print(f"Relative encoding shape: {rel_encoding.shape}")
    
    # Example 3: Positional similarity patterns
    print("\n=== Example 3: Positional Similarity Patterns ===")
    
    similarities = demonstrate_positional_similarity()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(similarities.numpy(), cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Cosine Similarity Between Positional Encodings')
    plt.xlabel('Position')
    plt.ylabel('Position')
    plt.show()
    
    # Show similarity for specific positions
    pos_10 = similarities[10, :]
    print(f"Similarity of position 10 with positions 8-12: {pos_10[8:13]}")
    
    # Example 4: Effect on attention
    print("\n=== Example 4: Impact on Attention ===")
    
    # Simple example showing how positional encoding affects attention
    from lessons.multi_head_attention import MultiHeadAttention
    
    # Create input with and without positional encoding
    seq_len = 6
    d_model = 64
    
    # Input without positional information (identical vectors)
    x_no_pos = torch.ones(1, seq_len, d_model)
    
    # Input with positional encoding
    pe = SinusoidalPositionalEncoding(d_model)
    x_with_pos = pe(x_no_pos)
    
    # Apply attention
    mha = MultiHeadAttention(d_model, num_heads=1)
    
    _, attn_no_pos = mha(x_no_pos, x_no_pos, x_no_pos)
    _, attn_with_pos = mha(x_with_pos, x_with_pos, x_with_pos)
    
    print("Attention without positional encoding (should be uniform):")
    print(attn_no_pos[0, 0, 0, :])  # First query position
    
    print("\nAttention with positional encoding (should vary by position):")
    print(attn_with_pos[0, 0, 0, :])  # First query position
    
    # Example 5: Relative positional encoding
    print("\n=== Example 5: Relative Positional Encoding ===")
    
    rel_pe = RelativePositionalEncoding(d_model=32, max_relative_distance=5)
    rel_matrix = rel_pe(seq_len=8)
    
    print(f"Relative encoding matrix shape: {rel_matrix.shape}")
    
    # Visualize relative positional encoding
    # Show encoding for relative distance of +2 (2 positions to the right)
    rel_dist_2 = rel_matrix[:, :, 0]  # Just first dimension for visualization
    
    plt.figure(figsize=(8, 6))
    plt.imshow(rel_dist_2.detach().numpy(), cmap='RdYlBu')
    plt.colorbar()
    plt.title('Relative Positional Encoding Matrix (First Dimension)')
    plt.xlabel('Position j')
    plt.ylabel('Position i')
    plt.show()
    
    print("Relative encoding captures relationships between positions rather than absolute positions.")