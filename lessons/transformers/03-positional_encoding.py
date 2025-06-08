"""
Positional Encoding Implementation and Visualization

This module implements positional encoding for transformers. Since transformers
have no inherent notion of position or order, positional encodings are added
to give the model information about the position of tokens in a sequence.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns


class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding described in
    "Attention is All You Need" paper.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    where pos is the position and i is the dimension.
    """
    
    def __init__(self, embed_dim, max_len=5000, dropout=0.1):
        """
        Args:
            embed_dim: Dimension of embeddings
            max_len: Maximum length of sequences
            dropout: Dropout probability
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create a matrix to hold positional encodings
        pe = torch.zeros(max_len, embed_dim)
        
        # Create position indices
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # Create dimension indices
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                            -(np.log(10000.0) / embed_dim))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices
        if embed_dim % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            Output tensor with positional encoding added
        """
        # Add positional encoding
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def visualize_positional_encoding():
    """Visualize positional encoding patterns."""
    
    print("Positional Encoding Visualization")
    print("=" * 50)
    
    # Create positional encoding
    embed_dim = 128
    max_len = 100
    pos_encoding = PositionalEncoding(embed_dim, max_len, dropout=0.0)
    
    # Get the positional encoding matrix
    pe_matrix = pos_encoding.pe[0].numpy()
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Heatmap of positional encodings
    ax = axes[0, 0]
    im = ax.imshow(pe_matrix[:50, :].T, aspect='auto', cmap='RdBu_r')
    ax.set_xlabel('Position')
    ax.set_ylabel('Dimension')
    ax.set_title('Positional Encoding Heatmap (First 50 Positions)')
    plt.colorbar(im, ax=ax)
    
    # 2. Individual sine waves for different dimensions
    ax = axes[0, 1]
    positions = np.arange(100)
    for i in range(0, 8, 2):
        ax.plot(positions, pe_matrix[:, i], label=f'dim {i} (sin)')
        ax.plot(positions, pe_matrix[:, i+1], '--', label=f'dim {i+1} (cos)')
    ax.set_xlabel('Position')
    ax.set_ylabel('Encoding Value')
    ax.set_title('Sinusoidal Patterns for Different Dimensions')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 3. Encoding values for specific positions
    ax = axes[1, 0]
    positions_to_show = [0, 10, 20, 30, 40]
    x_coords = np.arange(embed_dim)
    for pos in positions_to_show:
        ax.plot(x_coords, pe_matrix[pos, :], label=f'Position {pos}', alpha=0.7)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Encoding Value')
    ax.set_title('Encoding Patterns for Different Positions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Dot product similarity between positions
    ax = axes[1, 1]
    # Compute dot product similarity between positions
    similarity = np.dot(pe_matrix[:20, :], pe_matrix[:20, :].T)
    im = ax.imshow(similarity, cmap='coolwarm', aspect='auto')
    ax.set_xlabel('Position')
    ax.set_ylabel('Position')
    ax.set_title('Positional Encoding Similarity Matrix')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print(f"Positional encoding shape: {pe_matrix.shape}")
    print(f"Encoding range: [{pe_matrix.min():.3f}, {pe_matrix.max():.3f}]")
    print(f"Mean: {pe_matrix.mean():.3f}, Std: {pe_matrix.std():.3f}")


def demonstrate_positional_encoding_effect():
    """Demonstrate how positional encoding affects embeddings."""
    
    print("\nDemonstrating Positional Encoding Effect")
    print("=" * 50)
    
    # Parameters
    batch_size = 2
    seq_len = 8
    embed_dim = 64
    
    # Create identical embeddings at each position
    # This simulates having the same word at different positions
    embedding = torch.randn(1, embed_dim)
    x = embedding.repeat(batch_size, seq_len, 1)
    
    # Initialize positional encoding
    pos_encoding = PositionalEncoding(embed_dim, dropout=0.0)
    
    # Apply positional encoding
    x_with_pos = pos_encoding(x)
    
    print(f"Original embedding shape: {x.shape}")
    print(f"All positions have same embedding: {torch.allclose(x[0, 0], x[0, 1])}")
    print(f"After positional encoding: {torch.allclose(x_with_pos[0, 0], x_with_pos[0, 1])}")
    
    # Visualize the difference
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Original embeddings
    ax = axes[0, 0]
    im = ax.imshow(x[0].numpy(), aspect='auto', cmap='viridis')
    ax.set_xlabel('Embedding Dimension')
    ax.set_ylabel('Position')
    ax.set_title('Original Embeddings (All Same)')
    plt.colorbar(im, ax=ax)
    
    # Embeddings with positional encoding
    ax = axes[0, 1]
    im = ax.imshow(x_with_pos[0].numpy(), aspect='auto', cmap='viridis')
    ax.set_xlabel('Embedding Dimension')
    ax.set_ylabel('Position')
    ax.set_title('Embeddings + Positional Encoding')
    plt.colorbar(im, ax=ax)
    
    # Cosine similarity between positions (original)
    ax = axes[1, 0]
    cos_sim_orig = torch.nn.functional.cosine_similarity(
        x[0].unsqueeze(0), x[0].unsqueeze(1), dim=2
    )
    im = ax.imshow(cos_sim_orig.numpy(), cmap='Blues', vmin=0, vmax=1)
    ax.set_xlabel('Position')
    ax.set_ylabel('Position')
    ax.set_title('Cosine Similarity (Original)')
    plt.colorbar(im, ax=ax)
    
    # Cosine similarity between positions (with PE)
    ax = axes[1, 1]
    cos_sim_pe = torch.nn.functional.cosine_similarity(
        x_with_pos[0].unsqueeze(0), x_with_pos[0].unsqueeze(1), dim=2
    )
    im = ax.imshow(cos_sim_pe.numpy(), cmap='Blues', vmin=0, vmax=1)
    ax.set_xlabel('Position')
    ax.set_ylabel('Position')
    ax.set_title('Cosine Similarity (With Positional Encoding)')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.show()
    
    # Show how distance affects similarity
    print("\nPosition Distance vs Similarity:")
    print("-" * 30)
    pos_0 = x_with_pos[0, 0]
    for i in range(1, min(8, seq_len)):
        similarity = torch.nn.functional.cosine_similarity(
            pos_0.unsqueeze(0), x_with_pos[0, i].unsqueeze(0)
        ).item()
        print(f"Distance {i}: Similarity = {similarity:.4f}")


def analyze_frequency_components():
    """Analyze the frequency components of positional encoding."""
    
    print("\nAnalyzing Frequency Components")
    print("=" * 50)
    
    embed_dim = 128
    max_len = 1000
    
    # Calculate wavelengths for each dimension
    wavelengths = []
    for i in range(0, embed_dim, 2):
        wavelength = 2 * np.pi / (1 / (10000 ** (i / embed_dim)))
        wavelengths.append(wavelength)
    
    plt.figure(figsize=(12, 6))
    
    # Plot wavelengths
    plt.subplot(1, 2, 1)
    plt.plot(range(0, embed_dim, 2), wavelengths)
    plt.xlabel('Dimension Index (even)')
    plt.ylabel('Wavelength')
    plt.title('Wavelength by Dimension')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Plot frequencies
    plt.subplot(1, 2, 2)
    frequencies = 1 / np.array(wavelengths)
    plt.plot(range(0, embed_dim, 2), frequencies)
    plt.xlabel('Dimension Index (even)')
    plt.ylabel('Frequency')
    plt.title('Frequency by Dimension')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Lowest frequency (dim 0): {frequencies[0]:.6f}")
    print(f"Highest frequency (dim {embed_dim-2}): {frequencies[-1]:.6f}")
    print(f"Frequency ratio: {frequencies[-1] / frequencies[0]:.2f}")


if __name__ == "__main__":
    visualize_positional_encoding()
    demonstrate_positional_encoding_effect()
    analyze_frequency_components()