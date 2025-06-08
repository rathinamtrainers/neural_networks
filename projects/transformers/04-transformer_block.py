"""
Complete Transformer Block Implementation

This module implements a complete transformer encoder block, combining multi-head
attention with feed-forward networks, layer normalization, and residual connections.
This is the fundamental building block of transformer architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
# Using PyTorch's built-in MultiheadAttention instead of custom implementation

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, embed_dim, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                            -(np.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if embed_dim % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class FeedForward(nn.Module):
    """
    Implements the feed-forward network used in transformer blocks.
    
    This consists of two linear transformations with a ReLU activation
    in between, following the paper's specification.
    """
    
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        """
        Args:
            embed_dim: Dimension of input/output
            ff_dim: Dimension of the hidden layer (typically 4x embed_dim)
            dropout: Dropout probability
        """
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass: Linear -> ReLU -> Dropout -> Linear -> Dropout
        """
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Implements a single transformer encoder block.
    
    The block consists of:
    1. Multi-head self-attention
    2. Add & Norm (residual connection + layer normalization)
    3. Feed-forward network
    4. Add & Norm
    """
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        """
        Args:
            embed_dim: Dimension of embeddings
            num_heads: Number of attention heads
            ff_dim: Dimension of feed-forward hidden layer
            dropout: Dropout probability
        """
        super(TransformerEncoderBlock, self).__init__()
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Forward pass of transformer encoder block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerEncoder(nn.Module):
    """
    Stack of transformer encoder blocks with positional encoding.
    """
    
    def __init__(self, num_blocks, embed_dim, num_heads, ff_dim, 
                 max_seq_len=5000, dropout=0.1):
        """
        Args:
            num_blocks: Number of transformer blocks to stack
            embed_dim: Dimension of embeddings
            num_heads: Number of attention heads
            ff_dim: Dimension of feed-forward hidden layer
            max_seq_len: Maximum sequence length for positional encoding
            dropout: Dropout probability
        """
        super(TransformerEncoder, self).__init__()
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len, dropout)
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_blocks)
        ])
        
    def forward(self, x, mask=None):
        """
        Forward pass through the transformer encoder.
        
        Args:
            x: Input embeddings of shape (batch_size, seq_len, embed_dim)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask)
            
        return x


def visualize_transformer_block():
    """Visualize the information flow through a transformer block."""
    
    print("Transformer Block Visualization")
    print("=" * 50)
    
    # Parameters
    batch_size = 2
    seq_len = 10
    embed_dim = 64
    num_heads = 4
    ff_dim = 256
    
    # Create sample input
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Initialize transformer block
    transformer_block = TransformerEncoderBlock(embed_dim, num_heads, ff_dim)
    
    # Track intermediate outputs
    intermediate_outputs = {}
    
    # Hook to capture intermediate values
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                intermediate_outputs[name] = output[0].detach()
            else:
                intermediate_outputs[name] = output.detach()
        return hook
    
    # Register hooks
    transformer_block.attention.register_forward_hook(hook_fn('attention'))
    transformer_block.norm1.register_forward_hook(hook_fn('norm1'))
    transformer_block.feed_forward.register_forward_hook(hook_fn('feed_forward'))
    transformer_block.norm2.register_forward_hook(hook_fn('norm2'))
    
    # Forward pass
    output = transformer_block(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in transformer_block.parameters())}")
    
    # Visualize intermediate outputs
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot input
    ax = axes[0]
    im = ax.imshow(x[0].numpy(), aspect='auto', cmap='viridis')
    ax.set_title('Input')
    ax.set_xlabel('Embedding Dimension')
    ax.set_ylabel('Position')
    plt.colorbar(im, ax=ax)
    
    # Plot attention output
    ax = axes[1]
    attn_out = intermediate_outputs['attention']
    im = ax.imshow(attn_out[0].numpy(), aspect='auto', cmap='viridis')
    ax.set_title('After Multi-Head Attention')
    ax.set_xlabel('Embedding Dimension')
    ax.set_ylabel('Position')
    plt.colorbar(im, ax=ax)
    
    # Plot after first norm
    ax = axes[2]
    norm1_out = intermediate_outputs['norm1']
    im = ax.imshow(norm1_out[0].numpy(), aspect='auto', cmap='viridis')
    ax.set_title('After Add & Norm 1')
    ax.set_xlabel('Embedding Dimension')
    ax.set_ylabel('Position')
    plt.colorbar(im, ax=ax)
    
    # Plot feed-forward output
    ax = axes[3]
    ff_out = intermediate_outputs['feed_forward']
    im = ax.imshow(ff_out[0].numpy(), aspect='auto', cmap='viridis')
    ax.set_title('After Feed-Forward')
    ax.set_xlabel('Embedding Dimension')
    ax.set_ylabel('Position')
    plt.colorbar(im, ax=ax)
    
    # Plot final output
    ax = axes[4]
    im = ax.imshow(output[0].detach().numpy(), aspect='auto', cmap='viridis')
    ax.set_title('Final Output (After Add & Norm 2)')
    ax.set_xlabel('Embedding Dimension')
    ax.set_ylabel('Position')
    plt.colorbar(im, ax=ax)
    
    # Hide unused subplot
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Analyze residual connections
    print("\nResidual Connection Analysis:")
    print("-" * 30)
    
    # Compute norms
    input_norm = torch.norm(x[0], dim=-1).mean().item()
    output_norm = torch.norm(output[0], dim=-1).mean().item()
    
    print(f"Average input norm: {input_norm:.4f}")
    print(f"Average output norm: {output_norm:.4f}")
    print(f"Norm ratio (output/input): {output_norm/input_norm:.4f}")


def test_transformer_encoder():
    """Test the complete transformer encoder."""
    
    print("\nTesting Transformer Encoder")
    print("=" * 50)
    
    # Parameters
    batch_size = 2
    seq_len = 20
    embed_dim = 128
    num_heads = 8
    ff_dim = 512
    num_blocks = 3
    
    # Create input embeddings
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Initialize transformer encoder
    encoder = TransformerEncoder(num_blocks, embed_dim, num_heads, ff_dim)
    
    # Forward pass
    output = encoder(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of transformer blocks: {num_blocks}")
    print(f"Total parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    # Test with padding mask
    print("\nTesting with Padding Mask:")
    
    # Create a padding mask where last 5 positions are padded
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    mask[:, -5:] = True  # Mask out last 5 positions
    
    output_masked = encoder(x, mask)
    print(f"Masked output shape: {output_masked.shape}")
    
    # Visualize effect of stacking blocks
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot norm through layers
    ax = axes[0]
    norms_per_layer = []
    
    # Track norm after each block
    x_temp = encoder.pos_encoding(x)
    norms_per_layer.append(torch.norm(x_temp, dim=-1).mean().item())
    
    for i, block in enumerate(encoder.blocks):
        x_temp = block(x_temp)
        norms_per_layer.append(torch.norm(x_temp, dim=-1).mean().item())
    
    ax.plot(range(len(norms_per_layer)), norms_per_layer, 'o-')
    ax.set_xlabel('Layer (0 = after pos encoding)')
    ax.set_ylabel('Average Norm')
    ax.set_title('Norm Propagation Through Layers')
    ax.grid(True, alpha=0.3)
    
    # Compare input and output distributions
    ax = axes[1]
    ax.hist(x.flatten().numpy(), bins=50, alpha=0.5, label='Input', density=True)
    ax.hist(output.flatten().detach().numpy(), bins=50, alpha=0.5, 
            label='Output', density=True)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title('Input vs Output Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    # Import from local files
    import sys
    sys.path.append('.')
    
    visualize_transformer_block()
    test_transformer_encoder()