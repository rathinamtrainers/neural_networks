"""
Transformer Decoder Implementation

This module implements the decoder component of the transformer architecture,
including masked self-attention, encoder-decoder attention, and feed-forward layers.
The decoder generates output sequences autoregressively.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class TransformerDecoderBlock(nn.Module):
    """
    A single transformer decoder block.
    
    Components:
    1. Masked multi-head self-attention (prevents looking at future tokens)
    2. Multi-head encoder-decoder attention (attends to encoder output)
    3. Feed-forward network
    4. Layer normalization and residual connections
    """
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward hidden dimension
            dropout: Dropout probability
        """
        super(TransformerDecoderBlock, self).__init__()
        
        # Masked self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Encoder-decoder attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Forward pass through decoder block.
        
        Args:
            x: Decoder input of shape (batch_size, tgt_seq_len, embed_dim)
            encoder_output: Encoder output of shape (batch_size, src_seq_len, embed_dim)
            src_mask: Source padding mask
            tgt_mask: Target attention mask (causal mask)
            
        Returns:
            output: Decoder block output
            self_attn_weights: Self-attention weights
            cross_attn_weights: Cross-attention weights
        """
        # Masked self-attention
        self_attn_output, self_attn_weights = self.self_attention(
            x, x, x, attn_mask=tgt_mask
        )
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # Encoder-decoder attention
        cross_attn_output, cross_attn_weights = self.cross_attention(
            x, encoder_output, encoder_output, key_padding_mask=src_mask
        )
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x, self_attn_weights, cross_attn_weights


class TransformerDecoder(nn.Module):
    """
    Stack of transformer decoder blocks.
    """
    
    def __init__(self, num_blocks, embed_dim, num_heads, ff_dim, 
                 vocab_size, max_seq_len=100, dropout=0.1):
        """
        Args:
            num_blocks: Number of decoder blocks
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward hidden dimension
            vocab_size: Output vocabulary size
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super(TransformerDecoder, self).__init__()
        
        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(max_seq_len, embed_dim)
        
        # Decoder blocks
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_blocks)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def _create_positional_encoding(self, max_len, embed_dim):
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                            -(np.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def create_causal_mask(self, seq_len):
        """
        Create causal mask to prevent attention to future positions.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Causal mask of shape (seq_len, seq_len)
        """
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))
    
    def forward(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """
        Forward pass through decoder.
        
        Args:
            tgt: Target sequence indices of shape (batch_size, tgt_seq_len)
            encoder_output: Encoder output
            src_mask: Source padding mask
            tgt_mask: Target mask (if None, causal mask is created)
            
        Returns:
            output: Decoder output logits
            attention_weights: Dict containing attention weights from all layers
        """
        seq_len = tgt.size(1)
        
        # Create causal mask if not provided
        if tgt_mask is None:
            tgt_mask = self.create_causal_mask(seq_len).to(tgt.device)
        
        # Embed tokens and add positional encoding
        x = self.embedding(tgt)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        
        # Store attention weights
        self_attn_weights = []
        cross_attn_weights = []
        
        # Pass through decoder blocks
        for block in self.blocks:
            x, self_attn, cross_attn = block(x, encoder_output, src_mask, tgt_mask)
            self_attn_weights.append(self_attn)
            cross_attn_weights.append(cross_attn)
        
        # Output projection
        output = self.output_projection(x)
        
        attention_weights = {
            'self_attention': self_attn_weights,
            'cross_attention': cross_attn_weights
        }
        
        return output, attention_weights


def visualize_masked_attention():
    """Visualize how masked self-attention works in the decoder."""
    
    print("Masked Self-Attention Visualization")
    print("=" * 50)
    
    # Create a simple decoder
    embed_dim = 64
    num_heads = 4
    vocab_size = 100
    
    decoder = TransformerDecoder(
        num_blocks=1,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=256,
        vocab_size=vocab_size
    )
    
    # Create sample data
    batch_size = 1
    seq_len = 8
    tgt = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Mock encoder output
    encoder_seq_len = 10
    encoder_output = torch.randn(batch_size, encoder_seq_len, embed_dim)
    
    # Forward pass
    output, attn_weights = decoder(tgt, encoder_output)
    
    # Visualize causal mask
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot causal mask
    ax = axes[0, 0]
    causal_mask = decoder.create_causal_mask(seq_len).numpy()
    # Replace -inf with -1 for visualization
    causal_mask_vis = np.where(causal_mask == float('-inf'), -1, causal_mask)
    im = ax.imshow(causal_mask_vis, cmap='RdBu', vmin=-1, vmax=0)
    ax.set_title('Causal Mask (prevents looking ahead)')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    plt.colorbar(im, ax=ax)
    
    # Plot self-attention weights
    ax = axes[0, 1]
    self_attn = attn_weights['self_attention'][0].detach().mean(dim=1).squeeze().numpy()
    im = ax.imshow(self_attn, cmap='Blues', vmin=0, vmax=1)
    ax.set_title('Masked Self-Attention Weights')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    plt.colorbar(im, ax=ax)
    
    # Plot cross-attention weights
    ax = axes[1, 0]
    cross_attn = attn_weights['cross_attention'][0].detach().mean(dim=1).squeeze().numpy()
    im = ax.imshow(cross_attn, cmap='Greens', aspect='auto')
    ax.set_title('Cross-Attention Weights\n(Decoder attending to Encoder)')
    ax.set_xlabel('Encoder Position')
    ax.set_ylabel('Decoder Position')
    plt.colorbar(im, ax=ax)
    
    # Plot attention pattern for specific decoder position
    ax = axes[1, 1]
    decoder_pos = 4
    ax.bar(range(encoder_seq_len), cross_attn[decoder_pos])
    ax.set_title(f'Encoder Attention from Decoder Position {decoder_pos}')
    ax.set_xlabel('Encoder Position')
    ax.set_ylabel('Attention Weight')
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Target sequence shape: {tgt.shape}")
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Decoder output shape: {output.shape}")


def demonstrate_autoregressive_generation():
    """Demonstrate autoregressive generation with the decoder."""
    
    print("\nAutoregressive Generation Demo")
    print("=" * 50)
    
    # Initialize decoder
    vocab_size = 50
    embed_dim = 64
    decoder = TransformerDecoder(
        num_blocks=2,
        embed_dim=embed_dim,
        num_heads=4,
        ff_dim=256,
        vocab_size=vocab_size
    )
    
    # Mock encoder output (e.g., from encoding "Hello")
    encoder_output = torch.randn(1, 5, embed_dim)
    
    # Start token
    start_token = 1
    max_length = 10
    
    # Generate sequence autoregressively
    generated_sequence = [start_token]
    
    decoder.eval()
    with torch.no_grad():
        for i in range(max_length - 1):
            # Create input tensor from generated sequence
            tgt = torch.tensor([generated_sequence])
            
            # Get decoder output
            output, _ = decoder(tgt, encoder_output)
            
            # Get probabilities for next token
            next_token_logits = output[0, -1, :]
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample next token (you could also use argmax for greedy decoding)
            next_token = torch.multinomial(next_token_probs, 1).item()
            
            generated_sequence.append(next_token)
            
            # Stop if end token (let's say it's 2)
            if next_token == 2:
                break
    
    print(f"Generated sequence: {generated_sequence}")
    print(f"Length: {len(generated_sequence)}")
    
    # Visualize generation probabilities
    plt.figure(figsize=(10, 6))
    
    # Re-run to capture probabilities
    all_probs = []
    tgt_seq = [start_token]
    
    with torch.no_grad():
        for i in range(len(generated_sequence) - 1):
            tgt = torch.tensor([tgt_seq])
            output, _ = decoder(tgt, encoder_output)
            probs = F.softmax(output[0, -1, :], dim=-1).numpy()
            all_probs.append(probs)
            tgt_seq.append(generated_sequence[i + 1])
    
    # Plot top-k probabilities at each step
    top_k = 10
    positions = range(len(all_probs))
    
    for pos, probs in enumerate(all_probs):
        top_indices = np.argsort(probs)[-top_k:][::-1]
        top_probs = probs[top_indices]
        
        plt.bar(np.arange(top_k) + pos * (top_k + 1), top_probs, 
                label=f'Step {pos+1}' if pos < 3 else '')
        
        # Highlight the chosen token
        chosen_idx = generated_sequence[pos + 1]
        if chosen_idx in top_indices:
            chosen_pos = np.where(top_indices == chosen_idx)[0][0]
            plt.bar(chosen_pos + pos * (top_k + 1), top_probs[chosen_pos], 
                   color='red', alpha=0.7)
    
    plt.xlabel('Token Rank (per step)')
    plt.ylabel('Probability')
    plt.title('Top-10 Token Probabilities During Generation')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_masked_attention()
    demonstrate_autoregressive_generation()