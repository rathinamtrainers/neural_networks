"""
Complete Transformer Architecture (Encoder-Decoder)

This module implements the full transformer architecture combining
encoder and decoder components. This is the complete model as described
in "Attention is All You Need" paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                            -(math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerBlock(nn.Module):
    """Base transformer block with multi-head attention and feed-forward."""
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder blocks."""
    
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerDecoderBlock(nn.Module):
    """Transformer decoder block with masked self-attention and cross-attention."""
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerDecoderBlock, self).__init__()
        
        # Masked self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Masked self-attention
        self_attn_out, _ = self.self_attention(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))
        
        # Cross-attention
        cross_attn_out, _ = self.cross_attention(
            x, enc_output, enc_output, key_padding_mask=src_mask
        )
        x = self.norm2(x + self.dropout(cross_attn_out))
        
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
        
        return x


class TransformerDecoder(nn.Module):
    """Stack of transformer decoder blocks."""
    
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x


class Transformer(nn.Module):
    """
    Complete Transformer model with encoder and decoder.
    
    This implements the full transformer architecture for sequence-to-sequence tasks
    like machine translation, text summarization, etc.
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim=512, 
                 num_heads=8, num_encoder_layers=6, num_decoder_layers=6,
                 ff_dim=2048, max_seq_len=5000, dropout=0.1):
        """
        Args:
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            ff_dim: Feed-forward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
        """
        super(Transformer, self).__init__()
        
        self.embed_dim = embed_dim
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, embed_dim)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len)
        
        # Encoder and Decoder
        self.encoder = TransformerEncoder(
            num_encoder_layers, embed_dim, num_heads, ff_dim, dropout
        )
        self.decoder = TransformerDecoder(
            num_decoder_layers, embed_dim, num_heads, ff_dim, dropout
        )
        
        # Output projection
        self.output_projection = nn.Linear(embed_dim, tgt_vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def create_padding_mask(self, seq, pad_idx=0):
        """Create padding mask for sequences."""
        return (seq == pad_idx)
    
    def create_causal_mask(self, seq_len):
        """Create causal mask for decoder self-attention."""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))
    
    def encode(self, src, src_mask=None):
        """
        Encode source sequence.
        
        Args:
            src: Source token indices (batch_size, src_seq_len)
            src_mask: Source padding mask
            
        Returns:
            Encoder output (batch_size, src_seq_len, embed_dim)
        """
        # Embed and add positional encoding
        src_emb = self.src_embedding(src) * math.sqrt(self.embed_dim)
        src_emb = self.positional_encoding(src_emb)
        src_emb = self.dropout(src_emb)
        
        # Pass through encoder
        enc_output = self.encoder(src_emb, src_mask)
        
        return enc_output
    
    def decode(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        """
        Decode target sequence.
        
        Args:
            tgt: Target token indices (batch_size, tgt_seq_len)
            enc_output: Encoder output
            src_mask: Source padding mask
            tgt_mask: Target causal mask
            
        Returns:
            Decoder output (batch_size, tgt_seq_len, embed_dim)
        """
        # Create causal mask if not provided
        if tgt_mask is None:
            tgt_mask = self.create_causal_mask(tgt.size(1)).to(tgt.device)
        
        # Embed and add positional encoding
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.embed_dim)
        tgt_emb = self.positional_encoding(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)
        
        # Pass through decoder
        dec_output = self.decoder(tgt_emb, enc_output, src_mask, tgt_mask)
        
        return dec_output
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Forward pass through the transformer.
        
        Args:
            src: Source sequences (batch_size, src_seq_len)
            tgt: Target sequences (batch_size, tgt_seq_len)
            src_mask: Source padding mask
            tgt_mask: Target mask
            
        Returns:
            Output logits (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # Encode source
        enc_output = self.encode(src, src_mask)
        
        # Decode target
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        
        # Project to vocabulary
        output = self.output_projection(dec_output)
        
        return output
    
    def generate(self, src, max_length=50, temperature=1.0):
        """
        Generate output sequence using greedy decoding.
        
        Args:
            src: Source sequence (batch_size, src_seq_len)
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            Generated sequence
        """
        self.eval()
        device = src.device
        batch_size = src.size(0)
        
        # Encode source
        src_mask = self.create_padding_mask(src)
        enc_output = self.encode(src, src_mask)
        
        # Start with BOS token (assume it's 1)
        generated = torch.ones(batch_size, 1, dtype=torch.long).to(device)
        
        for _ in range(max_length - 1):
            # Decode
            dec_output = self.decode(generated, enc_output, src_mask)
            
            # Get next token probabilities
            logits = self.output_projection(dec_output[:, -1, :])
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS token (assume it's 2)
            if (next_token == 2).all():
                break
        
        return generated


def visualize_transformer_architecture():
    """Visualize the transformer architecture and data flow."""
    
    print("Transformer Architecture Visualization")
    print("=" * 50)
    
    # Create a small transformer
    transformer = Transformer(
        src_vocab_size=100,
        tgt_vocab_size=100,
        embed_dim=64,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        ff_dim=256
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in transformer.parameters())
    encoder_params = sum(p.numel() for p in transformer.encoder.parameters())
    decoder_params = sum(p.numel() for p in transformer.decoder.parameters())
    
    print(f"Total parameters: {total_params:,}")
    print(f"Encoder parameters: {encoder_params:,}")
    print(f"Decoder parameters: {decoder_params:,}")
    print(f"Embedding parameters: {total_params - encoder_params - decoder_params:,}")
    
    # Create sample data
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8
    
    src = torch.randint(0, 100, (batch_size, src_seq_len))
    tgt = torch.randint(0, 100, (batch_size, tgt_seq_len))
    
    # Forward pass
    output = transformer(src, tgt)
    
    print(f"\nData shapes:")
    print(f"Source input: {src.shape}")
    print(f"Target input: {tgt.shape}")
    print(f"Output: {output.shape}")
    
    # Visualize attention patterns
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Input embeddings
    ax = axes[0, 0]
    src_emb = transformer.src_embedding(src[0])
    im = ax.imshow(src_emb.detach().numpy(), aspect='auto', cmap='viridis')
    ax.set_title('Source Embeddings')
    ax.set_xlabel('Embedding Dimension')
    ax.set_ylabel('Position')
    plt.colorbar(im, ax=ax)
    
    # 2. Positional encoding
    ax = axes[0, 1]
    pe = transformer.positional_encoding.pe[0, :20, :].numpy()
    im = ax.imshow(pe, aspect='auto', cmap='coolwarm')
    ax.set_title('Positional Encoding Pattern')
    ax.set_xlabel('Embedding Dimension')
    ax.set_ylabel('Position')
    plt.colorbar(im, ax=ax)
    
    # 3. Model architecture diagram
    ax = axes[1, 0]
    ax.text(0.5, 0.9, 'TRANSFORMER', ha='center', fontsize=16, weight='bold')
    
    # Encoder
    ax.text(0.25, 0.75, 'ENCODER', ha='center', fontsize=12, weight='bold')
    ax.text(0.25, 0.65, f'{transformer.encoder.layers[0].attention.num_heads} heads', ha='center')
    ax.text(0.25, 0.55, f'{len(transformer.encoder.layers)} layers', ha='center')
    
    # Decoder
    ax.text(0.75, 0.75, 'DECODER', ha='center', fontsize=12, weight='bold')
    ax.text(0.75, 0.65, f'{transformer.decoder.layers[0].self_attention.num_heads} heads', ha='center')
    ax.text(0.75, 0.55, f'{len(transformer.decoder.layers)} layers', ha='center')
    
    # Connections
    ax.arrow(0.25, 0.45, 0.4, 0, head_width=0.02, head_length=0.05, fc='black')
    ax.text(0.5, 0.4, 'Cross-Attention', ha='center', fontsize=10)
    
    # Input/Output
    ax.text(0.25, 0.25, 'Source\nTokens', ha='center', fontsize=10)
    ax.text(0.75, 0.25, 'Target\nTokens', ha='center', fontsize=10)
    ax.text(0.75, 0.1, 'Output\nLogits', ha='center', fontsize=10)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # 4. Output distribution
    ax = axes[1, 1]
    # Get output probabilities for first position
    probs = F.softmax(output[0, 0], dim=-1).detach().numpy()
    top_k = 20
    top_indices = np.argsort(probs)[-top_k:][::-1]
    top_probs = probs[top_indices]
    
    ax.bar(range(top_k), top_probs)
    ax.set_xlabel('Token Index (Top-20)')
    ax.set_ylabel('Probability')
    ax.set_title('Output Distribution (First Position)')
    ax.set_xticks(range(0, top_k, 5))
    ax.set_xticklabels(top_indices[::5])
    
    plt.tight_layout()
    plt.show()


def demonstrate_translation_task():
    """Demonstrate transformer on a mock translation task."""
    
    print("\nTranslation Task Demonstration")
    print("=" * 50)
    
    # Create vocabularies
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    
    # Create transformer
    transformer = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        embed_dim=256,
        num_heads=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        ff_dim=1024
    )
    
    # Mock translation data
    # Source: [BOS, 10, 20, 30, EOS, PAD, PAD]
    # Target: [BOS, 40, 50, 60, EOS, PAD]
    
    src = torch.tensor([[1, 10, 20, 30, 2, 0, 0]])
    tgt_input = torch.tensor([[1, 40, 50, 60, 2, 0]])
    tgt_output = torch.tensor([[40, 50, 60, 2, 0, 0]])  # Shifted for training
    
    # Create masks
    src_mask = transformer.create_padding_mask(src)
    tgt_mask = transformer.create_causal_mask(tgt_input.size(1))
    
    # Forward pass
    output = transformer(src, tgt_input, src_mask, tgt_mask)
    
    # Calculate loss (mock)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    loss = criterion(output.view(-1, tgt_vocab_size), tgt_output.view(-1))
    
    print(f"Source shape: {src.shape}")
    print(f"Target input shape: {tgt_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Mock loss: {loss.item():.4f}")
    
    # Test generation
    print("\nTesting generation:")
    generated = transformer.generate(src, max_length=10)
    print(f"Generated sequence: {generated[0].tolist()}")
    
    # Visualize attention flow
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Visualize source mask
    ax1.imshow(src_mask[0].unsqueeze(0).numpy(), cmap='binary', aspect='auto')
    ax1.set_title('Source Padding Mask')
    ax1.set_xlabel('Position')
    ax1.set_yticks([])
    
    # Visualize target causal mask
    tgt_mask_vis = tgt_mask.numpy()
    tgt_mask_vis = np.where(tgt_mask_vis == float('-inf'), -1, tgt_mask_vis)
    ax2.imshow(tgt_mask_vis, cmap='RdBu', vmin=-1, vmax=0)
    ax2.set_title('Target Causal Mask')
    ax2.set_xlabel('Key Position')
    ax2.set_ylabel('Query Position')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_transformer_architecture()
    demonstrate_translation_task()