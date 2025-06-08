"""
Encoder-Decoder Cross-Attention Mechanism

This module demonstrates the cross-attention mechanism that allows the decoder
to attend to the encoder's output. This is crucial for tasks like translation
where the decoder needs to focus on relevant parts of the input sequence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class EncoderDecoderAttention(nn.Module):
    """
    Implements encoder-decoder cross-attention.
    
    The decoder queries attend to encoder keys and values, allowing
    the decoder to focus on relevant parts of the input sequence
    when generating each output token.
    """
    
    def __init__(self, embed_dim, num_heads=1, dropout=0.1):
        """
        Args:
            embed_dim: Dimension of embeddings
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(EncoderDecoderAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Linear projections for Q (from decoder), K, V (from encoder)
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
    def forward(self, decoder_states, encoder_output, encoder_mask=None):
        """
        Forward pass of cross-attention.
        
        Args:
            decoder_states: Decoder hidden states (batch_size, dec_seq_len, embed_dim)
            encoder_output: Encoder output (batch_size, enc_seq_len, embed_dim)
            encoder_mask: Optional mask for encoder padding
            
        Returns:
            output: Attended representation
            attention_weights: Cross-attention weights
        """
        batch_size = decoder_states.size(0)
        dec_seq_len = decoder_states.size(1)
        enc_seq_len = encoder_output.size(1)
        
        # Project to Q, K, V
        Q = self.query_proj(decoder_states)  # From decoder
        K = self.key_proj(encoder_output)    # From encoder
        V = self.value_proj(encoder_output)  # From encoder
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, dec_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, enc_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, enc_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply encoder mask if provided
        if encoder_mask is not None:
            # Expand mask for num_heads and decoder length
            encoder_mask = encoder_mask.unsqueeze(1).unsqueeze(1)
            encoder_mask = encoder_mask.expand(-1, self.num_heads, dec_seq_len, -1)
            scores = scores.masked_fill(encoder_mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(
            batch_size, dec_seq_len, self.embed_dim
        )
        output = self.out_proj(context)
        
        # Average attention weights across heads for visualization
        attention_weights_avg = attention_weights.mean(dim=1)
        
        return output, attention_weights_avg


def visualize_cross_attention_patterns():
    """Visualize different cross-attention patterns in translation-like tasks."""
    
    print("Cross-Attention Patterns Visualization")
    print("=" * 50)
    
    # Parameters
    embed_dim = 64
    num_heads = 4
    
    # Simulate translation scenario
    # Encoder: "The cat sat on mat" (5 tokens)
    # Decoder: "Le chat assis sur tapis" (5 tokens)
    
    enc_seq_len = 5
    dec_seq_len = 5
    batch_size = 1
    
    # Create encoder output (already processed through encoder layers)
    torch.manual_seed(42)
    encoder_output = torch.randn(batch_size, enc_seq_len, embed_dim)
    
    # Create decoder states at different time steps
    decoder_states = torch.randn(batch_size, dec_seq_len, embed_dim)
    
    # Initialize cross-attention
    cross_attention = EncoderDecoderAttention(embed_dim, num_heads)
    
    # Get cross-attention
    output, attention_weights = cross_attention(decoder_states, encoder_output)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Full attention matrix
    ax = axes[0, 0]
    im = ax.imshow(attention_weights[0].detach().numpy(), cmap='Blues', aspect='auto')
    ax.set_xlabel('Encoder Position')
    ax.set_ylabel('Decoder Position')
    ax.set_title('Cross-Attention Weights')
    ax.set_xticks(range(enc_seq_len))
    ax.set_xticklabels(['The', 'cat', 'sat', 'on', 'mat'])
    ax.set_yticks(range(dec_seq_len))
    ax.set_yticklabels(['Le', 'chat', 'assis', 'sur', 'tapis'])
    plt.colorbar(im, ax=ax)
    
    # 2. Attention distribution for each decoder position
    ax = axes[0, 1]
    for i in range(dec_seq_len):
        ax.plot(attention_weights[0, i].detach().numpy(), 
                marker='o', label=f'Dec pos {i}')
    ax.set_xlabel('Encoder Position')
    ax.set_ylabel('Attention Weight')
    ax.set_title('Attention Distribution by Decoder Position')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Simulate different attention patterns
    ax = axes[0, 2]
    
    # Create synthetic attention patterns
    # Pattern 1: Monotonic alignment (diagonal)
    monotonic = torch.zeros(dec_seq_len, enc_seq_len)
    for i in range(min(dec_seq_len, enc_seq_len)):
        monotonic[i, i] = 1.0
    
    im = ax.imshow(monotonic, cmap='Reds', aspect='auto')
    ax.set_xlabel('Encoder Position')
    ax.set_ylabel('Decoder Position')
    ax.set_title('Ideal Monotonic Alignment')
    plt.colorbar(im, ax=ax)
    
    # 4. Attention with encoder masking
    ax = axes[1, 0]
    
    # Create encoder mask (simulate padding on last 2 positions)
    encoder_mask = torch.ones(batch_size, enc_seq_len)
    encoder_mask[0, -2:] = 0
    
    # Apply cross-attention with mask
    _, attention_masked = cross_attention(decoder_states, encoder_output, encoder_mask)
    
    im = ax.imshow(attention_masked[0].detach().numpy(), cmap='Greens', aspect='auto')
    ax.set_xlabel('Encoder Position')
    ax.set_ylabel('Decoder Position')
    ax.set_title('Cross-Attention with Encoder Masking')
    plt.colorbar(im, ax=ax)
    
    # 5. Attention entropy (measure of focus)
    ax = axes[1, 1]
    
    # Calculate entropy for each decoder position
    attention_probs = attention_weights[0].detach()
    entropy = -torch.sum(attention_probs * torch.log(attention_probs + 1e-9), dim=1)
    
    ax.bar(range(dec_seq_len), entropy.numpy())
    ax.set_xlabel('Decoder Position')
    ax.set_ylabel('Attention Entropy')
    ax.set_title('Attention Focus (Lower = More Focused)')
    ax.set_xticks(range(dec_seq_len))
    ax.set_xticklabels(['Le', 'chat', 'assis', 'sur', 'tapis'])
    
    # 6. Attention head diversity
    ax = axes[1, 2]
    
    # Get attention patterns for each head
    cross_attention_multihead = EncoderDecoderAttention(embed_dim, num_heads=4)
    
    # Hook to capture per-head attention
    head_attentions = []
    def hook_fn(module, input, output):
        # Capture attention weights before averaging
        head_attentions.append(output[1])
    
    # Register hook temporarily
    hook = cross_attention_multihead.register_forward_hook(
        lambda m, i, o: head_attentions.append(o[1])
    )
    
    with torch.no_grad():
        # Compute multi-head cross-attention
        Q = cross_attention_multihead.query_proj(decoder_states)
        K = cross_attention_multihead.key_proj(encoder_output)
        V = cross_attention_multihead.value_proj(encoder_output)
        
        Q = Q.view(batch_size, dec_seq_len, num_heads, embed_dim // num_heads).transpose(1, 2)
        K = K.view(batch_size, enc_seq_len, num_heads, embed_dim // num_heads).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / cross_attention_multihead.scale
        head_attention_weights = F.softmax(scores, dim=-1)
    
    # Plot attention similarity between heads
    head_similarity = torch.zeros(num_heads, num_heads)
    for i in range(num_heads):
        for j in range(num_heads):
            # Compute cosine similarity between attention patterns
            attn_i = head_attention_weights[0, i].flatten()
            attn_j = head_attention_weights[0, j].flatten()
            similarity = F.cosine_similarity(attn_i.unsqueeze(0), attn_j.unsqueeze(0))
            head_similarity[i, j] = similarity
    
    im = ax.imshow(head_similarity.numpy(), cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xlabel('Head')
    ax.set_ylabel('Head')
    ax.set_title('Attention Head Similarity')
    ax.set_xticks(range(num_heads))
    ax.set_yticks(range(num_heads))
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Decoder states shape: {decoder_states.shape}")
    print(f"Cross-attention output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")


def demonstrate_attention_in_translation():
    """Demonstrate how cross-attention helps in translation tasks."""
    
    print("\nCross-Attention in Translation Demo")
    print("=" * 50)
    
    # Simulate a simple translation scenario
    embed_dim = 128
    vocab_size = 1000
    
    # Source sentence: "Hello world !" (3 tokens)
    # Target sentence: "Bonjour monde !" (3 tokens)
    
    # Create embeddings
    embedding = nn.Embedding(vocab_size, embed_dim)
    
    # Token IDs (mock)
    source_ids = torch.tensor([[10, 20, 30]])  # "Hello world !"
    target_ids = torch.tensor([[40, 50, 60]])  # "Bonjour monde !"
    
    # Get embeddings
    source_embeddings = embedding(source_ids)
    target_embeddings = embedding(target_ids)
    
    # Simulate encoder output (add some processing)
    encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=4, batch_first=True)
    encoder_output = encoder_layer(source_embeddings)
    
    # Simulate decoder states
    decoder_layer = nn.TransformerDecoderLayer(embed_dim, nhead=4, batch_first=True)
    decoder_states = target_embeddings  # In practice, this would be processed
    
    # Apply cross-attention
    cross_attn = EncoderDecoderAttention(embed_dim, num_heads=4)
    attended_output, attention_weights = cross_attn(decoder_states, encoder_output)
    
    # Visualize the translation attention
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Attention weights
    ax = axes[0]
    im = ax.imshow(attention_weights[0].detach().numpy(), cmap='Blues', aspect='auto')
    ax.set_xlabel('Source Token')
    ax.set_ylabel('Target Token')
    ax.set_title('Translation Attention Weights')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Hello', 'world', '!'])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Bonjour', 'monde', '!'])
    
    # Add annotations
    for i in range(3):
        for j in range(3):
            text = ax.text(j, i, f'{attention_weights[0, i, j]:.2f}',
                          ha='center', va='center', color='black')
    
    plt.colorbar(im, ax=ax)
    
    # Plot 2: Source vs attended representation
    ax = axes[1]
    source_norm = torch.norm(encoder_output[0], dim=-1).detach().numpy()
    attended_norm = torch.norm(attended_output[0], dim=-1).detach().numpy()
    
    x = np.arange(3)
    width = 0.35
    ax.bar(x - width/2, source_norm, width, label='Source Encoding Norm')
    ax.bar(x + width/2, attended_norm, width, label='Attended Output Norm')
    ax.set_xlabel('Position')
    ax.set_ylabel('L2 Norm')
    ax.set_title('Representation Magnitudes')
    ax.set_xticks(x)
    ax.set_xticklabels(['Pos 0', 'Pos 1', 'Pos 2'])
    ax.legend()
    
    # Plot 3: Attention focus per target position
    ax = axes[2]
    max_attention_indices = torch.argmax(attention_weights[0], dim=1).numpy()
    
    ax.bar(range(3), max_attention_indices)
    ax.set_xlabel('Target Position')
    ax.set_ylabel('Most Attended Source Position')
    ax.set_title('Primary Attention Focus')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Bonjour', 'monde', '!'])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Hello', 'world', '!'])
    
    plt.tight_layout()
    plt.show()
    
    # Analyze attention statistics
    print("Attention Statistics:")
    print("-" * 30)
    
    for i, target_word in enumerate(['Bonjour', 'monde', '!']):
        weights = attention_weights[0, i].detach().numpy()
        max_idx = np.argmax(weights)
        source_words = ['Hello', 'world', '!']
        
        print(f"{target_word} -> {source_words[max_idx]} "
              f"(weight: {weights[max_idx]:.3f})")


if __name__ == "__main__":
    visualize_cross_attention_patterns()
    demonstrate_attention_in_translation()