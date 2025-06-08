"""
Lesson 6: Transformer Decoder
=============================

The transformer decoder is used for autoregressive generation tasks like:
- Language modeling (GPT-style models)
- Machine translation (decoder part)
- Text generation and completion

Key differences from encoder:
- Causal (masked) self-attention to prevent looking at future tokens
- Cross-attention (when used with encoder) to attend to encoder outputs
- Autoregressive generation during inference

Key concepts:
- Causal masking for autoregressive modeling
- Cross-attention mechanism  
- Decoder-only vs encoder-decoder architectures
- Generation strategies (greedy, beam search, sampling)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from lessons.multi_head_attention import MultiHeadAttention
from lessons.transformer_encoder_block import FeedForward
from lessons.positional_encoding import SinusoidalPositionalEncoding

class CausalMultiHeadAttention(MultiHeadAttention):
    """Multi-head attention with causal masking"""
    
    def __init__(self, d_model, num_heads):
        super().__init__(d_model, num_heads)
        
    def create_causal_mask(self, seq_len, device):
        """Create causal mask to prevent attention to future positions"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
    
    def forward(self, query, key, value, mask=None):
        """
        Apply causal masking in addition to any provided mask
        """
        seq_len = query.size(1)
        device = query.device
        
        # Create causal mask
        causal_mask = self.create_causal_mask(seq_len, device)
        
        # Combine with provided mask if exists
        if mask is not None:
            mask = mask * causal_mask
        else:
            mask = causal_mask
            
        return super().forward(query, key, value, mask)

class TransformerDecoderBlock(nn.Module):
    """
    Single transformer decoder block
    Can be used in decoder-only (GPT-style) or encoder-decoder architectures
    """
    
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1, 
                 has_cross_attention=False):
        super().__init__()
        
        self.has_cross_attention = has_cross_attention
        
        # Masked self-attention (causal)
        self.self_attention = CausalMultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-attention (for encoder-decoder architectures)
        if has_cross_attention:
            self.cross_attention = MultiHeadAttention(d_model, num_heads)
            self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output=None, self_attn_mask=None, cross_attn_mask=None):
        """
        Args:
            x: decoder input (batch_size, tgt_seq_len, d_model)
            encoder_output: encoder output for cross-attention (batch_size, src_seq_len, d_model)
            self_attn_mask: mask for self-attention
            cross_attn_mask: mask for cross-attention
        """
        # 1. Masked self-attention
        attn_output, self_attn_weights = self.self_attention(x, x, x, self_attn_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        cross_attn_weights = None
        
        # 2. Cross-attention (if encoder-decoder architecture)
        if self.has_cross_attention and encoder_output is not None:
            cross_attn_output, cross_attn_weights = self.cross_attention(
                x, encoder_output, encoder_output, cross_attn_mask
            )
            x = self.norm2(x + self.dropout(cross_attn_output))
        
        # 3. Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x, self_attn_weights, cross_attn_weights

class TransformerDecoder(nn.Module):
    """
    Complete transformer decoder (like GPT)
    """
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff=None,
                 max_seq_len=1024, dropout=0.1, has_cross_attention=False, 
                 pad_token_id=0):
        super().__init__()
        
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.has_cross_attention = has_cross_attention
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len)
        
        # Decoder blocks
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(d_model, num_heads, d_ff, dropout, has_cross_attention)
            for _ in range(num_layers)
        ])
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
    
    def forward(self, input_ids, encoder_output=None, attention_mask=None,
                return_attention_weights=False):
        """
        Args:
            input_ids: (batch_size, seq_len)
            encoder_output: (batch_size, src_seq_len, d_model) for cross-attention
            attention_mask: mask for encoder output
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.token_embedding(input_ids) * (self.d_model ** 0.5)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Pass through decoder blocks
        all_self_attn_weights = []
        all_cross_attn_weights = []
        
        for block in self.blocks:
            x, self_attn_weights, cross_attn_weights = block(
                x, encoder_output, None, attention_mask
            )
            
            if return_attention_weights:
                all_self_attn_weights.append(self_attn_weights)
                if cross_attn_weights is not None:
                    all_cross_attn_weights.append(cross_attn_weights)
        
        # Project to vocabulary
        logits = self.output_projection(x)
        
        if return_attention_weights:
            return logits, all_self_attn_weights, all_cross_attn_weights
        return logits

class GPTLikeModel(nn.Module):
    """
    GPT-style decoder-only model for language modeling
    """
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff=None,
                 max_seq_len=1024, dropout=0.1):
        super().__init__()
        
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            has_cross_attention=False  # Decoder-only
        )
    
    def forward(self, input_ids):
        """Language modeling forward pass"""
        return self.decoder(input_ids)
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, 
                 do_sample=True, top_k=None):
        """
        Generate text autoregressively
        
        Args:
            input_ids: (batch_size, seq_len) - initial tokens
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature (lower = more deterministic)
            do_sample: whether to sample or use greedy decoding
            top_k: if set, only sample from top k tokens
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Get predictions for next token
            logits = self.forward(input_ids)
            next_token_logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                # Apply top-k filtering
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                next_token_logits.scatter_(1, top_k_indices, top_k_logits)
            
            # Sample next token
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids

def visualize_causal_attention(attention_weights, tokens=None):
    """Visualize causal attention pattern"""
    # Take first batch, first head, first layer
    attn = attention_weights[0][0, 0].detach().numpy()
    
    plt.figure(figsize=(8, 6))
    plt.imshow(attn, cmap='Blues', aspect='auto')
    plt.colorbar()
    plt.title('Causal Attention Pattern')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    
    if tokens:
        plt.xticks(range(len(tokens)), tokens, rotation=45)
        plt.yticks(range(len(tokens)), tokens)
    
    plt.tight_layout()
    plt.show()

def compare_encoder_decoder_attention():
    """Compare encoder vs decoder attention patterns"""
    from lessons.full_transformer_encoder import TransformerEncoder
    
    vocab_size = 1000
    d_model = 128
    num_heads = 4
    seq_len = 8
    
    # Create encoder and decoder
    encoder = TransformerEncoder(vocab_size, d_model, num_heads, 2)
    decoder = TransformerDecoder(vocab_size, d_model, num_heads, 2)
    
    # Same input for both
    input_ids = torch.randint(1, vocab_size, (1, seq_len))
    
    # Get attention weights
    _, encoder_attn = encoder(input_ids, return_attention_weights=True)
    _, decoder_attn, _ = decoder(input_ids, return_attention_weights=True)
    
    # Visualize comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Encoder attention (bidirectional)
    enc_attn = encoder_attn[0][0, 0].detach().numpy()
    im1 = ax1.imshow(enc_attn, cmap='Blues', aspect='auto')
    ax1.set_title('Encoder Attention (Bidirectional)')
    ax1.set_xlabel('Key Position')
    ax1.set_ylabel('Query Position')
    plt.colorbar(im1, ax=ax1)
    
    # Decoder attention (causal)
    dec_attn = decoder_attn[0][0, 0].detach().numpy()
    im2 = ax2.imshow(dec_attn, cmap='Blues', aspect='auto')
    ax2.set_title('Decoder Attention (Causal)')
    ax2.set_xlabel('Key Position')
    ax2.set_ylabel('Query Position')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Example 1: Basic Decoder Block
    print("=== Example 1: Basic Decoder Block ===")
    
    d_model = 256
    num_heads = 8
    seq_len = 10
    batch_size = 2
    
    # Create decoder block
    decoder_block = TransformerDecoderBlock(d_model, num_heads)
    
    # Create input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Forward pass
    output, self_attn, cross_attn = decoder_block(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Self-attention weights shape: {self_attn.shape}")
    print(f"Cross-attention weights: {cross_attn}")
    
    # Example 2: GPT-style Language Model
    print("\n=== Example 2: GPT-style Language Model ===")
    
    vocab_size = 5000
    d_model = 256
    num_heads = 8
    num_layers = 4
    seq_len = 20
    
    gpt_model = GPTLikeModel(vocab_size, d_model, num_heads, num_layers)
    
    # Create input sequence
    input_ids = torch.randint(1, vocab_size, (2, seq_len))
    
    # Forward pass
    logits = gpt_model(input_ids)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in gpt_model.parameters()):,}")
    
    # Example 3: Text Generation
    print("\n=== Example 3: Text Generation ===")
    
    # Start with a prompt
    prompt = torch.tensor([[1, 2, 3, 4]])  # Simple prompt
    
    # Generate text
    generated = gpt_model.generate(
        prompt, 
        max_new_tokens=10,
        temperature=0.8,
        do_sample=True,
        top_k=50
    )
    
    print(f"Prompt: {prompt[0].tolist()}")
    print(f"Generated: {generated[0].tolist()}")
    print(f"New tokens: {generated[0][len(prompt[0]):].tolist()}")
    
    # Example 4: Causal Attention Visualization
    print("\n=== Example 4: Causal Attention Visualization ===")
    
    # Get attention weights
    sample_input = input_ids[:1, :8]  # Smaller for visualization
    _, self_attn_weights, _ = gpt_model.decoder(sample_input, return_attention_weights=True)
    
    tokens = [f"tok_{i}" for i in range(8)]
    visualize_causal_attention(self_attn_weights, tokens)
    
    # Example 5: Compare Encoder vs Decoder Attention
    print("\n=== Example 5: Encoder vs Decoder Attention Patterns ===")
    
    compare_encoder_decoder_attention()
    
    # Example 6: Cross-Attention Decoder (Encoder-Decoder)
    print("\n=== Example 6: Encoder-Decoder with Cross-Attention ===")
    
    # Create encoder-decoder setup
    encoder_decoder = TransformerDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=2,
        has_cross_attention=True
    )
    
    # Encoder output (simulated)
    src_seq_len = 12
    encoder_output = torch.randn(1, src_seq_len, d_model)
    
    # Decoder input
    tgt_input = torch.randint(1, vocab_size, (1, 8))
    
    # Forward with cross-attention
    logits, self_attn, cross_attn = encoder_decoder(
        tgt_input, 
        encoder_output=encoder_output,
        return_attention_weights=True
    )
    
    print(f"Target input shape: {tgt_input.shape}")
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Decoder output shape: {logits.shape}")
    print(f"Cross-attention weights shape: {cross_attn[0].shape}")
    
    # Visualize cross-attention
    plt.figure(figsize=(10, 6))
    cross_attn_viz = cross_attn[0][0, 0].detach().numpy()  # First layer, first head
    plt.imshow(cross_attn_viz, cmap='Blues', aspect='auto')
    plt.colorbar()
    plt.title('Cross-Attention: Decoder attending to Encoder')
    plt.xlabel('Encoder Position')
    plt.ylabel('Decoder Position')
    plt.show()
    
    # Example 7: Different Generation Strategies
    print("\n=== Example 7: Different Generation Strategies ===")
    
    prompt = torch.tensor([[1, 2, 3]])
    
    # Greedy decoding
    greedy_output = gpt_model.generate(
        prompt.clone(), 
        max_new_tokens=5,
        do_sample=False
    )
    
    # High temperature sampling
    high_temp_output = gpt_model.generate(
        prompt.clone(),
        max_new_tokens=5,
        temperature=2.0,
        do_sample=True
    )
    
    # Low temperature sampling  
    low_temp_output = gpt_model.generate(
        prompt.clone(),
        max_new_tokens=5,
        temperature=0.1,
        do_sample=True
    )
    
    print(f"Prompt: {prompt[0].tolist()}")
    print(f"Greedy: {greedy_output[0][3:].tolist()}")
    print(f"High temp: {high_temp_output[0][3:].tolist()}")
    print(f"Low temp: {low_temp_output[0][3:].tolist()}")
    
    print("\nTransformer decoder enables autoregressive generation!")
    print("Next: We'll combine encoder and decoder into a complete transformer.")