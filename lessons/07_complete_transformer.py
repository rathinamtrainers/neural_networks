"""
Lesson 7: Complete Transformer Model
===================================

Bringing together encoder and decoder into the full transformer architecture
from "Attention Is All You Need". This creates the classic seq2seq model used for:
- Machine translation
- Text summarization  
- Question answering
- Any sequence-to-sequence task

Key concepts:
- Encoder-decoder architecture
- Cross-attention between encoder and decoder
- Teacher forcing during training
- Inference strategies for generation
- Model scaling and architecture variants
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from lessons.full_transformer_encoder import TransformerEncoder
from lessons.transformer_decoder import TransformerDecoder
from lessons.positional_encoding import SinusoidalPositionalEncoding

class Transformer(nn.Module):
    """
    Complete Transformer model (encoder-decoder)
    From "Attention Is All You Need"
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048,
                 max_seq_len=5000, dropout=0.1, pad_token_id=0):
        super().__init__()
        
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        
        # Encoder
        self.encoder = TransformerEncoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            pad_token_id=pad_token_id
        )
        
        # Decoder with cross-attention
        self.decoder = TransformerDecoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            has_cross_attention=True,
            pad_token_id=pad_token_id
        )
        
    def create_padding_mask(self, input_ids):
        """Create mask to ignore padding tokens"""
        return (input_ids != self.pad_token_id).unsqueeze(1).unsqueeze(2)
    
    def forward(self, src_input_ids, tgt_input_ids, src_attention_mask=None, 
                return_attention_weights=False):
        """
        Args:
            src_input_ids: (batch_size, src_seq_len) - source sequence
            tgt_input_ids: (batch_size, tgt_seq_len) - target sequence 
            src_attention_mask: mask for source padding
        """
        # Create source attention mask if not provided
        if src_attention_mask is None:
            src_attention_mask = self.create_padding_mask(src_input_ids)
        
        # Encode source sequence
        if return_attention_weights:
            encoder_output, encoder_attention = self.encoder(
                src_input_ids, src_attention_mask, return_attention_weights=True
            )
        else:
            encoder_output = self.encoder(src_input_ids, src_attention_mask)
            encoder_attention = None
        
        # Decode target sequence with cross-attention to encoder
        if return_attention_weights:
            decoder_output, self_attention, cross_attention = self.decoder(
                tgt_input_ids, encoder_output, src_attention_mask, 
                return_attention_weights=True
            )
            return decoder_output, encoder_attention, self_attention, cross_attention
        else:
            decoder_output = self.decoder(tgt_input_ids, encoder_output, src_attention_mask)
            return decoder_output

class TranslationModel(nn.Module):
    """
    Complete machine translation model with utilities
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_layers=6, d_ff=2048, dropout=0.1, 
                 bos_token_id=1, eos_token_id=2, pad_token_id=0):
        super().__init__()
        
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        
        self.transformer = Transformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
            pad_token_id=pad_token_id
        )
        
    def forward(self, src_input_ids, tgt_input_ids, src_attention_mask=None):
        """Training forward pass with teacher forcing"""
        return self.transformer(src_input_ids, tgt_input_ids, src_attention_mask)
    
    @torch.no_grad()
    def translate(self, src_input_ids, src_attention_mask=None, max_length=100, 
                  beam_size=1, temperature=1.0):
        """
        Translate source sequence to target sequence
        
        Args:
            src_input_ids: (batch_size, src_seq_len)
            max_length: maximum target length
            beam_size: beam search width (1 = greedy)
            temperature: sampling temperature
        """
        self.eval()
        batch_size = src_input_ids.size(0)
        device = src_input_ids.device
        
        # Encode source
        if src_attention_mask is None:
            src_attention_mask = self.transformer.create_padding_mask(src_input_ids)
        
        encoder_output = self.transformer.encoder(src_input_ids, src_attention_mask)
        
        if beam_size == 1:
            return self._greedy_decode(encoder_output, src_attention_mask, 
                                     batch_size, device, max_length, temperature)
        else:
            return self._beam_search(encoder_output, src_attention_mask,
                                   batch_size, device, max_length, beam_size)
    
    def _greedy_decode(self, encoder_output, src_attention_mask, batch_size, 
                      device, max_length, temperature):
        """Greedy decoding"""
        # Start with BOS token
        tgt_input_ids = torch.full((batch_size, 1), self.bos_token_id, 
                                  dtype=torch.long, device=device)
        
        for _ in range(max_length):
            # Get next token logits
            logits = self.transformer.decoder(tgt_input_ids, encoder_output, 
                                            src_attention_mask)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Sample next token
            if temperature == 1.0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            # Append token
            tgt_input_ids = torch.cat([tgt_input_ids, next_token], dim=1)
            
            # Check if all sequences have EOS
            if (next_token == self.eos_token_id).all():
                break
                
        return tgt_input_ids
    
    def _beam_search(self, encoder_output, src_attention_mask, batch_size, 
                    device, max_length, beam_size):
        """Simple beam search implementation"""
        # This is a simplified version - real beam search is more complex
        
        # Start with BOS token
        sequences = torch.full((batch_size, beam_size, 1), self.bos_token_id,
                              dtype=torch.long, device=device)
        scores = torch.zeros(batch_size, beam_size, device=device)
        
        for _ in range(max_length):
            all_candidates = []
            
            for batch_idx in range(batch_size):
                candidates = []
                
                for beam_idx in range(beam_size):
                    seq = sequences[batch_idx, beam_idx].unsqueeze(0)
                    
                    # Get logits for this sequence
                    logits = self.transformer.decoder(
                        seq, 
                        encoder_output[batch_idx:batch_idx+1], 
                        src_attention_mask[batch_idx:batch_idx+1] if src_attention_mask is not None else None
                    )
                    next_token_logits = F.log_softmax(logits[0, -1], dim=-1)
                    
                    # Get top-k candidates
                    top_logits, top_indices = torch.topk(next_token_logits, beam_size)
                    
                    for k in range(beam_size):
                        new_seq = torch.cat([seq[0], top_indices[k:k+1]])
                        new_score = scores[batch_idx, beam_idx] + top_logits[k]
                        candidates.append((new_score, new_seq))
                
                # Select top beam_size candidates
                candidates.sort(key=lambda x: x[0], reverse=True)
                all_candidates.append(candidates[:beam_size])
            
            # Update sequences and scores
            for batch_idx in range(batch_size):
                for beam_idx in range(beam_size):
                    score, seq = all_candidates[batch_idx][beam_idx]
                    sequences[batch_idx, beam_idx] = seq
                    scores[batch_idx, beam_idx] = score
        
        # Return best sequence for each batch
        best_sequences = sequences[:, 0, :]  # Take best beam
        return best_sequences

def create_translation_data(batch_size=4, src_seq_len=10, tgt_seq_len=12, 
                           src_vocab_size=1000, tgt_vocab_size=800):
    """Create dummy translation data"""
    # Avoid special tokens (0=PAD, 1=BOS, 2=EOS)
    src_input_ids = torch.randint(3, src_vocab_size, (batch_size, src_seq_len))
    tgt_input_ids = torch.randint(3, tgt_vocab_size, (batch_size, tgt_seq_len))
    
    # Add BOS to target input
    bos_tokens = torch.full((batch_size, 1), 1)  # BOS token
    tgt_input = torch.cat([bos_tokens, tgt_input_ids[:, :-1]], dim=1)
    
    # Target labels (with EOS)
    eos_tokens = torch.full((batch_size, 1), 2)  # EOS token  
    tgt_labels = torch.cat([tgt_input_ids, eos_tokens], dim=1)
    
    return src_input_ids, tgt_input, tgt_labels

def visualize_cross_attention(cross_attention_weights, src_tokens=None, 
                             tgt_tokens=None, layer_idx=0, head_idx=0):
    """Visualize cross-attention between source and target"""
    # Extract specific layer and head
    attn = cross_attention_weights[layer_idx][0, head_idx].detach().numpy()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(attn, cmap='Blues', aspect='auto')
    plt.colorbar()
    plt.title(f'Cross-Attention (Layer {layer_idx+1}, Head {head_idx+1})')
    plt.xlabel('Source Position')
    plt.ylabel('Target Position')
    
    if src_tokens and tgt_tokens:
        plt.xticks(range(len(src_tokens)), src_tokens, rotation=45)
        plt.yticks(range(len(tgt_tokens)), tgt_tokens)
    
    plt.tight_layout()
    plt.show()

def calculate_bleu_score(predictions, references):
    """Simplified BLEU score calculation"""
    # This is a very simplified version - real BLEU is more complex
    scores = []
    
    for pred, ref in zip(predictions, references):
        # Convert to lists if tensors
        if torch.is_tensor(pred):
            pred = pred.tolist()
        if torch.is_tensor(ref):
            ref = ref.tolist()
        
        # Remove special tokens for evaluation
        pred_clean = [token for token in pred if token > 2]  # Remove PAD, BOS, EOS
        ref_clean = [token for token in ref if token > 2]
        
        # Simple word-level precision
        if len(pred_clean) == 0:
            scores.append(0.0)
            continue
            
        matches = sum(1 for token in pred_clean if token in ref_clean)
        precision = matches / len(pred_clean)
        scores.append(precision)
    
    return sum(scores) / len(scores) if scores else 0.0

if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Example 1: Complete Transformer Model
    print("=== Example 1: Complete Transformer Model ===")
    
    src_vocab_size = 1000
    tgt_vocab_size = 800
    d_model = 256
    num_heads = 8
    num_layers = 3
    
    transformer = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers
    )
    
    # Create sample data
    src_input, tgt_input, tgt_labels = create_translation_data()
    
    # Forward pass
    output = transformer(src_input, tgt_input)
    
    print(f"Source input shape: {src_input.shape}")
    print(f"Target input shape: {tgt_input.shape}")
    print(f"Output logits shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in transformer.parameters()):,}")
    
    # Example 2: Translation Model with Utilities
    print("\n=== Example 2: Translation Model ===")
    
    translation_model = TranslationModel(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers
    )
    
    # Training forward pass (teacher forcing)
    train_output = translation_model(src_input, tgt_input)
    
    # Calculate loss
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    loss = loss_fn(train_output.view(-1, tgt_vocab_size), tgt_labels.view(-1))
    
    print(f"Training loss: {loss.item():.4f}")
    
    # Example 3: Translation/Inference
    print("\n=== Example 3: Translation Inference ===")
    
    # Translate a single sequence
    single_src = src_input[:1]  # Take first sequence
    
    # Greedy decoding
    translation_greedy = translation_model.translate(single_src, beam_size=1)
    
    # Beam search
    translation_beam = translation_model.translate(single_src, beam_size=3)
    
    print(f"Source: {single_src[0].tolist()}")
    print(f"Greedy translation: {translation_greedy[0].tolist()}")
    print(f"Beam search translation: {translation_beam[0].tolist()}")
    
    # Example 4: Attention Visualization
    print("\n=== Example 4: Cross-Attention Visualization ===")
    
    # Get attention weights
    output, enc_attn, dec_self_attn, dec_cross_attn = transformer(
        src_input[:1], tgt_input[:1], return_attention_weights=True
    )
    
    print(f"Encoder attention layers: {len(enc_attn)}")
    print(f"Decoder self-attention layers: {len(dec_self_attn)}")
    print(f"Decoder cross-attention layers: {len(dec_cross_attn)}")
    
    # Visualize cross-attention
    src_tokens = [f"src_{i}" for i in range(src_input.shape[1])]
    tgt_tokens = [f"tgt_{i}" for i in range(tgt_input.shape[1])]
    
    visualize_cross_attention(dec_cross_attn, src_tokens, tgt_tokens)
    
    # Example 5: Model Scaling Analysis
    print("\n=== Example 5: Model Scaling Analysis ===")
    
    # Compare different model sizes
    model_configs = [
        {"d_model": 128, "num_heads": 4, "num_layers": 2, "name": "Small"},
        {"d_model": 256, "num_heads": 8, "num_layers": 4, "name": "Medium"},
        {"d_model": 512, "num_heads": 8, "num_layers": 6, "name": "Large"},
    ]
    
    for config in model_configs:
        model = Transformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            num_encoder_layers=config["num_layers"],
            num_decoder_layers=config["num_layers"]
        )
        
        params = sum(p.numel() for p in model.parameters())
        memory_mb = params * 4 / (1024**2)  # Rough memory estimate
        
        print(f"{config['name']} model: {params:,} parameters, ~{memory_mb:.1f}MB")
    
    # Example 6: Training Loop Simulation
    print("\n=== Example 6: Training Loop Simulation ===")
    
    # Simple training simulation
    model = translation_model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Simulate a few training steps
    model.train()
    for step in range(3):
        # Get batch
        src_batch, tgt_input_batch, tgt_label_batch = create_translation_data()
        
        # Forward pass
        logits = model(src_batch, tgt_input_batch)
        
        # Calculate loss
        loss = loss_fn(logits.view(-1, tgt_vocab_size), tgt_label_batch.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (common in transformer training)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        print(f"Step {step+1}: Loss = {loss.item():.4f}")
    
    # Example 7: Evaluation with BLEU
    print("\n=== Example 7: Evaluation with BLEU Score ===")
    
    model.eval()
    
    # Generate translations for evaluation
    eval_src, _, eval_tgt = create_translation_data(batch_size=8)
    
    with torch.no_grad():
        predictions = model.translate(eval_src)
    
    # Calculate BLEU score
    bleu_score = calculate_bleu_score(predictions, eval_tgt)
    print(f"BLEU score: {bleu_score:.4f}")
    
    # Show some examples
    print("\nTranslation examples:")
    for i in range(3):
        print(f"Source: {eval_src[i][:8].tolist()}")  # First 8 tokens
        print(f"Reference: {eval_tgt[i][:8].tolist()}")
        print(f"Prediction: {predictions[i][:8].tolist()}")
        print()
    
    print("Complete transformer model ready for seq2seq tasks!")
    print("Next: We'll create a comprehensive training example.")