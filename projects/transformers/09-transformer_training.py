"""
Transformer Training Pipeline with Small Dataset

This module demonstrates how to train a transformer model on a small
synthetic dataset. It includes data preparation, training loop,
evaluation, and visualization of training progress.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
# from tqdm import tqdm  # Not available, using simple progress
import time


class SimpleSequenceDataset(Dataset):
    """
    A simple synthetic dataset for sequence-to-sequence tasks.
    
    The task: Reverse sequences with simple transformations.
    Example: [3, 5, 7] -> [7, 5, 3] or with arithmetic: [3, 5, 7] -> [8, 6, 4]
    """
    
    def __init__(self, num_samples=1000, max_seq_len=10, vocab_size=50, 
                 transform_type='reverse'):
        """
        Args:
            num_samples: Number of samples to generate
            max_seq_len: Maximum sequence length
            vocab_size: Vocabulary size (excluding special tokens)
            transform_type: Type of transformation ('reverse', 'add_one', 'sort')
        """
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.transform_type = transform_type
        
        # Special tokens
        self.pad_idx = 0
        self.bos_idx = 1
        self.eos_idx = 2
        
        # Generate data
        self.data = self._generate_data()
        
    def _generate_data(self):
        """Generate synthetic sequence pairs."""
        data = []
        
        for _ in range(self.num_samples):
            # Random sequence length
            seq_len = np.random.randint(3, self.max_seq_len + 1)
            
            # Generate source sequence (random tokens from 3 onwards)
            src_seq = np.random.randint(3, self.vocab_size, seq_len)
            
            # Apply transformation
            if self.transform_type == 'reverse':
                tgt_seq = src_seq[::-1]
            elif self.transform_type == 'add_one':
                tgt_seq = (src_seq + 1) % self.vocab_size
                tgt_seq[tgt_seq < 3] = 3  # Avoid special tokens
            elif self.transform_type == 'sort':
                tgt_seq = np.sort(src_seq)
            else:
                tgt_seq = src_seq  # Copy task
            
            # Add special tokens
            src_seq = np.concatenate([[self.bos_idx], src_seq, [self.eos_idx]])
            tgt_seq = np.concatenate([[self.bos_idx], tgt_seq, [self.eos_idx]])
            
            data.append((src_seq, tgt_seq))
        
        return data
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        return torch.tensor(src), torch.tensor(tgt)
    
    def collate_fn(self, batch):
        """Custom collate function for padding sequences."""
        srcs, tgts = zip(*batch)
        
        # Find max lengths
        max_src_len = max(len(s) for s in srcs)
        max_tgt_len = max(len(t) for t in tgts)
        
        # Pad sequences
        src_batch = []
        tgt_batch = []
        
        for src, tgt in zip(srcs, tgts):
            # Pad source
            src_padded = F.pad(src, (0, max_src_len - len(src)), value=self.pad_idx)
            src_batch.append(src_padded)
            
            # Pad target
            tgt_padded = F.pad(tgt, (0, max_tgt_len - len(tgt)), value=self.pad_idx)
            tgt_batch.append(tgt_padded)
        
        return torch.stack(src_batch), torch.stack(tgt_batch)


class TransformerSeq2Seq(nn.Module):
    """Simplified transformer for sequence-to-sequence tasks."""
    
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, 
                 num_layers=2, ff_dim=512, dropout=0.1):
        super(TransformerSeq2Seq, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = self._create_positional_encoding(100, embed_dim)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def _create_positional_encoding(self, max_len, embed_dim):
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                            -(np.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def create_masks(self, src, tgt):
        """Create masks for padding and causal attention."""
        # Padding masks
        src_padding_mask = (src == 0)
        tgt_padding_mask = (tgt == 0)
        
        # Causal mask for target
        tgt_len = tgt.size(1)
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool()
        tgt_mask = tgt_mask.to(tgt.device)
        
        return src_padding_mask, tgt_padding_mask, tgt_mask
    
    def forward(self, src, tgt):
        """Forward pass through the model."""
        # Create masks
        src_padding_mask, tgt_padding_mask, tgt_mask = self.create_masks(src, tgt)
        
        # Embed and add positional encoding
        src_emb = self.embedding(src) * np.sqrt(self.embed_dim)
        src_emb = src_emb + self.pos_encoding[:, :src.size(1)]
        src_emb = self.dropout(src_emb)
        
        tgt_emb = self.embedding(tgt) * np.sqrt(self.embed_dim)
        tgt_emb = tgt_emb + self.pos_encoding[:, :tgt.size(1)]
        tgt_emb = self.dropout(tgt_emb)
        
        # Pass through transformer
        output = self.transformer(
            src_emb, tgt_emb,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            tgt_mask=tgt_mask
        )
        
        # Project to vocabulary
        output = self.output_proj(output)
        
        return output
    
    @torch.no_grad()
    def generate(self, src, max_length=20):
        """Generate output sequence."""
        self.eval()
        device = src.device
        batch_size = src.size(0)
        
        # Start with BOS token
        generated = torch.ones(batch_size, 1, dtype=torch.long).to(device)
        
        for _ in range(max_length - 1):
            # Get prediction for next token
            output = self.forward(src, generated)
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check if all sequences have generated EOS
            if (next_token == 2).all():
                break
        
        return generated


def train_transformer(model, train_loader, val_loader, num_epochs=20, 
                     learning_rate=0.001, device='cpu'):
    """Train the transformer model."""
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print("Starting training...")
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_steps = 0
        
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            
            # Teacher forcing: use ground truth as input
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # Forward pass
            output = model(src, tgt_input)
            
            # Calculate loss
            loss = criterion(
                output.reshape(-1, model.vocab_size),
                tgt_output.reshape(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / train_steps
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_steps = 0
        correct_sequences = 0
        total_sequences = 0
        
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                # Forward pass
                output = model(src, tgt_input)
                
                # Calculate loss
                loss = criterion(
                    output.reshape(-1, model.vocab_size),
                    tgt_output.reshape(-1)
                )
                
                val_loss += loss.item()
                val_steps += 1
                
                # Calculate accuracy (exact sequence match)
                generated = model.generate(src, max_length=tgt.size(1))
                
                # Compare generated sequences with targets
                for gen, target in zip(generated, tgt):
                    # Find actual length (before padding)
                    gen_len = (gen != 0).sum()
                    tgt_len = (target != 0).sum()
                    
                    if gen_len == tgt_len and torch.equal(gen[:gen_len], target[:tgt_len]):
                        correct_sequences += 1
                    total_sequences += 1
        
        avg_val_loss = val_loss / val_steps
        val_accuracy = correct_sequences / total_sequences
        
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2%}")
    
    return train_losses, val_losses, val_accuracies


def visualize_training_results(train_losses, val_losses, val_accuracies):
    """Visualize training progress."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(val_accuracies, 'g-')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy (Exact Sequence Match)')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def test_model(model, test_dataset, device='cpu'):
    """Test the model with examples."""
    
    print("\nModel Testing")
    print("=" * 50)
    
    model.eval()
    model = model.to(device)
    
    # Test on random samples
    num_examples = 5
    indices = np.random.choice(len(test_dataset), num_examples)
    
    for idx in indices:
        src, tgt = test_dataset[idx]
        src = src.unsqueeze(0).to(device)
        tgt = tgt.unsqueeze(0).to(device)
        
        # Generate output
        with torch.no_grad():
            generated = model.generate(src, max_length=20)
        
        # Convert to lists for display
        src_tokens = src[0].cpu().tolist()
        tgt_tokens = tgt[0].cpu().tolist()
        gen_tokens = generated[0].cpu().tolist()
        
        # Remove padding
        src_tokens = [t for t in src_tokens if t != 0]
        tgt_tokens = [t for t in tgt_tokens if t != 0]
        gen_tokens = [t for t in gen_tokens if t != 0]
        
        print(f"\nExample {idx}:")
        print(f"Source:    {src_tokens}")
        print(f"Target:    {tgt_tokens}")
        print(f"Generated: {gen_tokens}")
        print(f"Correct:   {gen_tokens == tgt_tokens}")


def analyze_attention_patterns(model, dataset, device='cpu'):
    """Analyze attention patterns in the trained model."""
    
    print("\nAttention Pattern Analysis")
    print("=" * 50)
    
    model.eval()
    model = model.to(device)
    
    # Get a sample
    src, tgt = dataset[0]
    src = src.unsqueeze(0).to(device)
    tgt = tgt.unsqueeze(0).to(device)
    
    # Forward pass with hooks to capture attention
    attention_weights = []
    
    def hook_fn(module, input, output):
        # For nn.Transformer, we need to access internal attention weights
        # This is a simplified visualization
        pass
    
    # Since PyTorch's nn.Transformer doesn't easily expose attention weights,
    # we'll visualize the learned embeddings instead
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Visualize embeddings
    ax = axes[0]
    embeddings = model.embedding.weight.detach().cpu().numpy()
    
    # Use PCA to reduce to 2D
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Plot special tokens differently
    ax.scatter(embeddings_2d[3:, 0], embeddings_2d[3:, 1], alpha=0.5, label='Regular tokens')
    ax.scatter(embeddings_2d[0, 0], embeddings_2d[0, 1], color='red', s=100, label='PAD')
    ax.scatter(embeddings_2d[1, 0], embeddings_2d[1, 1], color='green', s=100, label='BOS')
    ax.scatter(embeddings_2d[2, 0], embeddings_2d[2, 1], color='blue', s=100, label='EOS')
    
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    ax.set_title('Token Embeddings (PCA)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Visualize output distribution
    ax = axes[1]
    
    # Get model output for the sample
    with torch.no_grad():
        tgt_input = tgt[:, :-1]
        output = model(src, tgt_input)
        probs = F.softmax(output[0, -1], dim=-1).cpu().numpy()
    
    # Plot top-k probabilities
    top_k = 20
    top_indices = np.argsort(probs)[-top_k:][::-1]
    top_probs = probs[top_indices]
    
    ax.bar(range(top_k), top_probs)
    ax.set_xlabel('Token Index')
    ax.set_ylabel('Probability')
    ax.set_title('Output Probability Distribution (Last Position)')
    ax.set_xticks(range(0, top_k, 5))
    ax.set_xticklabels(top_indices[::5])
    
    plt.tight_layout()
    plt.show()


def main():
    """Main training pipeline."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = SimpleSequenceDataset(
        num_samples=2000, 
        transform_type='reverse'
    )
    val_dataset = SimpleSequenceDataset(
        num_samples=500, 
        transform_type='reverse'
    )
    test_dataset = SimpleSequenceDataset(
        num_samples=100, 
        transform_type='reverse'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False,
        collate_fn=val_dataset.collate_fn
    )
    
    # Create model
    print("Creating model...")
    model = TransformerSeq2Seq(
        vocab_size=50,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        ff_dim=512,
        dropout=0.1
    )
    
    # Train model
    train_losses, val_losses, val_accuracies = train_transformer(
        model, train_loader, val_loader, 
        num_epochs=20, 
        learning_rate=0.001,
        device=device
    )
    
    # Visualize results
    visualize_training_results(train_losses, val_losses, val_accuracies)
    
    # Test model
    test_model(model, test_dataset, device)
    
    # Analyze attention
    analyze_attention_patterns(model, test_dataset, device)
    
    print("\nTraining complete!")
    print(f"Final validation accuracy: {val_accuracies[-1]:.2%}")


if __name__ == "__main__":
    # Check for sklearn (for PCA visualization)
    try:
        import sklearn
        main()
    except ImportError:
        print("Please install scikit-learn for full visualization: pip install scikit-learn")
        print("Running without PCA visualization...")
        main()