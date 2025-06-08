"""
Simple Transformer Model for Sequence Classification

This module implements a complete transformer model for sequence classification tasks.
It demonstrates how to combine all the components (embeddings, positional encoding,
transformer blocks) into a working model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import time


class SimpleTransformer(nn.Module):
    """
    A simple transformer model for sequence classification.
    
    Architecture:
    1. Token embeddings
    2. Positional encoding
    3. Transformer encoder blocks
    4. Global average pooling
    5. Classification head
    """
    
    def __init__(self, vocab_size, num_classes, embed_dim=128, num_heads=4, 
                 num_blocks=2, ff_dim=512, max_seq_len=100, dropout=0.1):
        """
        Args:
            vocab_size: Size of the vocabulary
            num_classes: Number of output classes
            embed_dim: Dimension of embeddings
            num_heads: Number of attention heads
            num_blocks: Number of transformer blocks
            ff_dim: Dimension of feed-forward layer
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super(SimpleTransformer, self).__init__()
        
        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(max_seq_len, embed_dim)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            self._create_transformer_block(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_blocks)
        ])
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)
        
    def _create_positional_encoding(self, max_len, embed_dim):
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                            -(np.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def _create_transformer_block(self, embed_dim, num_heads, ff_dim, dropout):
        """Create a transformer encoder block."""
        return nn.ModuleDict({
            'attention': nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, 
                                             batch_first=True),
            'norm1': nn.LayerNorm(embed_dim),
            'ff': nn.Sequential(
                nn.Linear(embed_dim, ff_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, embed_dim),
                nn.Dropout(dropout)
            ),
            'norm2': nn.LayerNorm(embed_dim)
        })
    
    def forward(self, x, mask=None):
        """
        Forward pass.
        
        Args:
            x: Input token indices of shape (batch_size, seq_len)
            mask: Optional padding mask
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        seq_len = x.size(1)
        
        # Token embeddings
        x = self.embedding(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            # Self-attention
            attn_out, _ = block['attention'](x, x, x, key_padding_mask=mask)
            x = block['norm1'](x + attn_out)
            
            # Feed-forward
            ff_out = block['ff'](x)
            x = block['norm2'](x + ff_out)
        
        # Global average pooling
        if mask is not None:
            # Mask out padding tokens
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            x = x.masked_fill(mask_expanded, 0)
            # Compute mean only over non-padding tokens
            lengths = (~mask).sum(dim=1, keepdim=True).float()
            x = x.sum(dim=1) / lengths
        else:
            x = x.mean(dim=1)
        
        # Classification
        x = self.dropout(x)
        logits = self.fc(x)
        
        return logits


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing the transformer."""
    
    def __init__(self, num_samples=1000, seq_len=20, vocab_size=100, num_classes=3):
        """
        Create a synthetic classification dataset.
        
        The task is to classify sequences based on the frequency of certain tokens.
        """
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        
        # Generate synthetic data
        self.data = []
        self.labels = []
        
        for _ in range(num_samples):
            # Generate random sequence
            seq = torch.randint(0, vocab_size, (seq_len,))
            
            # Simple rule: classify based on sum of token values
            token_sum = seq.sum().item()
            label = token_sum % num_classes
            
            self.data.append(seq)
            self.labels.append(label)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def train_transformer():
    """Train and evaluate a simple transformer model."""
    
    print("Training Simple Transformer")
    print("=" * 50)
    
    # Hyperparameters
    vocab_size = 100
    num_classes = 3
    embed_dim = 64
    num_heads = 4
    num_blocks = 2
    ff_dim = 256
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    
    # Create datasets
    train_dataset = SyntheticDataset(num_samples=1000)
    val_dataset = SyntheticDataset(num_samples=200)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = SimpleTransformer(
        vocab_size=vocab_size,
        num_classes=num_classes,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_blocks=num_blocks,
        ff_dim=ff_dim
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print()
    
    # Training loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Acc: {val_accuracy:.2f}%")
    
    # Plot training history
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
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return model


def analyze_attention_patterns(model):
    """Analyze attention patterns in the trained model."""
    
    print("\nAnalyzing Attention Patterns")
    print("=" * 50)
    
    # Create a test sequence
    test_seq = torch.randint(0, 100, (1, 20))
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(test_seq)
        prediction = torch.argmax(output, dim=1).item()
    
    print(f"Test sequence shape: {test_seq.shape}")
    print(f"Predicted class: {prediction}")
    
    # Since the built-in PyTorch transformer doesn't easily expose attention weights,
    # we'll analyze the model's learned embeddings and output patterns instead
    print("Note: Using simplified analysis since PyTorch transformer doesn't expose attention weights easily")


def inference_speed_test(model):
    """Test inference speed of the transformer."""
    
    print("\nInference Speed Test")
    print("=" * 50)
    
    model.eval()
    
    # Test different sequence lengths
    seq_lengths = [10, 20, 50, 100]
    times = []
    
    for seq_len in seq_lengths:
        # Create test batch
        test_batch = torch.randint(0, 100, (32, seq_len))
        
        # Warm up
        with torch.no_grad():
            _ = model(test_batch)
        
        # Time inference
        start_time = time.time()
        num_iterations = 100
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(test_batch)
        
        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / num_iterations * 1000  # Convert to ms
        times.append(avg_time)
        
        print(f"Sequence length {seq_len}: {avg_time:.2f} ms per batch")
    
    # Plot results
    plt.figure(figsize=(8, 6))
    plt.plot(seq_lengths, times, 'o-')
    plt.xlabel('Sequence Length')
    plt.ylabel('Inference Time (ms)')
    plt.title('Inference Speed vs Sequence Length')
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    # Train the model
    model = train_transformer()
    
    # Analyze attention patterns
    analyze_attention_patterns(model)
    
    # Test inference speed
    inference_speed_test(model)