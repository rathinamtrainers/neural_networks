"""
Lesson 8: Transformer Training
==============================

Complete training pipeline for transformer models, covering:
- Data preparation and tokenization
- Training loop with proper optimization
- Learning rate scheduling  
- Evaluation and metrics
- Model checkpointing
- Practical training tips and best practices

This lesson demonstrates training a transformer on a simple task with 
realistic techniques used in practice.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import math
import time
from lessons.complete_transformer import TranslationModel
from lessons.transformer_decoder import GPTLikeModel

class SimpleTokenizer:
    """Simple tokenizer for demonstration"""
    
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        
    def encode(self, text, max_length=None):
        """Convert text to token ids (simplified)"""
        # In practice, this would use real tokenization
        # Here we just simulate with random tokens
        tokens = [self.bos_token_id]
        length = np.random.randint(5, 20) if max_length is None else max_length - 2
        tokens.extend(np.random.randint(4, self.vocab_size, length).tolist())
        tokens.append(self.eos_token_id)
        
        if max_length and len(tokens) < max_length:
            tokens.extend([self.pad_token_id] * (max_length - len(tokens)))
        elif max_length and len(tokens) > max_length:
            tokens = tokens[:max_length-1] + [self.eos_token_id]
            
        return tokens
    
    def decode(self, token_ids):
        """Convert token ids back to text (simplified)"""
        # Remove special tokens for readability
        clean_tokens = [t for t in token_ids if t > 3]
        return f"tokens_{len(clean_tokens)}"

class TranslationDataset(Dataset):
    """Dataset for machine translation"""
    
    def __init__(self, size=1000, src_vocab_size=1000, tgt_vocab_size=800, 
                 max_src_len=32, max_tgt_len=36):
        self.size = size
        self.src_tokenizer = SimpleTokenizer(src_vocab_size)
        self.tgt_tokenizer = SimpleTokenizer(tgt_vocab_size)
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Generate random source and target sequences
        src_tokens = self.src_tokenizer.encode(max_length=self.max_src_len)
        tgt_tokens = self.tgt_tokenizer.encode(max_length=self.max_tgt_len)
        
        # Prepare decoder input (shift right, start with BOS)
        tgt_input = [self.tgt_tokenizer.bos_token_id] + tgt_tokens[1:-1]
        tgt_labels = tgt_tokens[1:]  # Labels start from second token
        
        # Pad to max length
        while len(src_tokens) < self.max_src_len:
            src_tokens.append(self.src_tokenizer.pad_token_id)
        while len(tgt_input) < self.max_tgt_len - 1:
            tgt_input.append(self.tgt_tokenizer.pad_token_id)
        while len(tgt_labels) < self.max_tgt_len - 1:
            tgt_labels.append(self.tgt_tokenizer.pad_token_id)
        
        return {
            'src_input_ids': torch.tensor(src_tokens[:self.max_src_len], dtype=torch.long),
            'tgt_input_ids': torch.tensor(tgt_input[:self.max_tgt_len-1], dtype=torch.long),
            'tgt_labels': torch.tensor(tgt_labels[:self.max_tgt_len-1], dtype=torch.long),
        }

class LanguageModelingDataset(Dataset):
    """Dataset for causal language modeling (GPT-style)"""
    
    def __init__(self, size=1000, vocab_size=1000, max_seq_len=128):
        self.size = size
        self.tokenizer = SimpleTokenizer(vocab_size)
        self.max_seq_len = max_seq_len
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Generate random sequence
        tokens = self.tokenizer.encode(max_length=self.max_seq_len)
        
        # For language modeling, input and labels are the same (shifted by 1)
        input_ids = tokens[:-1]  # All tokens except last
        labels = tokens[1:]      # All tokens except first
        
        # Pad if necessary
        while len(input_ids) < self.max_seq_len - 1:
            input_ids.append(self.tokenizer.pad_token_id)
            labels.append(self.tokenizer.pad_token_id)
        
        return {
            'input_ids': torch.tensor(input_ids[:self.max_seq_len-1], dtype=torch.long),
            'labels': torch.tensor(labels[:self.max_seq_len-1], dtype=torch.long),
        }

class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine decay"""
    
    def __init__(self, optimizer, warmup_steps, total_steps, max_lr=1e-3, min_lr=1e-5):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0
        
    def step(self):
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            lr = self.max_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

class Trainer:
    """Training manager for transformer models"""
    
    def __init__(self, model, train_dataloader, val_dataloader, device='cpu'):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
    def setup_training(self, max_lr=1e-3, weight_decay=0.01, warmup_ratio=0.1, 
                      total_steps=1000):
        """Setup optimizer and scheduler"""
        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=max_lr, 
            weight_decay=weight_decay,
            betas=(0.9, 0.95)  # Common values for transformers
        )
        
        # Learning rate scheduler
        warmup_steps = int(total_steps * warmup_ratio)
        self.scheduler = WarmupCosineScheduler(
            self.optimizer, warmup_steps, total_steps, max_lr
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        
    def train_step(self, batch):
        """Single training step"""
        self.model.train()
        
        # Move batch to device
        for key in batch:
            batch[key] = batch[key].to(self.device)
        
        # Forward pass
        if 'tgt_input_ids' in batch:  # Translation task
            logits = self.model(batch['src_input_ids'], batch['tgt_input_ids'])
            labels = batch['tgt_labels']
        else:  # Language modeling
            logits = self.model(batch['input_ids'])
            labels = batch['labels']
        
        # Calculate loss
        loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update parameters
        self.optimizer.step()
        
        # Update learning rate
        current_lr = self.scheduler.step()
        
        self.step += 1
        return loss.item(), current_lr
    
    @torch.no_grad()
    def validation_step(self):
        """Validation on entire validation set"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        for batch in self.val_dataloader:
            # Move batch to device
            for key in batch:
                batch[key] = batch[key].to(self.device)
            
            # Forward pass
            if 'tgt_input_ids' in batch:  # Translation task
                logits = self.model(batch['src_input_ids'], batch['tgt_input_ids'])
                labels = batch['tgt_labels']
            else:  # Language modeling
                logits = self.model(batch['input_ids'])
                labels = batch['labels']
            
            # Calculate loss
            loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # Count non-padding tokens
            valid_tokens = (labels != 0).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')
        
        return avg_loss, perplexity
    
    def train(self, epochs=5, save_every=100, eval_every=50, log_every=10):
        """Main training loop"""
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            self.epoch = epoch
            epoch_losses = []
            
            print(f"\\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)
            
            for batch_idx, batch in enumerate(self.train_dataloader):
                # Training step
                loss, lr = self.train_step(batch)
                epoch_losses.append(loss)
                self.train_losses.append(loss)
                self.learning_rates.append(lr)
                
                # Logging
                if self.step % log_every == 0:
                    print(f"Step {self.step}: Loss = {loss:.4f}, LR = {lr:.6f}")
                
                # Evaluation
                if self.step % eval_every == 0:
                    val_loss, perplexity = self.validation_step()
                    self.val_losses.append(val_loss)
                    
                    print(f"Validation - Loss: {val_loss:.4f}, Perplexity: {perplexity:.2f}")
                    
                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint('best_model.pt')
                        print("New best model saved!")
                
                # Periodic checkpoint
                if self.step % save_every == 0:
                    self.save_checkpoint(f'checkpoint_step_{self.step}.pt')
            
            # Epoch summary
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"Epoch {epoch + 1} completed - Average Loss: {avg_epoch_loss:.4f}")
        
        total_time = time.time() - start_time
        print(f"\\nTraining completed in {total_time:.2f} seconds")
        
        return self.train_losses, self.val_losses
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        torch.save(checkpoint, filename)
    
    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']

def plot_training_curves(train_losses, val_losses, learning_rates):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(train_losses, label='Training Loss', alpha=0.7)
    
    # Plot validation loss at evaluation points
    val_steps = [i * (len(train_losses) // len(val_losses)) for i in range(len(val_losses))]
    ax1.plot(val_steps, val_losses, label='Validation Loss', marker='o')
    
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Learning rate curve
    ax2.plot(learning_rates, label='Learning Rate')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Example 1: Training a Translation Model
    print("=== Example 1: Training Translation Model ===")
    
    # Model configuration
    src_vocab_size = 1000
    tgt_vocab_size = 800
    d_model = 256
    num_heads = 8
    num_layers = 3
    
    # Create model
    translation_model = TranslationModel(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=0.1
    )
    
    print(f"Translation model parameters: {sum(p.numel() for p in translation_model.parameters()):,}")
    
    # Create datasets
    train_dataset = TranslationDataset(size=800, src_vocab_size=src_vocab_size, 
                                     tgt_vocab_size=tgt_vocab_size)
    val_dataset = TranslationDataset(size=200, src_vocab_size=src_vocab_size,
                                   tgt_vocab_size=tgt_vocab_size)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Setup trainer
    translation_trainer = Trainer(translation_model, train_dataloader, val_dataloader, device)
    translation_trainer.setup_training(max_lr=1e-3, total_steps=500)
    
    # Train model
    print("Starting translation model training...")
    train_losses, val_losses = translation_trainer.train(epochs=2, eval_every=25, log_every=5)
    
    # Plot results
    plot_training_curves(train_losses, val_losses, translation_trainer.learning_rates)
    
    # Example 2: Training a Language Model (GPT-style)
    print("\\n=== Example 2: Training Language Model ===")
    
    # Model configuration  
    vocab_size = 1000
    d_model = 256
    num_heads = 8
    num_layers = 4
    
    # Create GPT-style model
    language_model = GPTLikeModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=0.1
    )
    
    print(f"Language model parameters: {sum(p.numel() for p in language_model.parameters()):,}")
    
    # Create datasets
    lm_train_dataset = LanguageModelingDataset(size=600, vocab_size=vocab_size)
    lm_val_dataset = LanguageModelingDataset(size=150, vocab_size=vocab_size)
    
    # Create dataloaders
    lm_train_dataloader = DataLoader(lm_train_dataset, batch_size=8, shuffle=True)
    lm_val_dataloader = DataLoader(lm_val_dataset, batch_size=8, shuffle=False)
    
    # Setup trainer
    lm_trainer = Trainer(language_model, lm_train_dataloader, lm_val_dataloader, device)
    lm_trainer.setup_training(max_lr=5e-4, total_steps=400)
    
    # Train model
    print("Starting language model training...")
    lm_train_losses, lm_val_losses = lm_trainer.train(epochs=2, eval_every=20, log_every=5)
    
    # Plot results
    plot_training_curves(lm_train_losses, lm_val_losses, lm_trainer.learning_rates)
    
    # Example 3: Model Evaluation and Generation
    print("\\n=== Example 3: Model Evaluation and Generation ===")
    
    # Test generation with the trained language model
    language_model.eval()
    
    # Generate some text
    prompt = torch.tensor([[1, 5, 10, 15]], device=device)  # Simple prompt
    
    with torch.no_grad():
        generated = language_model.generate(
            prompt,
            max_new_tokens=20,
            temperature=0.8,
            do_sample=True,
            top_k=50
        )
    
    print(f"Generated sequence: {generated[0].cpu().tolist()}")
    
    # Test translation with the trained translation model
    translation_model.eval()
    
    # Create test source
    test_src = torch.randint(4, src_vocab_size, (1, 16), device=device)
    
    with torch.no_grad():
        translation = translation_model.translate(test_src, max_length=20)
    
    print(f"Source: {test_src[0].cpu().tolist()}")
    print(f"Translation: {translation[0].cpu().tolist()}")
    
    # Example 4: Learning Rate Schedule Visualization
    print("\\n=== Example 4: Learning Rate Schedule Analysis ===")
    
    # Create scheduler for visualization
    dummy_optimizer = torch.optim.Adam([torch.randn(1, requires_grad=True)], lr=1e-3)
    scheduler = WarmupCosineScheduler(dummy_optimizer, warmup_steps=100, 
                                    total_steps=1000, max_lr=1e-3, min_lr=1e-5)
    
    # Simulate learning rate schedule
    lrs = []
    for step in range(1000):
        lr = scheduler.step()
        lrs.append(lr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(lrs)
    plt.axvline(x=100, color='r', linestyle='--', label='End of Warmup')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule (Warmup + Cosine Decay)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Example 5: Training Tips Summary
    print("\\n=== Example 5: Training Best Practices Summary ===")
    
    training_tips = [
        "1. Use warmup for learning rate - prevents early training instability",
        "2. Apply gradient clipping (max_norm=1.0) to prevent exploding gradients",
        "3. Use weight decay (0.01-0.1) for regularization",
        "4. Monitor both loss and perplexity for language models",
        "5. Save checkpoints regularly and keep the best validation model",
        "6. Use appropriate batch sizes (depends on memory and model size)",
        "7. Consider mixed precision training for faster training on modern GPUs",
        "8. Validate on held-out data to monitor overfitting",
        "9. Use dropout during training but disable during inference",
        "10. Scale learning rate with batch size (linear scaling rule)"
    ]
    
    print("Transformer Training Best Practices:")
    for tip in training_tips:
        print(tip)
    
    print("\\nTransformer training pipeline complete!")
    print("You now have all the components to build and train transformer models from scratch!")