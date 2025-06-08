"""
Lesson 1: Basic Attention Mechanism
===================================

The attention mechanism is the core building block of transformers.
It allows the model to focus on different parts of the input when processing each element.

Key concepts:
- Query (Q): What we're looking for
- Key (K): What we're comparing against  
- Value (V): What we actually use if there's a match
- Attention weights: How much to focus on each element
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def basic_attention(query, key, value):
    """
    Basic attention mechanism implementation
    
    Args:
        query: (batch_size, seq_len, d_model)
        key: (batch_size, seq_len, d_model)  
        value: (batch_size, seq_len, d_model)
    
    Returns:
        output: (batch_size, seq_len, d_model)
        attention_weights: (batch_size, seq_len, seq_len)
    """
    # Calculate attention scores
    # scores = Q * K^T / sqrt(d_k)
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    # Apply softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)
    
    # Apply attention weights to values
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights

class SimpleAttention(nn.Module):
    """Simple attention layer"""
    
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        # Generate Q, K, V from input
        query = self.query_linear(x)
        key = self.key_linear(x)
        value = self.value_linear(x)
        
        # Apply attention
        output, attention_weights = basic_attention(query, key, value)
        
        return output, attention_weights

def visualize_attention(attention_weights, tokens=None):
    """Visualize attention weights as a heatmap"""
    # Take first batch and first head
    attn = attention_weights[0].detach().numpy()
    
    plt.figure(figsize=(8, 6))
    plt.imshow(attn, cmap='Blues', aspect='auto')
    plt.colorbar()
    plt.title('Attention Weights')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    
    if tokens:
        plt.xticks(range(len(tokens)), tokens, rotation=45)
        plt.yticks(range(len(tokens)), tokens)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Example 1: Basic attention with simple vectors
    print("=== Example 1: Basic Attention ===")
    
    batch_size, seq_len, d_model = 1, 4, 6
    
    # Create sample input (representing "The cat sat on")
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Create attention layer
    attention_layer = SimpleAttention(d_model)
    
    # Forward pass
    output, attn_weights = attention_layer(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Attention weights:\n{attn_weights[0]}")
    
    # Example 2: Demonstrate attention focusing
    print("\n=== Example 2: Attention Focusing ===")
    
    # Create input where last token is very different
    x_focused = torch.zeros(1, 4, 6)
    x_focused[0, -1, :] = 5.0  # Make last token stand out
    
    output_focused, attn_weights_focused = attention_layer(x_focused)
    
    print("Attention weights when last token is prominent:")
    print(attn_weights_focused[0])
    
    # Example 3: Visualize attention
    print("\n=== Example 3: Attention Visualization ===")
    tokens = ["The", "cat", "sat", "on"]
    visualize_attention(attn_weights, tokens)
    
    # Show how attention weights sum to 1
    print(f"Sum of attention weights per query: {attn_weights[0].sum(dim=-1)}")