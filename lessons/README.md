# Transformer Architecture Lessons

A comprehensive step-by-step guide to understanding and implementing transformer architectures using PyTorch from scratch.

## Lesson Overview

### 01. Basic Attention Mechanism (`01_attention_mechanism.py`)
- **Core concepts**: Query, Key, Value, attention weights
- **Implementation**: Simple attention function and layer
- **Visualization**: Attention weight heatmaps
- **Examples**: Basic attention patterns and focusing behavior

### 02. Multi-Head Attention (`02_multi_head_attention.py`) 
- **Core concepts**: Parallel attention heads, scaled dot-product attention
- **Implementation**: Multi-head attention layer with linear projections
- **Features**: Masking support, self-attention
- **Examples**: Comparing single vs multi-head attention patterns

### 03. Positional Encoding (`03_positional_encoding.py`)
- **Core concepts**: Why position matters in transformers
- **Implementations**: 
  - Sinusoidal positional encoding (original Transformer)
  - Learned positional embeddings
  - Relative positional encoding
- **Examples**: Position similarity patterns and their impact on attention

### 04. Transformer Encoder Block (`04_transformer_encoder_block.py`)
- **Core concepts**: Residual connections, layer normalization, feed-forward networks
- **Implementations**: 
  - Post-norm encoder block (original)
  - Pre-norm encoder block (modern)
- **Features**: Attention analysis across layers, causal masking examples
- **Examples**: Single block behavior and attention evolution

### 05. Full Transformer Encoder (`05_full_transformer_encoder.py`)
- **Core concepts**: Stacking encoder blocks, different output strategies
- **Implementations**: 
  - Complete transformer encoder (BERT-style)
  - Classification heads
  - Sequence tagging models
- **Features**: Variable sequence handling, attention pooling, model analysis
- **Examples**: Text classification, NER, attention visualization

### 06. Transformer Decoder (`06_transformer_decoder.py`)
- **Core concepts**: Causal attention, autoregressive generation, cross-attention
- **Implementations**:
  - Causal multi-head attention
  - Decoder blocks (with/without cross-attention)
  - GPT-style language models
- **Features**: Text generation, different sampling strategies
- **Examples**: Encoder vs decoder attention, generation techniques

### 07. Complete Transformer (`07_complete_transformer.py`)
- **Core concepts**: Encoder-decoder architecture, sequence-to-sequence tasks
- **Implementations**:
  - Full transformer model
  - Translation model with utilities
  - Beam search and greedy decoding
- **Features**: Cross-attention visualization, BLEU scoring, model scaling
- **Examples**: Machine translation, attention analysis, evaluation metrics

### 08. Transformer Training (`08_transformer_training.py`)
- **Core concepts**: Training pipelines, optimization, evaluation
- **Implementations**:
  - Complete training framework
  - Data loading and tokenization
  - Learning rate scheduling (warmup + cosine decay)
  - Model checkpointing and evaluation
- **Features**: Training both translation and language models
- **Examples**: Full training loops, generation examples, best practices

## Learning Path

**Recommended order**: Follow lessons 01-08 sequentially. Each lesson builds on concepts from previous ones.

**Prerequisites**: 
- Basic PyTorch knowledge
- Understanding of neural networks and backpropagation
- Familiarity with NLP concepts (helpful but not required)

## Running the Lessons

Each lesson is self-contained and can be run independently:

```bash
# Run individual lessons
python lessons/01_attention_mechanism.py
python lessons/02_multi_head_attention.py
# ... and so on

# Or run all lessons
for lesson in lessons/*.py; do
    echo "Running $lesson"
    python "$lesson"
done
```

**Note**: Some lessons require matplotlib for visualizations. Install with `pip install matplotlib`.

## Key Concepts Covered

### Attention Mechanisms
- **Scaled dot-product attention**: Core attention computation
- **Multi-head attention**: Parallel attention in different subspaces  
- **Self-attention**: Attending within the same sequence
- **Cross-attention**: Attending between different sequences
- **Causal attention**: Preventing attention to future positions

### Architecture Components
- **Positional encoding**: Adding position information to embeddings
- **Layer normalization**: Stabilizing training with normalization
- **Residual connections**: Enabling deep network training
- **Feed-forward networks**: Position-wise transformations
- **Embeddings**: Converting tokens to dense representations

### Training and Optimization
- **Teacher forcing**: Training technique for sequence generation
- **Learning rate scheduling**: Warmup and decay strategies
- **Gradient clipping**: Preventing exploding gradients
- **Model checkpointing**: Saving and loading model states
- **Evaluation metrics**: Loss, perplexity, BLEU scores

### Generation Strategies
- **Greedy decoding**: Always selecting most probable token
- **Beam search**: Exploring multiple candidate sequences
- **Sampling**: Probabilistic token selection with temperature
- **Top-k sampling**: Sampling from top-k most probable tokens

## Architecture Variants

The lessons cover these transformer variants:

1. **Encoder-only** (BERT-style): For understanding and classification
2. **Decoder-only** (GPT-style): For text generation
3. **Encoder-decoder** (T5-style): For sequence-to-sequence tasks

## Advanced Topics

While these lessons cover the fundamentals, advanced topics for further exploration include:

- **Attention optimizations**: Flash attention, linear attention
- **Positional encodings**: RoPE, ALiBi, learned relative positions  
- **Architecture improvements**: RMSNorm, SwiGLU, parallel blocks
- **Scaling laws**: How performance scales with model/data size
- **Fine-tuning strategies**: LoRA, prefix tuning, prompt tuning

## Tips for Learning

1. **Run the code**: Don't just read - execute each lesson and experiment
2. **Modify parameters**: Change model sizes, sequence lengths, etc.
3. **Visualize attention**: Use the provided visualization functions
4. **Start small**: Begin with tiny models to understand the concepts
5. **Compare outputs**: See how different configurations affect results
6. **Read the comments**: Each implementation is heavily documented

## Next Steps

After completing these lessons, you'll be ready to:

- Implement transformer variants from research papers
- Train transformers on real datasets
- Fine-tune pre-trained models for specific tasks
- Optimize transformer training and inference
- Understand modern transformer architectures (GPT, BERT, T5, etc.)

Happy learning! 🚀