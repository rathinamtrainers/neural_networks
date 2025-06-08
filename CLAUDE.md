# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a PyTorch-based educational repository demonstrating neural network concepts through a progressive series of numbered Python scripts (010-130). Each file builds upon previous concepts, starting from basic tensor operations and advancing to deep neural networks.

## Key Dependencies

- PyTorch (with optional CUDA support for GPU acceleration)
- matplotlib (for visualization)
- scikit-learn (for utilities)
- pandas (for data handling)
- python-dotenv (for environment variables)

## Code Structure

The codebase follows a numbered progression system:
- **010-020**: PyTorch fundamentals (tensors, basic operations)
- **030**: GPU acceleration demonstrations
- **040**: Advanced tensor operations
- **050-054**: Autograd (automatic differentiation) - from basics to gradient descent
- **060**: First neural network implementation
- **070-080**: Forward/backward propagation, custom modules
- **090**: Linear regression using PyTorch
- **100**: Training and evaluation patterns
- **110-113**: Model persistence (saving/loading patterns)
- **120-122**: Data loading (built-in datasets, custom datasets)
- **130**: Deep neural network implementation

## Development Commands

No formal build, test, or lint commands are configured. To run any script:
```bash
python src/<filename>.py
```

For Jupyter notebooks:
```bash
jupyter notebook src/<filename>.ipynb
```

## Architecture Notes

1. Each numbered file is self-contained and demonstrates specific concepts
2. The codebase progresses from low-level tensor operations to high-level neural network abstractions
3. Model checkpoints are saved as `.pth` files in the root directory
4. Custom datasets use CSV format (see `122-sample_data.csv`)
5. GPU support is demonstrated but not required - code typically includes CPU fallbacks

## Important Patterns

- When demonstrating GPU usage, always check for CUDA availability first
- Model saving/loading examples show multiple approaches (full model, weights only, with optimizer state)
- Custom datasets inherit from `torch.utils.data.Dataset`
- Training loops typically follow the pattern: forward pass → loss calculation → backward pass → optimizer step