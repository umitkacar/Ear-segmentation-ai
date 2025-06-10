# Installation Guide

## Requirements

- Python 3.8, 3.9, or 3.10
- pip or Poetry package manager
- (Optional) CUDA-capable GPU for faster processing

## Installation Methods

### Method 1: Using pip (Recommended for users)

```bash
pip install earsegmentationai
```

### Method 2: Using Poetry (Recommended for developers)

```bash
# Clone the repository
git clone https://github.com/umitkacar/Ear-segmentation-ai.git
cd Ear-segmentation-ai

# Install using Poetry
poetry install
```

### Method 3: From source

```bash
# Clone the repository
git clone https://github.com/umitkacar/Ear-segmentation-ai.git
cd Ear-segmentation-ai

# Install in development mode
pip install -e .
```

## Verify Installation

```bash
# Check version
earsegmentationai --version

# Run help
earsegmentationai --help
```

## GPU Support

If you have a CUDA-capable GPU, install PyTorch with CUDA support:

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Model Download

The ear segmentation model (approximately 55MB) will be automatically downloaded on first use to:
```
~/.cache/earsegmentationai/models/
```

## Troubleshooting

### Import Error
If you get import errors, ensure the package is installed:
```bash
pip show earsegmentationai
```

### CUDA Not Available
Check PyTorch CUDA availability:
```python
import torch
print(torch.cuda.is_available())
```

### Permission Errors
On Linux/Mac, you may need to use `sudo` or create a virtual environment:
```bash
python -m venv ear-env
source ear-env/bin/activate  # On Windows: ear-env\Scripts\activate
pip install earsegmentationai
```
