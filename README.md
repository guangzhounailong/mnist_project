# MNIST Handwritten Digit Recognition

A beginner-friendly PyTorch project for learning the complete workflow from
data loading to training, evaluation, and inference.

## Environment setup (macOS / Apple Silicon)

This machine's system Python is newer than the version recommended by PyTorch,
so the project uses Python 3.12 in an isolated virtual environment.

```bash
cd /Users/kenny/study/mnist_project
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
python -m pip install torch torchvision matplotlib pillow
```

Verify the installation:

```bash
python -c "import torch, torchvision, matplotlib, PIL; print('PyTorch:', torch.__version__); print('torchvision:', torchvision.__version__); print('MPS available:', torch.backends.mps.is_available())"
```
