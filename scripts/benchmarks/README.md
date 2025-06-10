# Benchmark Scripts

This directory contains scripts for benchmarking and profiling the Ear Segmentation AI library.

## Available Scripts

### 1. run_benchmarks.py
Comprehensive performance benchmarks including:
- Single image processing speed
- Batch processing performance
- Different image sizes
- CPU vs GPU comparison
- Memory usage

**Usage:**
```bash
python scripts/benchmarks/run_benchmarks.py
```

**Output:** `benchmark_results.json`

### 2. compare_models.py
Compare different model architectures and configurations:
- Different encoder architectures (ResNet, MobileNet, EfficientNet)
- Different segmentation architectures (U-Net, DeepLabV3, PSPNet)
- Input size effects
- Model size vs performance trade-offs

**Usage:**
```bash
python scripts/benchmarks/compare_models.py
```

**Output:** `model_comparison_results.json`

### 3. profile_memory.py
Detailed memory profiling:
- Memory usage during initialization
- Memory per image processed
- Memory leak detection
- GPU memory tracking
- Line-by-line memory profiling

**Usage:**
```bash
python scripts/benchmarks/profile_memory.py
```

**Requirements:**
```bash
pip install memory-profiler psutil
```

## Quick Benchmark

Use the CLI for quick benchmarks:
```bash
# CPU benchmark
earsegmentationai benchmark examples/basic/0210.png --iterations 100

# GPU benchmark
earsegmentationai benchmark examples/basic/0210.png --device cuda:0 --iterations 100
```

## Performance Targets

Based on our benchmarks, expected performance:

| Device | Image Size | Target FPS | Memory |
|--------|------------|------------|---------|
| CPU (i7) | 480×320 | 10-20 | <300 MB |
| GPU (RTX 3080) | 480×320 | 100-150 | <500 MB |
| GPU (RTX 3080) | 1920×1080 | 30-50 | <1 GB |

## Optimization Tips

1. **Batch Processing**: Use batch sizes of 4-8 for optimal GPU utilization
2. **Frame Skipping**: For video, skip frames to achieve real-time performance
3. **Input Size**: Smaller input sizes (320×240) can double FPS with minimal accuracy loss
4. **Device Selection**: Use GPU for >10 FPS requirements

## Profiling Tools

Additional profiling with external tools:

```bash
# PyTorch profiler
python -m torch.utils.bottleneck scripts/benchmarks/run_benchmarks.py

# cProfile
python -m cProfile -o profile.stats scripts/benchmarks/run_benchmarks.py

# Memory profiler
mprof run scripts/benchmarks/profile_memory.py
mprof plot
```