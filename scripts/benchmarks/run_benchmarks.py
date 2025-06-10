#!/usr/bin/env python3
"""
Comprehensive benchmark suite for Ear Segmentation AI.

This script runs various benchmarks to measure performance across different:
- Devices (CPU vs GPU)
- Image sizes
- Batch sizes
- Model architectures
"""

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from earsegmentationai import ImageProcessor
from earsegmentationai.core.model import ModelManager


def benchmark_single_image(
    processor: ImageProcessor,
    image_path: Path,
    iterations: int = 100,
    warmup: int = 10
) -> Dict:
    """Benchmark single image processing."""
    
    # Warmup
    for _ in range(warmup):
        processor.process(image_path)
    
    # Benchmark
    times = []
    for _ in tqdm(range(iterations), desc="Single image"):
        start = time.time()
        result = processor.process(image_path)
        end = time.time()
        times.append(end - start)
    
    times = np.array(times)
    return {
        "mean_time": float(np.mean(times)),
        "std_time": float(np.std(times)),
        "min_time": float(np.min(times)),
        "max_time": float(np.max(times)),
        "fps": float(1.0 / np.mean(times))
    }


def benchmark_batch_processing(
    processor: ImageProcessor,
    image_paths: List[Path],
    batch_sizes: List[int],
    iterations: int = 10
) -> Dict:
    """Benchmark batch processing with different batch sizes."""
    
    results = {}
    for batch_size in batch_sizes:
        processor.batch_size = batch_size
        
        times = []
        for _ in tqdm(range(iterations), desc=f"Batch size {batch_size}"):
            start = time.time()
            batch_result = processor.process(image_paths[:batch_size])
            end = time.time()
            times.append(end - start)
        
        times = np.array(times)
        results[f"batch_{batch_size}"] = {
            "mean_time": float(np.mean(times)),
            "images_per_second": float(batch_size / np.mean(times)),
            "speedup": float(batch_size / np.mean(times)) / results.get("batch_1", {}).get("images_per_second", 1.0) if batch_size > 1 else 1.0
        }
    
    return results


def benchmark_image_sizes(
    image_path: Path,
    sizes: List[tuple],
    device: str = "cpu"
) -> Dict:
    """Benchmark different image sizes."""
    
    import cv2
    from earsegmentationai import ImageProcessor
    
    processor = ImageProcessor(device=device)
    results = {}
    
    # Load original image
    original = cv2.imread(str(image_path))
    
    for size in sizes:
        # Resize image
        resized = cv2.resize(original, size)
        
        # Benchmark
        times = []
        for _ in tqdm(range(50), desc=f"Size {size[0]}x{size[1]}"):
            start = time.time()
            result = processor.process(resized)
            end = time.time()
            times.append(end - start)
        
        times = np.array(times)
        results[f"{size[0]}x{size[1]}"] = {
            "mean_time": float(np.mean(times)),
            "fps": float(1.0 / np.mean(times)),
            "pixels_per_second": float(size[0] * size[1] / np.mean(times))
        }
    
    return results


def benchmark_memory_usage(processor: ImageProcessor, image_path: Path) -> Dict:
    """Benchmark memory usage."""
    
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    # Initial memory
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Process image
    result = processor.process(image_path)
    
    # Peak memory
    peak_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # GPU memory if available
    gpu_memory = None
    if torch.cuda.is_available() and processor.device != "cpu":
        gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        torch.cuda.reset_peak_memory_stats()
    
    return {
        "initial_memory_mb": initial_memory,
        "peak_memory_mb": peak_memory,
        "memory_increase_mb": peak_memory - initial_memory,
        "gpu_memory_mb": gpu_memory
    }


def main():
    """Run comprehensive benchmarks."""
    
    # Setup
    test_image = Path("examples/basic/0210.png")
    if not test_image.exists():
        print(f"Test image not found: {test_image}")
        return
    
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": {
            "python_version": torch.__version__,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None
        }
    }
    
    # Test devices
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda:0")
    
    for device in devices:
        print(f"\n=== Benchmarking on {device} ===")
        processor = ImageProcessor(device=device)
        
        # Single image benchmark
        print("\n1. Single Image Processing")
        results[f"{device}_single"] = benchmark_single_image(
            processor, test_image, iterations=100, warmup=10
        )
        
        # Batch processing benchmark
        print("\n2. Batch Processing")
        image_list = [test_image] * 16  # Simulate batch
        results[f"{device}_batch"] = benchmark_batch_processing(
            processor, image_list, batch_sizes=[1, 2, 4, 8, 16], iterations=10
        )
        
        # Memory usage
        print("\n3. Memory Usage")
        results[f"{device}_memory"] = benchmark_memory_usage(processor, test_image)
    
    # Image size benchmarks (CPU only for consistency)
    print("\n4. Image Size Benchmarks")
    sizes = [(320, 240), (640, 480), (1280, 720), (1920, 1080)]
    results["image_sizes"] = benchmark_image_sizes(test_image, sizes, device="cpu")
    
    # Save results
    output_path = Path("benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Print summary
    print("\n=== Summary ===")
    for device in devices:
        if f"{device}_single" in results:
            single = results[f"{device}_single"]
            print(f"\n{device.upper()}:")
            print(f"  Mean time: {single['mean_time']*1000:.2f} ms")
            print(f"  FPS: {single['fps']:.1f}")
            if f"{device}_memory" in results:
                mem = results[f"{device}_memory"]
                print(f"  Memory increase: {mem['memory_increase_mb']:.1f} MB")
                if mem['gpu_memory_mb']:
                    print(f"  GPU memory: {mem['gpu_memory_mb']:.1f} MB")


if __name__ == "__main__":
    main()