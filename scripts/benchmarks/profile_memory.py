#!/usr/bin/env python3
"""
Profile memory usage of the Ear Segmentation AI models.

This script helps identify memory bottlenecks and optimize memory usage.
"""

import gc
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import psutil
import torch
from memory_profiler import profile

from earsegmentationai import ImageProcessor


def get_memory_info() -> Dict:
    """Get current memory usage information."""
    process = psutil.Process(os.getpid())
    
    info = {
        "cpu_memory_mb": process.memory_info().rss / 1024 / 1024,
        "cpu_memory_percent": process.memory_percent()
    }
    
    if torch.cuda.is_available():
        info.update({
            "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
            "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
            "gpu_memory_max_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024
        })
    
    return info


def profile_single_image_memory(image_path: Path, device: str = "cpu") -> Dict:
    """Profile memory usage for single image processing."""
    
    print(f"\n=== Profiling Single Image Memory Usage ({device}) ===")
    
    # Initial state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    initial_memory = get_memory_info()
    print(f"Initial CPU memory: {initial_memory['cpu_memory_mb']:.1f} MB")
    
    # Create processor
    processor = ImageProcessor(device=device)
    after_init = get_memory_info()
    print(f"After initialization: {after_init['cpu_memory_mb']:.1f} MB "
          f"(+{after_init['cpu_memory_mb'] - initial_memory['cpu_memory_mb']:.1f} MB)")
    
    # Process image
    result = processor.process(image_path)
    after_process = get_memory_info()
    print(f"After processing: {after_process['cpu_memory_mb']:.1f} MB "
          f"(+{after_process['cpu_memory_mb'] - after_init['cpu_memory_mb']:.1f} MB)")
    
    # Multiple processing (check for memory leaks)
    for i in range(10):
        result = processor.process(image_path)
    
    after_multiple = get_memory_info()
    print(f"After 10 iterations: {after_multiple['cpu_memory_mb']:.1f} MB "
          f"(+{after_multiple['cpu_memory_mb'] - after_process['cpu_memory_mb']:.1f} MB)")
    
    return {
        "initial": initial_memory,
        "after_init": after_init,
        "after_process": after_process,
        "after_multiple": after_multiple
    }


def profile_batch_memory(image_paths: List[Path], batch_sizes: List[int], device: str = "cpu") -> Dict:
    """Profile memory usage for different batch sizes."""
    
    print(f"\n=== Profiling Batch Memory Usage ({device}) ===")
    
    results = {}
    processor = ImageProcessor(device=device)
    
    for batch_size in batch_sizes:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        before = get_memory_info()
        
        # Process batch
        batch_paths = image_paths[:batch_size]
        result = processor.process(batch_paths)
        
        after = get_memory_info()
        
        memory_increase = after['cpu_memory_mb'] - before['cpu_memory_mb']
        
        results[f"batch_{batch_size}"] = {
            "cpu_memory_increase_mb": memory_increase,
            "memory_per_image_mb": memory_increase / batch_size
        }
        
        if torch.cuda.is_available() and device != "cpu":
            results[f"batch_{batch_size}"]["gpu_memory_mb"] = after.get('gpu_memory_allocated_mb', 0)
        
        print(f"Batch size {batch_size}: CPU +{memory_increase:.1f} MB "
              f"({memory_increase/batch_size:.1f} MB per image)")
    
    return results


@profile
def detailed_memory_profile(image_path: Path):
    """Detailed line-by-line memory profiling."""
    
    # Import here to profile imports
    import cv2
    from earsegmentationai import ImageProcessor
    from earsegmentationai.core.model import ModelManager
    
    # Create processor
    processor = ImageProcessor(device="cpu")
    
    # Load image
    image = cv2.imread(str(image_path))
    
    # Process
    result = processor.process(image)
    
    # Access different attributes
    mask = result.mask
    confidence = result.confidence
    
    # Create visualization
    if result.visualization is not None:
        viz = result.visualization
    
    # Cleanup
    del processor
    del result
    gc.collect()


def profile_model_loading():
    """Profile model loading and initialization."""
    
    print("\n=== Profiling Model Loading ===")
    
    from earsegmentationai.core.model import ModelManager
    
    # Clear any existing model
    ModelManager._instance = None
    ModelManager._model = None
    gc.collect()
    
    before = get_memory_info()
    
    # Create model manager
    manager = ModelManager()
    after_init = get_memory_info()
    
    # Get model (triggers loading)
    model = manager.get_model()
    after_load = get_memory_info()
    
    print(f"Before: {before['cpu_memory_mb']:.1f} MB")
    print(f"After init: {after_init['cpu_memory_mb']:.1f} MB "
          f"(+{after_init['cpu_memory_mb'] - before['cpu_memory_mb']:.1f} MB)")
    print(f"After load: {after_load['cpu_memory_mb']:.1f} MB "
          f"(+{after_load['cpu_memory_mb'] - after_init['cpu_memory_mb']:.1f} MB)")
    
    # Model size
    param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers()) / 1024 / 1024
    
    print(f"\nModel parameter size: {param_size:.1f} MB")
    print(f"Model buffer size: {buffer_size:.1f} MB")
    print(f"Total model size: {param_size + buffer_size:.1f} MB")


def main():
    """Run memory profiling suite."""
    
    # Setup
    test_image = Path("examples/basic/0210.png")
    if not test_image.exists():
        print(f"Test image not found: {test_image}")
        return
    
    # Profile model loading
    profile_model_loading()
    
    # Profile single image
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda:0")
    
    for device in devices:
        profile_single_image_memory(test_image, device)
    
    # Profile batch processing
    image_list = [test_image] * 16
    batch_sizes = [1, 2, 4, 8, 16]
    
    for device in devices:
        profile_batch_memory(image_list, batch_sizes, device)
    
    # Detailed profiling (CPU only)
    print("\n=== Detailed Memory Profile ===")
    print("Running line-by-line memory profiler...")
    detailed_memory_profile(test_image)
    
    print("\n=== Memory Profiling Complete ===")


if __name__ == "__main__":
    main()