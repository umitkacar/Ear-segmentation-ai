#!/usr/bin/env python3
"""
Compare performance of different model configurations.

This script benchmarks different model architectures and configurations
to help choose the best model for specific use cases.
"""

import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from earsegmentationai.core.config import Config, ModelConfig
from earsegmentationai.core.model import ModelManager
from earsegmentationai.core.predictor import EarPredictor


def benchmark_model_config(
    config: ModelConfig,
    test_image_path: Path,
    iterations: int = 50,
    device: str = "cpu",
) -> Dict:
    """Benchmark a specific model configuration."""

    # Create model with config
    model_manager = ModelManager()
    model_manager.config = Config(model=config)
    model = model_manager.get_model()

    # Create predictor
    predictor = EarPredictor(threshold=0.5)

    # Load test image
    import cv2

    image = cv2.imread(str(test_image_path))

    # Warmup
    for _ in range(10):
        predictor.predict(image)

    # Benchmark
    times = []
    for _ in tqdm(
        range(iterations), desc=f"{config.architecture}-{config.encoder}"
    ):
        start = time.time()
        mask, prob = predictor.predict(image, return_probability=True)
        end = time.time()
        times.append(end - start)

    times = np.array(times)

    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    return {
        "architecture": config.architecture,
        "encoder": config.encoder,
        "mean_time": float(np.mean(times)),
        "std_time": float(np.std(times)),
        "fps": float(1.0 / np.mean(times)),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": total_params * 4 / 1024 / 1024,  # Assuming float32
    }


def compare_architectures(test_image_path: Path, device: str = "cpu") -> Dict:
    """Compare different model architectures."""

    architectures = [
        ("Unet", "resnet18"),
        ("Unet", "resnet34"),
        ("Unet", "mobilenet_v2"),
        ("Unet", "efficientnet-b0"),
        ("UnetPlusPlus", "resnet18"),
        ("FPN", "resnet18"),
        ("PSPNet", "resnet18"),
        ("DeepLabV3", "resnet18"),
        ("DeepLabV3Plus", "resnet18"),
    ]

    results = []

    for arch, encoder in architectures:
        try:
            config = ModelConfig(
                architecture=arch,
                encoder=encoder,
                encoder_weights="imagenet",
                in_channels=3,
                classes=1,
            )

            result = benchmark_model_config(
                config, test_image_path, iterations=30, device=device
            )

            results.append(result)

        except Exception as e:
            print(f"Failed to benchmark {arch}-{encoder}: {e}")
            continue

    return results


def compare_input_sizes(test_image_path: Path, device: str = "cpu") -> Dict:
    """Compare different input sizes."""

    input_sizes = [(256, 256), (320, 320), (480, 320), (512, 512), (640, 480)]

    results = []

    for size in input_sizes:
        config = Config()
        config.processing.input_size = size

        # Create predictor
        predictor = EarPredictor(threshold=0.5)

        # Load and resize image
        import cv2

        image = cv2.imread(str(test_image_path))

        # Benchmark
        times = []
        for _ in tqdm(range(50), desc=f"Size {size[0]}x{size[1]}"):
            start = time.time()
            predictor.predict(image)
            end = time.time()
            times.append(end - start)

        times = np.array(times)

        results.append(
            {
                "input_size": f"{size[0]}x{size[1]}",
                "mean_time": float(np.mean(times)),
                "fps": float(1.0 / np.mean(times)),
                "pixels": size[0] * size[1],
            }
        )

    return results


def compare_quantization(test_image_path: Path) -> Dict:
    """Compare model quantization effects."""

    # This would require implementing model quantization
    # Placeholder for future implementation

    return {
        "float32": {"size_mb": 55, "fps": 15},
        "int8": {"size_mb": 14, "fps": 45},
        "note": "Quantization comparison not yet implemented",
    }


def main():
    """Run model comparison benchmarks."""

    # Setup
    test_image = Path("examples/basic/0210.png")
    if not test_image.exists():
        print(f"Test image not found: {test_image}")
        return

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
    }

    # Compare architectures
    print("\n=== Comparing Model Architectures ===")
    results["architectures"] = compare_architectures(test_image, device)

    # Compare input sizes
    print("\n=== Comparing Input Sizes ===")
    results["input_sizes"] = compare_input_sizes(test_image, device)

    # Save results
    output_path = Path("model_comparison_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Print summary
    print("\n=== Architecture Comparison Summary ===")
    if "architectures" in results:
        architectures = sorted(
            results["architectures"], key=lambda x: x["fps"], reverse=True
        )
        print(
            f"{'Model':<30} {'FPS':<10} {'Params (M)':<12} {'Size (MB)':<10}"
        )
        print("-" * 65)
        for arch in architectures[:5]:  # Top 5
            model_name = f"{arch['architecture']}-{arch['encoder']}"
            params_m = arch["total_params"] / 1e6
            print(
                f"{model_name:<30} {arch['fps']:<10.1f} {params_m:<12.1f} {arch['model_size_mb']:<10.1f}"
            )

    print("\n=== Input Size Comparison ===")
    if "input_sizes" in results:
        print(f"{'Size':<15} {'FPS':<10} {'Pixels':<10}")
        print("-" * 35)
        for size_result in results["input_sizes"]:
            print(
                f"{size_result['input_size']:<15} {size_result['fps']:<10.1f} {size_result['pixels']:<10}"
            )


if __name__ == "__main__":
    main()
