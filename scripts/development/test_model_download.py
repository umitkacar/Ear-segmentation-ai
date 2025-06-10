#!/usr/bin/env python3
"""Test model download and loading functionality."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from earsegmentationai.core.config import Config, set_config
from earsegmentationai.core.model import ModelManager
from earsegmentationai.api.image import ImageProcessor
import numpy as np


def test_model_download():
    """Test model download functionality."""
    print("Testing model download and loading...")
    
    # Create test config with temp directory
    test_dir = Path.home() / ".cache" / "earsegmentationai_test"
    config = Config(
        cache_dir=test_dir,
        processing={"device": "cpu"}  # Use CPU for testing
    )
    set_config(config)
    
    try:
        # Test 1: Model Manager initialization
        print("\n1. Testing ModelManager initialization...")
        manager = ModelManager()
        print("✓ ModelManager created")
        
        # Test 2: Model download
        print("\n2. Testing model download...")
        model_path = config.model_path
        if model_path.exists():
            print(f"Model already exists at: {model_path}")
            print("Deleting for fresh download test...")
            model_path.unlink()
        
        model = manager.load_model()
        print("✓ Model downloaded and loaded successfully")
        
        # Test 3: Model info
        print("\n3. Testing model info...")
        info = manager.get_model_info()
        print(f"Architecture: {info['architecture']}")
        print(f"Encoder: {info['encoder']}")
        print(f"Device: {info['device']}")
        print(f"Parameters: {info.get('parameters', 'N/A'):,}")
        print(f"Memory (MB): {info.get('memory_mb', 'N/A'):.2f}")
        
        # Test 4: Test prediction
        print("\n4. Testing prediction...")
        processor = ImageProcessor(model_manager=manager)
        
        # Create test image
        test_image = np.ones((256, 256, 3), dtype=np.uint8) * 128
        result = processor.process(test_image)
        
        print(f"✓ Prediction successful")
        print(f"Ear detected: {result.has_ear}")
        print(f"Mask shape: {result.mask.shape}")
        
        # Test 5: Model reload
        print("\n5. Testing model reload...")
        model2 = manager.reload_model()
        print("✓ Model reloaded successfully")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        if test_dir.exists():
            print(f"\nCleaning up test directory: {test_dir}")
            import shutil
            # Keep the model for future tests
            # shutil.rmtree(test_dir)


def test_cli_download():
    """Test CLI download command."""
    print("\n\nTesting CLI download command...")
    
    import subprocess
    
    try:
        # Run download command
        result = subprocess.run(
            ["python", "-m", "earsegmentationai.cli.app", "download-model"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        if result.returncode == 0:
            print("✓ CLI download command successful")
            print(result.stdout)
        else:
            print("❌ CLI download command failed")
            print(result.stderr)
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False


def test_different_devices():
    """Test model on different devices."""
    print("\n\nTesting different devices...")
    
    import torch
    
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda:0")
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, testing CPU only")
    
    for device in devices:
        print(f"\nTesting on {device}...")
        
        try:
            processor = ImageProcessor(device=device)
            test_image = np.ones((256, 256, 3), dtype=np.uint8) * 128
            
            # Warmup
            processor.warmup()
            
            # Time prediction
            import time
            start = time.time()
            result = processor.process(test_image)
            elapsed = time.time() - start
            
            print(f"✓ {device}: {elapsed*1000:.2f} ms")
            
        except Exception as e:
            print(f"❌ {device} failed: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("EAR SEGMENTATION AI - MODEL DOWNLOAD TEST")
    print("=" * 60)
    
    success = True
    
    # Run tests
    success &= test_model_download()
    success &= test_cli_download()
    success &= test_different_devices()
    
    if success:
        print("\n🎉 All model tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)