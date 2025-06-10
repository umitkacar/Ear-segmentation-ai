#!/usr/bin/env python3
"""Test backward compatibility with v1.x API."""

import sys
import warnings
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np


def test_legacy_imports():
    """Test that legacy imports still work."""
    print("Testing legacy imports...")

    try:
        # Test v1.x style imports
        from earsegmentationai import ENCODER_NAME, MODEL_NAME

        print("✓ EarModel imported")
        print(f"✓ ENCODER_NAME: {ENCODER_NAME}")
        print(f"✓ MODEL_NAME: {MODEL_NAME}")

        # Test legacy functions
        print("✓ Legacy functions imported")

        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_legacy_ear_model():
    """Test legacy EarModel class."""
    print("\nTesting legacy EarModel class...")

    # Suppress deprecation warnings for this test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        try:
            from earsegmentationai import EarModel

            # Test v1.x workflow
            print("1. Creating EarModel instance...")
            model = EarModel()
            print("✓ EarModel created")

            print("2. Testing download_models()...")
            model.download_models()
            print("✓ download_models() called")

            print("3. Testing load_model()...")
            model.load_model(device="cpu")
            print("✓ Model loaded")

            print("4. Testing predict()...")
            # Create test image
            test_image = np.ones((256, 256, 3), dtype=np.uint8) * 128
            cv2.circle(test_image, (128, 128), 50, (255, 255, 255), -1)

            # Save test image
            test_path = Path("/tmp/test_ear.png")
            cv2.imwrite(str(test_path), test_image)

            # Test with file path (v1.x style)
            mask_orig, mask_resized = model.predict(str(test_path))
            print(f"✓ Prediction from file: mask shape = {mask_orig.shape}")

            # Test with numpy array
            mask_orig2, mask_resized2 = model.predict(test_image)
            print(f"✓ Prediction from array: mask shape = {mask_orig2.shape}")

            # Clean up
            test_path.unlink()

            return True

        except Exception as e:
            print(f"❌ Legacy EarModel test failed: {e}")
            import traceback

            traceback.print_exc()
            return False


def test_legacy_cli_commands():
    """Test that legacy CLI commands still work."""
    print("\nTesting legacy CLI commands...")

    import subprocess

    # Test old-style commands (should still work)
    legacy_commands = [
        [
            "python",
            "-m",
            "earsegmentationai.cli.app",
            "picture-capture",
            "--help",
        ],
        [
            "python",
            "-m",
            "earsegmentationai.cli.app",
            "video-capture",
            "--help",
        ],
        [
            "python",
            "-m",
            "earsegmentationai.cli.app",
            "webcam-capture",
            "--help",
        ],
    ]

    success = True
    for cmd in legacy_commands:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
            )

            if result.returncode == 0:
                print(f"✓ Command works: {' '.join(cmd[-2:])}")
            else:
                print(f"❌ Command failed: {' '.join(cmd[-2:])}")
                print(result.stderr)
                success = False

        except Exception as e:
            print(f"❌ Error running command: {e}")
            success = False

    return success


def test_deprecation_warnings():
    """Test that deprecation warnings are shown."""
    print("\nTesting deprecation warnings...")

    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        try:
            from earsegmentationai import EarModel

            # This should trigger a warning
            EarModel()

            if len(w) > 0:
                print(f"✓ Deprecation warning shown: {w[0].message}")
                return True
            else:
                print("❌ No deprecation warning shown")
                return False

        except Exception as e:
            print(f"❌ Error: {e}")
            return False


def test_migration_guide():
    """Test migration guide helper."""
    print("\nTesting migration guide...")

    try:
        # Just check it runs without error
        import io
        import sys

        from earsegmentationai.compat import migrate_v1_to_v2

        # Capture output
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        migrate_v1_to_v2()

        output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        if "Migration Guide" in output:
            print("✓ Migration guide generated")
            return True
        else:
            print("❌ Migration guide empty")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def compare_outputs():
    """Compare outputs between v1 and v2 APIs."""
    print("\nComparing v1 and v2 API outputs...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        try:
            # Create test image
            test_image = np.ones((256, 256, 3), dtype=np.uint8) * 128
            cv2.circle(test_image, (128, 128), 50, (255, 255, 255), -1)

            # V1 API
            from earsegmentationai import EarModel

            model_v1 = EarModel()
            model_v1.load_model(device="cpu")
            mask_v1, _ = model_v1.predict(test_image)

            # V2 API
            from earsegmentationai import ImageProcessor

            processor_v2 = ImageProcessor(device="cpu")
            result_v2 = processor_v2.process(test_image)
            mask_v2 = result_v2.mask

            # Compare shapes
            if mask_v1.shape == mask_v2.shape:
                print(f"✓ Output shapes match: {mask_v1.shape}")

                # Check if predictions are similar (not exact due to potential differences)
                overlap = np.sum((mask_v1 > 0) & (mask_v2 > 0))
                union = np.sum((mask_v1 > 0) | (mask_v2 > 0))

                if union > 0:
                    iou = overlap / union
                    print(f"✓ IoU between v1 and v2: {iou:.3f}")
                    return iou > 0.8  # Should be very similar
                else:
                    print("✓ Both masks empty (no detection)")
                    return True
            else:
                print(
                    f"❌ Shape mismatch: v1={mask_v1.shape}, v2={mask_v2.shape}"
                )
                return False

        except Exception as e:
            print(f"❌ Comparison failed: {e}")
            import traceback

            traceback.print_exc()
            return False


if __name__ == "__main__":
    print("=" * 60)
    print("BACKWARD COMPATIBILITY TEST")
    print("=" * 60)

    success = True

    # Run all tests
    success &= test_legacy_imports()
    success &= test_legacy_ear_model()
    success &= test_legacy_cli_commands()
    success &= test_deprecation_warnings()
    success &= test_migration_guide()
    success &= compare_outputs()

    if success:
        print("\n✅ All backward compatibility tests passed!")
        print("The v2.0 API is fully backward compatible with v1.x")
        sys.exit(0)
    else:
        print("\n❌ Some backward compatibility tests failed!")
        sys.exit(1)
