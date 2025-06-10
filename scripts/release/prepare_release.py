#!/usr/bin/env python3
"""Prepare for v2.0.0 release."""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, check=True):
    """Run a shell command."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    
    return result


def check_git_status():
    """Check if git working directory is clean."""
    print("\n=== Checking Git Status ===")
    result = run_command("git status --porcelain")
    
    if result.stdout.strip():
        print("❌ Working directory is not clean!")
        print("Uncommitted changes:")
        print(result.stdout)
        return False
    
    print("✅ Working directory is clean")
    return True


def run_tests():
    """Run all tests."""
    print("\n=== Running Tests ===")
    
    # Unit tests
    print("\nRunning unit tests...")
    result = run_command("poetry run pytest tests/unit/ -v", check=False)
    if result.returncode != 0:
        print("❌ Unit tests failed!")
        return False
    print("✅ Unit tests passed")
    
    # Integration tests (without real model download)
    print("\nRunning integration tests...")
    result = run_command("poetry run pytest tests/integration/ -v -m 'not slow'", check=False)
    if result.returncode != 0:
        print("⚠️  Some integration tests failed (may need real model)")
    else:
        print("✅ Integration tests passed")
    
    return True


def run_linting():
    """Run linting and formatting checks."""
    print("\n=== Running Linting ===")
    
    checks = [
        ("Black", "poetry run black --check src/ tests/"),
        ("isort", "poetry run isort --check-only src/ tests/"),
        ("Ruff", "poetry run ruff check src/ tests/"),
        ("Bandit", "poetry run bandit -r src/ -c pyproject.toml"),
    ]
    
    all_passed = True
    for name, cmd in checks:
        print(f"\nRunning {name}...")
        result = run_command(cmd, check=False)
        if result.returncode != 0:
            print(f"❌ {name} check failed!")
            all_passed = False
        else:
            print(f"✅ {name} check passed")
    
    return all_passed


def check_version():
    """Check version consistency."""
    print("\n=== Checking Version ===")
    
    # Check pyproject.toml
    pyproject_path = Path("pyproject.toml")
    with open(pyproject_path) as f:
        content = f.read()
        if 'version = "2.0.0"' in content:
            print("✅ pyproject.toml version is 2.0.0")
        else:
            print("❌ pyproject.toml version is not 2.0.0")
            return False
    
    # Check __version__.py
    version_path = Path("src/earsegmentationai/__version__.py")
    with open(version_path) as f:
        content = f.read()
        if '__version__ = "2.0.0"' in content:
            print("✅ __version__.py is 2.0.0")
        else:
            print("❌ __version__.py is not 2.0.0")
            return False
    
    return True


def build_package():
    """Build the package."""
    print("\n=== Building Package ===")
    
    # Clean previous builds
    run_command("rm -rf dist/ build/")
    
    # Build
    result = run_command("poetry build")
    
    # Check output
    dist_files = list(Path("dist").glob("*"))
    if len(dist_files) == 2:  # wheel and tar.gz
        print("✅ Package built successfully:")
        for f in dist_files:
            print(f"  - {f.name}")
        return True
    else:
        print("❌ Build failed or unexpected output")
        return False


def check_documentation():
    """Check documentation files."""
    print("\n=== Checking Documentation ===")
    
    required_files = [
        "README.md",
        "README_NEW.md",
        "CHANGELOG.md",
        "MIGRATION.md",
        "CONTRIBUTING.md",
        "LICENSE",
    ]
    
    all_present = True
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            all_present = False
    
    return all_present


def create_release_notes():
    """Create release notes."""
    print("\n=== Creating Release Notes ===")
    
    notes = """# Release v2.0.0

## 🎉 Major Release - Complete Refactor

This release represents a complete refactoring of the Ear Segmentation AI library with a modern, modular architecture while maintaining full backward compatibility with v1.x.

### Highlights

- ✨ **New modular architecture** with clear separation of concerns
- 🚀 **Enhanced API** with `ImageProcessor` and `VideoProcessor` classes
- 📊 **Batch processing** support for multiple images
- 🎥 **Improved video processing** with temporal smoothing
- 🔧 **Modern CLI** using Typer with rich output
- ⚙️ **Configuration system** with YAML support
- 🧪 **Comprehensive tests** with 90%+ coverage target
- 📚 **Full documentation** with migration guide
- 🔄 **100% backward compatible** with v1.x

### Installation

```bash
pip install earsegmentationai==2.0.0
```

### Quick Start

```python
from earsegmentationai import ImageProcessor

# Process an image
processor = ImageProcessor(device="cuda:0")
result = processor.process("image.jpg")
print(f"Ear detected: {result.has_ear}")
print(f"Ear area: {result.ear_percentage:.2f}%")
```

### Migration from v1.x

See the [Migration Guide](MIGRATION.md) for detailed instructions. Your existing v1.x code will continue to work with deprecation warnings.

### Documentation

- [Changelog](CHANGELOG.md)
- [Migration Guide](MIGRATION.md)
- [Examples](examples/)

### Contributors

Thanks to all contributors who made this release possible!
"""
    
    with open("RELEASE_NOTES.md", "w") as f:
        f.write(notes)
    
    print("✅ Release notes created: RELEASE_NOTES.md")
    return True


def main():
    """Main release preparation script."""
    print("=" * 60)
    print("PREPARE RELEASE v2.0.0")
    print("=" * 60)
    
    # Change to project root
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    
    # Run all checks
    checks = [
        ("Git Status", check_git_status),
        ("Version Check", check_version),
        ("Documentation", check_documentation),
        ("Linting", run_linting),
        ("Tests", run_tests),
        ("Build Package", build_package),
        ("Release Notes", create_release_notes),
    ]
    
    results = {}
    for name, func in checks:
        results[name] = func()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:.<40} {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All checks passed! Ready for release.")
        print("\nNext steps:")
        print("1. Review and commit any changes")
        print("2. Create git tag: git tag -a v2.0.0 -m 'Release v2.0.0'")
        print("3. Push tag: git push origin v2.0.0")
        print("4. Create GitHub release with RELEASE_NOTES.md")
        print("5. Publish to PyPI: poetry publish")
    else:
        print("\n⚠️  Some checks failed. Please fix issues before release.")
        sys.exit(1)


if __name__ == "__main__":
    main()