# Release v2.0.0

## 🎉 Major Release - Complete Architecture Overhaul

This is a complete rewrite of Ear Segmentation AI with modern architecture, better performance, and enhanced features while maintaining backward compatibility.

## What's Changed

### 🏗️ Architecture & Structure
* **refactor**: Complete project restructure with `src/` layout by @umitkacar in #160
* **refactor**: Modular architecture with clear separation of concerns by @umitkacar in #160
* **refactor**: Migrated from monolithic to plugin-based design by @umitkacar in #160

### ✨ New Features
* **feat**: New `ImageProcessor` API with batch processing support by @umitkacar in #160
* **feat**: New `VideoProcessor` API with temporal smoothing by @umitkacar in #160
* **feat**: Modern CLI using Typer with rich output by @umitkacar in #160
* **feat**: Pydantic v2 configuration management by @umitkacar in #160
* **feat**: Benchmark command for performance testing by @umitkacar in #160
* **feat**: URL processing support for images by @umitkacar in #160
* **feat**: Frame skipping for video performance optimization by @umitkacar in #160

### 🔧 Improvements
* **perf**: 2-3x faster batch processing with optimized pipeline by @umitkacar in #160
* **perf**: Reduced memory usage with singleton model management by @umitkacar in #160
* **fix**: Fixed image mode bug - now supports both files and directories by @umitkacar in #160
* **fix**: Fixed memory leaks in video processing by @umitkacar in #160
* **fix**: Improved error messages and validation by @umitkacar in #160

### 📚 Documentation
* **docs**: Complete API documentation in `docs/api/` by @umitkacar in #160
* **docs**: Migration guide from v1.x to v2.0 by @umitkacar in #160
* **docs**: Architecture documentation by @umitkacar in #160
* **docs**: Installation and quickstart guides by @umitkacar in #160
* **docs**: Reorganized all documentation into logical structure by @umitkacar in #160

### 🧪 Testing & Quality
* **test**: Comprehensive test suite with 100% passing rate (134 tests) by @umitkacar in #160
* **test**: Unit and integration tests with pytest by @umitkacar in #160
* **ci**: Updated GitHub Actions workflows by @umitkacar in #160
* **style**: Applied black formatting to entire codebase by @umitkacar in #160

### 🔄 Backward Compatibility
* **compat**: Full v1.x API compatibility with deprecation warnings by @umitkacar in #160
* **compat**: Legacy CLI commands still supported by @umitkacar in #160

### 🛠️ Development
* **build**: Poetry for dependency management by @umitkacar in #160
* **build**: Makefile for common development tasks by @umitkacar in #160
* **build**: Pre-commit hooks for code quality by @umitkacar in #160

## Breaking Changes

While we maintain backward compatibility, the internal structure has completely changed:
- Source code moved from `earsegmentationai/` to `src/earsegmentationai/`
- Configuration now uses Pydantic v2
- Some internal APIs have changed (see migration guide)

## Migration Guide

See [Migration Guide](https://github.com/umitkacar/Ear-segmentation-ai/blob/main/docs/migration/MIGRATION.md) for detailed instructions on upgrading from v1.x.

## Installation

```bash
pip install --upgrade earsegmentationai==2.0.0
```

## Quick Example

```python
from earsegmentationai import ImageProcessor

# Process an image
processor = ImageProcessor(device="cuda:0")
result = processor.process("image.jpg")
print(f"Ear detected: {result.num_ears > 0}")
print(f"Confidence: {result.confidence:.2f}")
```

## Contributors

Thanks to @umitkacar for this major refactor!

**Full Changelog**: https://github.com/umitkacar/Ear-segmentation-ai/compare/v1.0.2...v2.0.0