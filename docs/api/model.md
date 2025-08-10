# Model Management API

## ModelManager
The `ModelManager` handles model lifecycle tasks such as downloading weights, caching them locally and validating their integrity before use.

### Download and Cache
Models are stored under the path defined by configuration. `load_model()` checks the cache and downloads weights if missing or when `force_download=True`.

```python
from earsegmentationai.core.model import ModelManager

manager = ModelManager()
model = manager.model  # downloads and caches on first access

# Force re-download
manager.load_model(force_download=True)
```

### Validation
Each download can be verified against an expected SHA256 hash. If the hash mismatches the file is rejected.

```python
model_path = manager.config.model_path
manager._verify_model(model_path, manager.config.model.expected_hash)
```

### Device Management
`ModelManager` keeps a single instance of the model and moves it between devices on demand.

```python
manager.set_device("cuda:0")
info = manager.get_model_info()
```

### Clearing Cache
GPU memory can be released using:

```python
manager.clear_cache()
```

