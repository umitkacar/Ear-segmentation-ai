# Efficient and Lightweight Ear Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/umitkacar/Ear-segmentation-ai/main.svg)](https://results.pre-commit.ci/latest/github/umitkacar/Ear-segmentation-ai/main)
![PyPI](https://img.shields.io/pypi/v/earsegmentationai)
![PyPI - Downloads](https://img.shields.io/pypi/dm/earsegmentationai?color=red)
![PyPI - Format](https://img.shields.io/pypi/format/earsegmentationai)
![PyPI - Status](https://img.shields.io/pypi/status/earsegmentationai?color=orange)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/earsegmentationai)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/earsegmentationai)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff)

<p>
  <img alt="Python38" src="https://img.shields.io/badge/Python-3.8-3776AB.svg?logo=Python&logoColor=white"/>
  <img alt="Python39" src="https://img.shields.io/badge/Python-3.9-3776AB.svg?logo=Python&logoColor=white"/>
  <img alt="Python310" src="https://img.shields.io/badge/Python-3.10-3776AB.svg?logo=Python&logoColor=white"/>
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-v1.13.1-EE4C2C.svg?logo=PyTorch&logoColor=white"/>
  <img alt="Torchvision" src="https://img.shields.io/badge/Torchvision-v0.14.1-EE4C2C.svg?logo=PyTorch&logoColor=white"/>
  <img alt="Torchvision" src="https://img.shields.io/badge/segmentationModelsPytorch-v0.3.2-EE4C2C.svg?logo=PyTorch&logoColor=white"/>
  <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-4.7.0-5C3EE8?logo=OpenCV&logoColor=white"/>
  <img alt="Cuda" src="https://img.shields.io/badge/Cuda-Enabled-76B900.svg?logo=Nvidia&logoColor=white"/>
  <img alt="Poetry" src="https://img.shields.io/badge/Poetry-60A5FA.svg?logo=Poetry&logoColor=white"/>
  <img alt="Black" src="https://img.shields.io/badge/code%20style-black-black"/>
  <img alt="Mypy" src="https://img.shields.io/badge/mypy-checked-blue"/>
  <img alt="isort" src="https://img.shields.io/badge/isort-checked-yellow"/>
</p>

## Download Model 📂

<p>
<a href="https://drive.google.com/drive/folders/1l88PrrNESBDZ4Jd3QJSG9EbIe0CjXC_j?usp=sharing"><img alt="GoogleDrive" src="https://img.shields.io/badge/GoogleDrive-4285F4?logo=GoogleDrive&logoColor=white"></a>
<a href="https://github.com/umitkacar/Ear-segmentation-ai/releases/download/v1.0.0/earsegmentation_model_v1_46.pth"><img alt="Github" src="https://img.shields.io/badge/Github Download-181717?logo=Github&logoColor=white"></a>
</p>

## ⚙️ Requirements ⚙️

* Python 3.8 to Python3.10 (Virtualenv recommended)
* Display Server for showing results
* Optional: poetry
* Optional: Nvidia CUDA for cuda usage

## 🛠️ Installation 🛠️

### Pip installation

```bash
pip install -r requirements.txt
```

### Poetry installation

```bash
poetry shell
poetry install
```

## Optional (If you have multiple python installation)

```bash
poetry env use $(which python3.10)
poetry shell
poetry install
```

## Usage

```bash
earsegmentationai --help
 Usage: earsegmentationai [OPTIONS] COMMAND [ARGS]...

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.                                          │
│ --show-completion             Show completion for the current shell, to copy it or customize the installation.   │
│ --help                        Show this message and exit.                                                        │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ picture-capture                                                                                                  │
│ version                                                                                                          │
│ video-capture                                                                                                    │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## Webcam Mode 📷

```bash
python -m earsegmentationai.main webcam-capture --deviceid 1 --device "cpu"
python -m earsegmentationai.main webcam-capture --deviceid 1 --device "cuda:0"
```

## Image Mode 🖼️

```bash
python -m earsegmentationai.main picture-capture --folderpath "/path/filename.png" --device "cpu"
python -m earsegmentationai.main picture-capture --folderpath "/path/filename.png" --device "cuda:0"
```

## Youtube Video 📸 ✨

<p>
<a href="https://www.youtube.com/watch?v=5Puxj7Q0EEo"><img alt="Youtube" src="https://img.shields.io/badge/Youtube-FF0000?logo=Youtube&logoColor=white"></a>
</p>
