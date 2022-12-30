# Efficient and Lightweight Ear Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/umitkacar/Ear-segmentation-ai/main.svg)](https://results.pre-commit.ci/latest/github/umitkacar/Ear-segmentation-ai/main)
<p>
  <img alt="Python38" src="https://img.shields.io/badge/Python-3.8-3776AB.svg?logo=Python&logoColor=white"/>
  <img alt="Python39" src="https://img.shields.io/badge/Python-3.9-3776AB.svg?logo=Python&logoColor=white"/>
  <img alt="Python310" src="https://img.shields.io/badge/Python-3.10-3776AB.svg?logo=Python&logoColor=white"/>
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-v1.13.1-EE4C2C.svg?logo=PyTorch&logoColor=white"/>
  <img alt="Torchvision" src="https://img.shields.io/badge/Torchvision-v0.14.1-EE4C2C.svg?logo=PyTorch&logoColor=white"/>
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

```
usage: earsegmentationai_cli.py [-h] -m {c,p} [-d [{cpu,cuda}]] [-fp FOLDERPATH] [-id [DEVICEID]]

options:
  -h, --help            show this help message and exit
  -m {c,p}, --mode {c,p}
                        Select camera or picture mode
  -d [{cpu,cuda}], --device [{cpu,cuda}]
                        Run in gpu or cpu mode
  -fp FOLDERPATH, --folderpath FOLDERPATH
                        Folder path for image(s) for image mode only
  -id [DEVICEID], --deviceId [DEVICEID]
                        Camera deviceId /dev/videoX for camera mode only
```

## Webcam Mode 📷

```bash
python earsegmentationai_cli.py --mode c --device cpu
python earsegmentationai_cli.py --mode c --device cuda
python earsegmentationai_cli.py --mode c --deviceId 1 --device cuda
```

## Image Mode 🖼️

```bash
python earsegmentationai_cli.py --mode p --fp /path/xxx/
```

## Youtube Video 📸 ✨

<p>
<a href="https://www.youtube.com/watch?v=5Puxj7Q0EEo"><img alt="Youtube" src="https://img.shields.io/badge/Youtube-FF0000?logo=Youtube&logoColor=white"></a>
</p>
