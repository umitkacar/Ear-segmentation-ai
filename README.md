# Ear Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<p>
  <img alt="Python38" src="https://img.shields.io/badge/Python-3.8-3776AB.svg?logo=Python&logoColor=white"></img>
  <img alt="Python38" src="https://img.shields.io/badge/Python-3.9-3776AB.svg?logo=Python&logoColor=white"></img>
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-1.8.1+cpu-EE4C2C.svg?logo=PyTorch&logoColor=white"></img>
  <img alt="Torchvision" src="https://img.shields.io/badge/Torchvision-0.9.1+cpu-EE4C2C.svg?logo=PyTorch&logoColor=white"></img>
  <img alt="Poetry" src="https://img.shields.io/badge/Poetry-60A5FA.svg?logo=Poetry&logoColor=white"></img>
  <img alt="Black" src="https://img.shields.io/badge/code%20style-black-black"></img>
  <img alt="Mypy" src="https://img.shields.io/badge/mypy-checked-blue"></img>
  <img alt="isort" src="https://img.shields.io/badge/isort-checked-yellow"></img>
</p>

## Download Model

<p>
<a href="https://drive.google.com/drive/folders/1_M_8uuTgU__wRVbE2g2jOagLD7Eog1F8?usp=sharing"><img alt="GoogleDrive" src="https://img.shields.io/badge/GoogleDrive-4285F4?logo=GoogleDrive&logoColor=white"></a>
</p>

## Requirements :open_file_folder:

* Python 3.8 to Python3.9 (Virtualenv recommended)
* Download Ear Model file and put into `model_ear` folder
* Image mode require `test-images` folder and add jpg file(s) 
* Optionally poetry

Note: Python3.9+ not working at the moment.
## Installation


### Pip installation

```
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html 
```

### Poetry installation 

```
poetry shell
poetry install
poe pytorch_cpu 
```

## Optional (If you have multiple python installation)

### Python 3.8 and 3.9

```
poetry env use $(which python3.8)
poetry shell
poetry install
poe pytorch_cpu 
```

## Usage

Webcam Mode
```
python Deploy_ear_segmentation_webcam.py
```

Image Mode
```
python Deploy_ear_segmentation_image.py
```

## Youtube Video

<p>
<a href="https://www.youtube.com/watch?v=5Puxj7Q0EEo"><img alt="Youtube" src="https://img.shields.io/badge/Youtube-FF0000?logo=Youtube&logoColor=white"></a>
</p>



