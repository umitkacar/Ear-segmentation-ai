MODELNAME = "earsegmentation_model_v1_46.pth"
REPOURL = "https://github.com/umitkacar/Ear-segmentation-ai"
MODELURL = f"{REPOURL}/releases/download/v1.0.0/{MODELNAME}"
MODEL_PATH = f"./earsegmentationai/model_ear/{MODELNAME}"
ENCODER = "resnet18"
ENCODER_WEIGHTS = "imagenet"
__version__ = "1.0.2"
__copyright__ = (
    "Copyright 2023, The Efficient and Lightweight Ear Segmentation Project"
)
