"""
Earsegmentationai

Copyright (c) 2023 Ear-Segmentation-Ai Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

from functools import partial
from os import path
from pathlib import Path
from shutil import copyfileobj
from typing import List

from albumentations import Compose, Lambda
from cv2 import Mat
from requests import Response, get
from segmentation_models_pytorch import encoders as smp_encoders
from segmentation_models_pytorch.decoders.unet.model import Unet
from torch import Tensor, from_numpy
from torch import load as torch_load
from tqdm.auto import tqdm

from earsegmentationai import ENCODER, ENCODER_WEIGHTS, MODEL_PATH, MODELURL


class EarModel:
    """
    EarModel class.
    Args:
        device (str): select cpu or gpu

    Attributes:
        preprocessing (callbale): data normalization function
            (can be specific for each pretrained neural network)
    """

    def __init__(self, device: str):
        self.device = device
        self.model: Unet = self.get_model()

    def get_model(self) -> Unet:
        if path.exists(MODEL_PATH):
            return self.load_model()

        print("Downloading necessary deep learning model \n")
        res: Response = get(MODELURL, stream=True, allow_redirects=True)
        if res.status_code != 200:
            res.raise_for_status()
            raise RuntimeError(
                f"Request to {MODELURL} returned status code {res.status_code}"
            )

        file_size: int = int(res.headers.get("Content-Length", 0))
        foler_path: Path = Path(MODEL_PATH).expanduser().resolve()
        foler_path.parent.mkdir(parents=True, exist_ok=True)

        desc = "(Unknown total file size)" if file_size == 0 else ""
        res.raw.read = partial(
            res.raw.read, decode_content=True
        )  # Decompress if needed
        with tqdm.wrapattr(
            res.raw, "read", total=file_size, desc=desc
        ) as r_raw:
            with foler_path.open("wb") as f:
                copyfileobj(r_raw, f)

        return self.load_model()

    def load_model(self):
        self.model = torch_load(MODEL_PATH, map_location=self.device)
        self.model.eval()
        self.model.to(self.device)
        return self.model

    def get_prediction(self, image: Mat) -> Tensor:
        cv_image: Mat = self.get_preprocessing()(image=image)["image"]
        x_tensor: Tensor = from_numpy(cv_image).to(self.device).unsqueeze(0)
        predict_mask: Tensor = self.model.predict(x_tensor)
        return predict_mask.squeeze().cpu().numpy().round()

    def to_tensor(self, x, **_):
        return x.transpose(2, 0, 1).astype("float32")

    def get_preprocessing(self) -> Compose:
        """Construct preprocessing transform
        Return:
            transform: albumentations.Compose
        """

        transform: List[Lambda] = [
            Lambda(
                image=smp_encoders.get_preprocessing_fn(
                    ENCODER, ENCODER_WEIGHTS
                )
            ),
            Lambda(image=self.to_tensor),
        ]
        return Compose(transform)
