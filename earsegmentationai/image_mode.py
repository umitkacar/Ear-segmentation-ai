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

from glob import glob

import cv2
import segmentation_models_pytorch as smp
import torch
from cv2 import Mat

from earsegmentationai import ENCODER, ENCODER_WEIGHTS, MODEL_PATH

from .download_model import get_model
from .pre_processing import get_preprocessing
from .predict_mask import get_prediction


def ear_segmentation_image(folder_path: str, device="cpu") -> None:

    get_model()
    model = torch.load(MODEL_PATH, map_location=device)
    model.eval()
    model.to(device)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        ENCODER, ENCODER_WEIGHTS
    )
    preprocessing = get_preprocessing(preprocessing_fn)

    # JPEG/PNG/JPG
    data_samples: list[str] = (
        glob(folder_path + "/*.jpg")
        + glob(folder_path + "/*.jpeg")
        + glob(folder_path + "/*.png")
    )

    for path in data_samples:

        img: Mat = cv2.imread(path)
        cv2.resize(img, (480, 320))
        image: Mat = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with torch.no_grad():

            predict_mask = get_prediction(
                preprocessing=preprocessing,
                image=image,
                device=device,
                model=model,
            )

        cv2.imshow("Ear Mask", predict_mask)
        cv2.imshow("Ear Image", img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
