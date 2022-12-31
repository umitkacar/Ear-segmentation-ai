from glob import glob

import cv2
import segmentation_models_pytorch as smp
import torch
from cv2 import Mat

from .const import ENCODER, ENCODER_WEIGHTS, MODEL_PATH
from .download_model import get_model
from .predict_mask import get_prediction
from .preprocessing import get_preprocessing


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
