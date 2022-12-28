from glob import glob

import cv2
import segmentation_models_pytorch as smp
import torch

from earsegmentationai.const import (
    ENCODER,
    ENCODER_WEIGHTS,
    LOAD_MODEL_DEPLOY_PATH,
)
from earsegmentationai.preprocessing import get_preprocessing


def ear_segmentation_image(folder_path: str, device="cpu") -> None:

    model = torch.load(LOAD_MODEL_DEPLOY_PATH, map_location=device)
    model.eval()
    model.to(device)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        ENCODER, ENCODER_WEIGHTS
    )
    preprocessing = get_preprocessing(preprocessing_fn)

    # JPEG/PNG/JPG
    data_samples = (
        glob(folder_path + "/*.jpg")
        + glob(folder_path + "/*.jpeg")
        + glob(folder_path + "/*.png")
    )

    for path in data_samples:

        img = cv2.imread(path)
        img = cv2.resize(img, (480, 320))
        h, w = img.shape[:2]
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with torch.no_grad():

            sample = preprocessing(image=image)
            image = sample["image"]

            x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)

            pr_mask = model.predict(x_tensor)
            pr_mask = pr_mask.squeeze().cpu().numpy().round()

        cv2.imshow("Ear Mask", pr_mask)
        cv2.imshow("Ear Image", img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
