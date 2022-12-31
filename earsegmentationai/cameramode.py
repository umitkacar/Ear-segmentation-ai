from functools import partial
from typing import Any, Optional

import cv2
import imgviz
import numpy as np
import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.decoders.unet.model import Unet

from .const import ENCODER, ENCODER_WEIGHTS, MODEL_PATH
from .download_model import get_model
from .predict_mask import get_prediction
from .preprocessing import get_preprocessing


def ear_segmentation_webcam(
    video_capture: int = 0, record: bool = False, device="cpu"
):
    get_model()
    return_frame_status: bool = True
    video_record_out: Optional[cv2.VideoWriter] = None

    model: Unet = torch.load(MODEL_PATH, map_location=device)
    model.eval()
    model.to(device)

    preprocess_fn: partial[Any] = smp.encoders.get_preprocessing_fn(
        ENCODER, ENCODER_WEIGHTS
    )
    preprocessing = get_preprocessing(preprocess_fn)

    # Webcam acquisition
    cap: cv2.VideoCapture = cv2.VideoCapture(video_capture)

    # If Camera Device is not opened, exit the program
    if not cap.isOpened():
        print(
            "Video device couldn't be opened. Please change your video device number"
        )
        exit()

    # # Video Writer
    if record:
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        print(frame_width)
        print(frame_height)
        frame_fps = 20

        video_record_out = cv2.VideoWriter(
            "output.avi",
            cv2.VideoWriter_fourcc("M", "J", "P", "G"),
            frame_fps,
            (640, 480),
        )

    print("Press Q to exit from camera")
    while return_frame_status is True:

        return_frame_status, camera_frame_bgr = cap.read()
        # frame = cv2.flip(cv2.transpose(frame), flipCode=1)

        frame_rgb = cv2.cvtColor(camera_frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb_resize = cv2.resize(frame_rgb, (480, 320))

        with torch.no_grad():

            pr_mask_orj = cv2.resize(
                cv2.Mat(
                    get_prediction(
                        preprocessing=preprocessing,
                        image=frame_rgb_resize,
                        device=device,
                        model=model,
                    )
                ),
                (frame_rgb.shape[1], frame_rgb.shape[0]),
            )

            # colorize label image
            class_label = pr_mask_orj.squeeze().astype(int)
            labelviz: np.ndarray = imgviz.label2rgb(
                class_label,
                image=frame_rgb,
                label_names=["bg", "ear"],
                font_size=25,
                loc="rb",
            )

            camera_frame_bgr = cv2.cvtColor(labelviz, cv2.COLOR_BGR2RGB)

        # cv2.imshow('Ear Mask',pr_mask)
        cv2.imshow("Ear Image", camera_frame_bgr)

        if record:
            video_record_out.write(camera_frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()

    if record:
        video_record_out.release()

    cv2.destroyAllWindows()
