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

from typing import Optional

import cv2
import imgviz
import numpy as np
import torch

from earsegmentationai.ear_models import EarModel


def ear_segmentation_camera(
    video_capture: int = 0, record: bool = False, device="cpu"
):
    ear_model = EarModel(device=device)

    return_frame_status: bool = True
    video_record_out: Optional[cv2.VideoWriter] = None

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
                cv2.Mat(ear_model.get_prediction(image=frame_rgb_resize)),
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
