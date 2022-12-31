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

from cv2 import COLOR_BGR2RGB
from cv2 import Mat as cv_Mat
from cv2 import cvtColor
from cv2 import destroyAllWindows as cv_destroyAllWindows
from cv2 import imread as cv_imread
from cv2 import imshow as cv_imshow
from cv2 import resize as cv_resize
from cv2 import waitKey as cv_waitKey
from torch.autograd.grad_mode import no_grad

from earsegmentationai.ear_models import EarModel


def ear_segmentation_image(folder_path: str, device="cpu") -> None:

    ear_model = EarModel(device=device)

    # JPEG/PNG/JPG
    data_samples: list[str] = (
        glob(folder_path + "/*.jpg")
        + glob(folder_path + "/*.jpeg")
        + glob(folder_path + "/*.png")
    )

    for path in data_samples:

        img: cv_Mat = cv_imread(path)
        cv_resize(img, (480, 320))
        image: cv_Mat = cvtColor(img, COLOR_BGR2RGB)

        with no_grad():

            predict_mask = ear_model.get_prediction(image=image)

        cv_imshow("Ear Mask", predict_mask)
        cv_imshow("Ear Image", img)
        cv_waitKey(0)

    cv_destroyAllWindows()
