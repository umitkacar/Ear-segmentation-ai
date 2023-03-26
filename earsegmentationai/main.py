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

from os import path

import typer

from earsegmentationai import (
    __copyright__,
    __version__,
    camera_mode,
    image_mode,
)

app = typer.Typer()


@app.command()
def version():
    print(f"Ear Segmentation Ai - {__version__}")
    print(__copyright__)


@app.command()
def webcam_picture(
    deviceId: int = typer.Option(1), device: str = typer.Option("cuda:0")
) -> None:
    if deviceId < 0:
        print("Device cannot be lower than 0")
        return

    camera_mode.ear_segmentation_camera(
        video_capture=deviceId, record=False, device=device
    )
    return


@app.command()
def picture_capture(folderPath: str = "", device: str = "cuda:0") -> None:
    if path.isfile(folderPath) is False:
        print("Please use valid file path")
        return

    if folderPath.lower().endswith((".png", ".jpg", ".jpeg")) is False:
        print("Please use valid file extension")
        return
    image_mode.ear_segmentation_image(folder_path=folderPath, device=device)
    return


if __name__ == "__main__":
    app()
