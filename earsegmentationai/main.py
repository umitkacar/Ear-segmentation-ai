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


import argparse
from sys import exit

from earsegmentationai import __version__, camera_mode, image_mode


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    parser.add_argument(
        "-m",
        "--mode",
        choices=["c", "p"],
        help="Select camera or picture mode",
        required=True,
    )

    parser.add_argument(
        "-d",
        "--device",
        nargs="?",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Run in gpu or cpu mode",
        type=str,
    )

    parser.add_argument(
        "-fp",
        "--folderpath",
        help="Folder path for image(s) for image mode only",
    )

    parser.add_argument(
        "-id",
        "--deviceId",
        nargs="?",
        default=1,
        help="Camera deviceId /dev/videoX for camera mode only",
        type=int,
    )

    # Read arguments from command line
    args = parser.parse_args()

    if args.mode == "c":
        if args.deviceId is not None:
            camera_mode.ear_segmentation_camera(
                video_capture=args.deviceId, record=False, device=args.device
            )
    elif args.mode == "p":
        if args.folderpath is not None:
            print(args.folderpath)
            image_mode.ear_segmentation_image(
                folder_path=args.folderpath, device=args.device
            )
    else:
        print("Folder path required")
        exit()


if __name__ == "__main__":
    main()
