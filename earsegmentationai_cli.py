import argparse
from sys import exit

from earsegmentationai import camera_mode, image_mode


def main():
    parser = argparse.ArgumentParser()

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
