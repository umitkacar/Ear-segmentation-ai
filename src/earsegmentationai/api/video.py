"""Video processing API for ear segmentation."""

from pathlib import Path
from typing import Callable, Optional, Union

import cv2
import numpy as np

from earsegmentationai.api.base import BaseProcessor, ProcessingResult
from earsegmentationai.core.predictor import StreamPredictor
from earsegmentationai.preprocessing.validators import (
    validate_camera_id,
    validate_directory,
    validate_video_path,
)
from earsegmentationai.utils.exceptions import ProcessingError, VideoError
from earsegmentationai.utils.logging import get_logger

logger = get_logger(__name__)


class VideoProcessor(BaseProcessor):
    """Video processor for ear segmentation.

    This class provides high-level API for processing video files
    and real-time camera streams.
    """

    def __init__(
        self,
        config=None,
        model_manager=None,
        device=None,
        threshold=0.5,
        skip_frames=0,
        smooth_masks=True,
    ):
        """Initialize video processor.

        Args:
            config: Configuration object
            model_manager: Model manager
            device: Processing device
            threshold: Binary threshold
            skip_frames: Number of frames to skip between predictions
            smooth_masks: Whether to apply temporal smoothing
        """
        super().__init__(config, model_manager, device, threshold)

        # Create stream predictor
        self.predictor = StreamPredictor(
            model_manager=self.model_manager,
            transform=self.transform,
            threshold=threshold,
            skip_frames=skip_frames,
            smooth_masks=smooth_masks,
        )

        self.skip_frames = skip_frames
        self.smooth_masks = smooth_masks

    def process(
        self,
        source: Union[str, Path, int],
        output_path: Optional[Union[str, Path]] = None,
        display: bool = True,
        callback: Optional[Callable[[ProcessingResult], None]] = None,
        max_frames: Optional[int] = None,
    ) -> dict:
        """Process video source.

        Args:
            source: Video source - can be:
                - Path to video file
                - Camera device ID (integer)
                - Camera device path (string)
                - Video URL (rtsp://, http://)
            output_path: Optional path to save output video
            display: Whether to display results in window
            callback: Optional callback function for each frame result
            max_frames: Maximum number of frames to process

        Returns:
            Dictionary with processing statistics

        Raises:
            VideoError: If video processing fails
        """
        try:
            # Determine source type
            if isinstance(source, int) or (
                isinstance(source, str) and source.isdigit()
            ):
                # Camera ID
                return self.process_camera(
                    int(source), output_path, display, callback, max_frames
                )
            elif isinstance(source, (str, Path)):
                source_str = str(source)
                if source_str.startswith(("http://", "https://", "rtsp://")):
                    # Video stream URL
                    return self.process_stream(
                        source_str, output_path, display, callback, max_frames
                    )
                else:
                    # Video file
                    return self.process_video_file(
                        Path(source),
                        output_path,
                        display,
                        callback,
                        max_frames,
                    )
            else:
                raise VideoError(f"Unsupported source type: {type(source)}")

        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            raise VideoError(f"Failed to process video: {e}")

    def process_video_file(
        self,
        video_path: Path,
        output_path: Optional[Path] = None,
        display: bool = True,
        callback: Optional[Callable] = None,
        max_frames: Optional[int] = None,
    ) -> dict:
        """Process video file.

        Args:
            video_path: Path to video file
            output_path: Optional output video path
            display: Whether to display results
            callback: Optional frame callback
            max_frames: Maximum frames to process

        Returns:
            Processing statistics
        """
        # Validate input
        video_path = validate_video_path(video_path)
        logger.info(f"Processing video file: {video_path}")

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise VideoError(f"Failed to open video file: {video_path}")

        try:
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            logger.info(
                f"Video properties: {width}x{height}, {fps} FPS, "
                f"{total_frames} frames"
            )

            # Process video
            stats = self._process_video_capture(
                cap, output_path, display, callback, max_frames, fps
            )

            stats["source"] = str(video_path)
            stats["total_frames_in_video"] = total_frames

            return stats

        finally:
            cap.release()

    def process_camera(
        self,
        camera_id: Union[int, str] = 0,
        output_path: Optional[Path] = None,
        display: bool = True,
        callback: Optional[Callable] = None,
        max_frames: Optional[int] = None,
    ) -> dict:
        """Process camera stream.

        Args:
            camera_id: Camera device ID or path
            output_path: Optional output video path
            display: Whether to display results
            callback: Optional frame callback
            max_frames: Maximum frames to process

        Returns:
            Processing statistics
        """
        # Validate camera
        camera_id = validate_camera_id(camera_id)
        logger.info(f"Processing camera: {camera_id}")

        # Open camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise VideoError(f"Failed to open camera: {camera_id}")

        try:
            # Set camera properties for better performance
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Get default FPS (usually 30 for webcams)
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

            # Process stream
            stats = self._process_video_capture(
                cap, output_path, display, callback, max_frames, fps
            )

            stats["source"] = f"camera_{camera_id}"

            return stats

        finally:
            cap.release()

    def process_stream(
        self,
        stream_url: str,
        output_path: Optional[Path] = None,
        display: bool = True,
        callback: Optional[Callable] = None,
        max_frames: Optional[int] = None,
    ) -> dict:
        """Process video stream from URL.

        Args:
            stream_url: Stream URL (http://, rtsp://)
            output_path: Optional output video path
            display: Whether to display results
            callback: Optional frame callback
            max_frames: Maximum frames to process

        Returns:
            Processing statistics
        """
        logger.info(f"Processing stream: {stream_url}")

        # Open stream
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            raise VideoError(f"Failed to open stream: {stream_url}")

        try:
            # Get FPS (might not be accurate for streams)
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

            # Process stream
            stats = self._process_video_capture(
                cap, output_path, display, callback, max_frames, fps
            )

            stats["source"] = stream_url

            return stats

        finally:
            cap.release()

    def _process_video_capture(
        self,
        cap: cv2.VideoCapture,
        output_path: Optional[Path],
        display: bool,
        callback: Optional[Callable],
        max_frames: Optional[int],
        fps: int,
    ) -> dict:
        """Internal method to process video capture.

        Args:
            cap: OpenCV VideoCapture object
            output_path: Optional output path
            display: Whether to display
            callback: Optional callback
            max_frames: Maximum frames
            fps: Frames per second

        Returns:
            Processing statistics
        """
        # Initialize statistics
        stats = {
            "frames_processed": 0,
            "frames_with_ear": 0,
            "average_fps": 0.0,
            "processing_time": 0.0,
        }

        # Setup video writer if output requested
        writer = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Get frame size
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Create writer
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(
                str(output_path), fourcc, fps, (width, height)
            )

            if not writer.isOpened():
                raise VideoError(
                    f"Failed to create output video: {output_path}"
                )

        # Reset predictor state
        self.predictor.reset()

        # Process frames
        import time

        start_time = time.time()
        frame_times = []

        try:
            while True:
                # Check frame limit
                if max_frames and stats["frames_processed"] >= max_frames:
                    break

                # Read frame
                frame_start = time.time()
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                result = self._process_frame(frame)

                # Update statistics
                stats["frames_processed"] += 1
                if result and result.has_ear:
                    stats["frames_with_ear"] += 1

                # Callback
                if callback and result:
                    callback(result)

                # Display
                if display and result:
                    vis_frame = self.visualizer.visualize_mask(
                        frame, result.mask
                    )

                    # Add statistics overlay
                    self._add_overlay(vis_frame, stats, result)

                    cv2.imshow("Ear Segmentation", vis_frame)

                    # Check for quit
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("User requested quit")
                        break

                # Write output
                if writer and result:
                    # Write visualization or original with mask
                    if display:
                        writer.write(vis_frame)
                    else:
                        vis_frame = self.visualizer.visualize_mask(
                            frame, result.mask
                        )
                        writer.write(vis_frame)

                # Track timing
                frame_time = time.time() - frame_start
                frame_times.append(frame_time)

        finally:
            # Cleanup
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()

            # Calculate final statistics
            total_time = time.time() - start_time
            stats["processing_time"] = total_time

            if frame_times:
                stats["average_fps"] = 1.0 / np.mean(frame_times)

            if stats["frames_processed"] > 0:
                stats["detection_rate"] = (
                    stats["frames_with_ear"] / stats["frames_processed"] * 100
                )
            else:
                stats["detection_rate"] = 0.0

            logger.info(
                f"Video processing complete. "
                f"Processed {stats['frames_processed']} frames in {total_time:.1f}s "
                f"({stats['average_fps']:.1f} FPS). "
                f"Detection rate: {stats['detection_rate']:.1f}%"
            )

        return stats

    def _process_frame(self, frame: np.ndarray) -> Optional[ProcessingResult]:
        """Process single video frame.

        Args:
            frame: Video frame

        Returns:
            Processing result or None if skipped
        """
        try:
            # Run prediction
            mask = self.predictor.predict_frame(frame)

            if mask is None:
                return None

            # Create result
            result = ProcessingResult(
                image=frame,
                mask=mask,
                metadata={"frame_number": self.predictor.frame_count},
            )

            return result

        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            return None

    def _add_overlay(
        self,
        frame: np.ndarray,
        stats: dict,
        result: ProcessingResult,
    ) -> None:
        """Add statistics overlay to frame.

        Args:
            frame: Frame to modify
            stats: Statistics dictionary
            result: Processing result
        """
        # Create overlay text
        lines = [
            f"FPS: {stats.get('average_fps', 0):.1f}",
            f"Frame: {stats['frames_processed']}",
            f"Detection: {'Yes' if result.has_ear else 'No'}",
        ]

        if result.has_ear:
            lines.append(f"Area: {result.ear_percentage:.1f}%")

        # Draw background rectangle
        y_offset = 10
        for i, line in enumerate(lines):
            y = y_offset + i * 25
            cv2.rectangle(frame, (10, y - 20), (200, y + 5), (0, 0, 0), -1)
            cv2.putText(
                frame,
                line,
                (15, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

    def set_skip_frames(self, skip_frames: int) -> None:
        """Set frame skipping.

        Args:
            skip_frames: Number of frames to skip
        """
        self.skip_frames = skip_frames
        self.predictor.skip_frames = skip_frames

    def set_smooth_masks(self, smooth: bool) -> None:
        """Enable/disable mask smoothing.

        Args:
            smooth: Whether to smooth masks
        """
        self.smooth_masks = smooth
        self.predictor.smooth_masks = smooth
