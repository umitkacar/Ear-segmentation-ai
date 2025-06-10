"""Image processing API for ear segmentation."""

from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np

from earsegmentationai.api.base import (
    BaseProcessor,
    BatchProcessingResult,
    ProcessingResult,
)
from earsegmentationai.preprocessing.validators import (
    get_image_files,
    validate_directory,
    validate_image_array,
    validate_image_path,
)
from earsegmentationai.utils.exceptions import ProcessingError
from earsegmentationai.utils.logging import get_logger

logger = get_logger(__name__)


class ImageProcessor(BaseProcessor):
    """Image processor for ear segmentation.

    This class provides high-level API for processing single images,
    multiple images, and directories of images.
    """

    def process(
        self,
        input_data: Union[
            str, Path, np.ndarray, List[Union[str, Path, np.ndarray]]
        ],
        return_probability: bool = False,
        return_visualization: bool = False,
        save_results: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Union[ProcessingResult, BatchProcessingResult]:
        """Process image(s) for ear segmentation.

        Args:
            input_data: Input image(s) - can be:
                - Path to single image file
                - Path to directory containing images
                - Numpy array of single image
                - List of paths or numpy arrays
            return_probability: Whether to include probability maps in results
            return_visualization: Whether to include visualization images
            save_results: Whether to save results to disk
            output_dir: Directory to save results (required if save_results=True)

        Returns:
            ProcessingResult for single image or BatchProcessingResult for multiple

        Raises:
            ProcessingError: If processing fails
        """
        try:
            # Determine input type and process accordingly
            if isinstance(input_data, (str, Path)):
                path = Path(input_data)
                if path.is_file():
                    # Single image file
                    return self._process_single_file(
                        path,
                        return_probability,
                        return_visualization,
                        save_results,
                        output_dir,
                    )
                elif path.is_dir():
                    # Directory of images
                    return self._process_directory(
                        path,
                        return_probability,
                        return_visualization,
                        save_results,
                        output_dir,
                    )
                else:
                    raise ProcessingError(f"Path does not exist: {path}")

            elif isinstance(input_data, np.ndarray):
                # Single numpy array
                return self._process_single_array(
                    input_data,
                    return_probability,
                    return_visualization,
                    save_results,
                    output_dir,
                )

            elif isinstance(input_data, list):
                # List of inputs
                return self._process_list(
                    input_data,
                    return_probability,
                    return_visualization,
                    save_results,
                    output_dir,
                )

            else:
                raise ProcessingError(
                    f"Unsupported input type: {type(input_data)}"
                )

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise ProcessingError(f"Failed to process image(s): {e}")

    def _process_single_file(
        self,
        path: Path,
        return_probability: bool,
        return_visualization: bool,
        save_results: bool,
        output_dir: Optional[Path],
    ) -> ProcessingResult:
        """Process single image file."""
        # Validate path
        path = validate_image_path(path)
        logger.info(f"Processing image: {path}")

        # Load image
        image = cv2.imread(str(path))
        if image is None:
            raise ProcessingError(f"Failed to load image: {path}")

        # Process
        result = self._process_single_array(
            image,
            return_probability,
            return_visualization,
            save_results,
            output_dir,
            filename=path.stem,
        )

        # Add metadata
        result.metadata["source_path"] = str(path)
        result.metadata["filename"] = path.name

        return result

    def _process_single_array(
        self,
        image: np.ndarray,
        return_probability: bool,
        return_visualization: bool,
        save_results: bool,
        output_dir: Optional[Path],
        filename: Optional[str] = None,
    ) -> ProcessingResult:
        """Process single image array."""
        # Validate image
        image = validate_image_array(image)

        # Run prediction
        if return_probability:
            mask, probability_map = self.predictor.predict(
                image, return_probability=True
            )
        else:
            mask = self.predictor.predict(image, return_probability=False)
            probability_map = None

        # Create result
        result = ProcessingResult(
            image=image,
            mask=mask,
            probability_map=probability_map,
            metadata={"filename": filename or "image"},
        )

        # Add visualization if requested
        if return_visualization:
            vis_image = self.visualizer.visualize_mask(image, mask)
            result.metadata["visualization"] = vis_image

        # Save results if requested
        if save_results and output_dir:
            self._save_result(result, output_dir)

        return result

    def _process_directory(
        self,
        directory: Path,
        return_probability: bool,
        return_visualization: bool,
        save_results: bool,
        output_dir: Optional[Path],
    ) -> BatchProcessingResult:
        """Process directory of images."""
        # Validate directory
        directory = validate_directory(directory)

        # Get image files
        image_files = get_image_files(directory)
        if not image_files:
            logger.warning(f"No images found in {directory}")
            return BatchProcessingResult([])

        logger.info(f"Processing {len(image_files)} images from {directory}")

        # Process as list
        return self._process_list(
            image_files,
            return_probability,
            return_visualization,
            save_results,
            output_dir,
        )

    def _process_list(
        self,
        input_list: List[Union[str, Path, np.ndarray]],
        return_probability: bool,
        return_visualization: bool,
        save_results: bool,
        output_dir: Optional[Path],
    ) -> BatchProcessingResult:
        """Process list of inputs."""
        results = []

        # Prepare images and metadata
        images = []
        metadata_list = []

        for i, item in enumerate(input_list):
            if isinstance(item, (str, Path)):
                # Load image from file
                path = validate_image_path(item)
                image = cv2.imread(str(path))
                if image is None:
                    logger.error(f"Failed to load image: {path}")
                    continue

                images.append(image)
                metadata_list.append(
                    {
                        "source_path": str(path),
                        "filename": path.name,
                        "index": i,
                    }
                )

            elif isinstance(item, np.ndarray):
                # Use array directly
                image = validate_image_array(item)
                images.append(image)
                metadata_list.append({"filename": f"image_{i}", "index": i})

            else:
                logger.error(
                    f"Unsupported item type at index {i}: {type(item)}"
                )
                continue

        if not images:
            logger.warning("No valid images to process")
            return BatchProcessingResult([])

        # Batch prediction
        logger.info(f"Running batch prediction on {len(images)} images")

        if return_probability:
            predictions = self.predictor.predict_batch(
                images, return_probability=True
            )
            masks = [pred[0] for pred in predictions]
            probability_maps = [pred[1] for pred in predictions]
        else:
            masks = self.predictor.predict_batch(
                images, return_probability=False
            )
            probability_maps = [None] * len(masks)

        # Create results
        for i, (image, mask, prob_map, metadata) in enumerate(
            zip(images, masks, probability_maps, metadata_list)
        ):
            result = ProcessingResult(
                image=image,
                mask=mask,
                probability_map=prob_map,
                metadata=metadata,
            )

            # Add visualization if requested
            if return_visualization:
                vis_image = self.visualizer.visualize_mask(image, mask)
                result.metadata["visualization"] = vis_image

            results.append(result)

        batch_result = BatchProcessingResult(results)

        # Save results if requested
        if save_results and output_dir:
            self._save_batch_results(batch_result, output_dir)

        logger.info(
            f"Batch processing complete. "
            f"Detection rate: {batch_result.detection_rate:.1f}%"
        )

        return batch_result

    def _save_result(
        self,
        result: ProcessingResult,
        output_dir: Path,
    ) -> None:
        """Save single result to disk."""
        output_dir = validate_directory(output_dir, create=True)
        filename = result.metadata.get("filename", "result")

        # Save mask
        mask_path = output_dir / f"{filename}_mask.png"
        cv2.imwrite(str(mask_path), result.mask * 255)

        # Save probability map if available
        if result.probability_map is not None:
            prob_path = output_dir / f"{filename}_probability.png"
            prob_uint8 = (result.probability_map * 255).astype(np.uint8)
            cv2.imwrite(str(prob_path), prob_uint8)

        # Save visualization if available
        if "visualization" in result.metadata:
            vis_path = output_dir / f"{filename}_visualization.png"
            cv2.imwrite(str(vis_path), result.metadata["visualization"])

        logger.debug(f"Saved results for {filename} to {output_dir}")

    def _save_batch_results(
        self,
        batch_result: BatchProcessingResult,
        output_dir: Path,
    ) -> None:
        """Save batch results to disk."""
        output_dir = validate_directory(output_dir, create=True)

        # Save individual results
        for result in batch_result:
            self._save_result(result, output_dir)

        # Save summary
        import json

        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(batch_result.get_summary(), f, indent=2)

        logger.info(f"Saved batch results to {output_dir}")

    def process_from_url(
        self,
        url: str,
        return_probability: bool = False,
        return_visualization: bool = False,
    ) -> ProcessingResult:
        """Process image from URL.

        Args:
            url: Image URL
            return_probability: Whether to return probability map
            return_visualization: Whether to return visualization

        Returns:
            Processing result
        """
        import requests
        from io import BytesIO

        try:
            # Download image
            logger.info(f"Downloading image from {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Decode image
            image_bytes = BytesIO(response.content)
            image_array = np.asarray(
                bytearray(image_bytes.read()), dtype=np.uint8
            )
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image is None:
                raise ProcessingError("Failed to decode image from URL")

            # Process
            result = self._process_single_array(
                image,
                return_probability,
                return_visualization,
                False,
                None,
                filename=url.split("/")[-1].split("?")[0],
            )

            result.metadata["source_url"] = url

            return result

        except Exception as e:
            raise ProcessingError(f"Failed to process image from URL: {e}")
