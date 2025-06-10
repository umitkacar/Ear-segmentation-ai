"""Backward compatibility layer for Ear Segmentation AI v1.x."""

import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from earsegmentationai.api.image import ImageProcessor
from earsegmentationai.api.video import VideoProcessor
from earsegmentationai.core.model import ModelManager


# Legacy constants (from v1.x)
ENCODER_NAME = "resnet18"
ENCODER_WEIGHTS = "imagenet"
MODEL_NAME = "earsegmentation_model_v1_46.pth"
MODEL_URL = f"https://github.com/umitkacar/Ear-segmentation-ai/releases/download/v1.0.0/{MODEL_NAME}"


class EarModel:
    """Legacy EarModel class for backward compatibility.
    
    This class provides the same interface as v1.x EarModel
    but uses the new architecture internally.
    
    .. deprecated:: 2.0.0
        Use :class:`earsegmentationai.api.image.ImageProcessor` instead.
    """
    
    def __init__(self):
        """Initialize legacy EarModel."""
        warnings.warn(
            "EarModel is deprecated. Use ImageProcessor or VideoProcessor instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        self.model = None
        self.device = "cuda:0"
        self._processor = None
        
        # Legacy paths
        self.project_dir = Path.home() / ".cache" / "earsegmentationai" / "models"
        self.model_folder = self.project_dir
        self.model_path = self.model_folder / MODEL_NAME
    
    def download_models(self) -> None:
        """Download models (legacy method).
        
        .. deprecated:: 2.0.0
            Models are downloaded automatically when needed.
        """
        warnings.warn(
            "download_models() is deprecated. Models are downloaded automatically.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Ensure processor is initialized (will download model)
        if self._processor is None:
            self._processor = ImageProcessor(device=self.device)
    
    def load_model(self, device: str = "cuda:0") -> None:
        """Load model (legacy method).
        
        Args:
            device: Device to load model on
            
        .. deprecated:: 2.0.0
            Models are loaded automatically when needed.
        """
        self.device = device
        
        if self._processor is None:
            self._processor = ImageProcessor(device=device)
        
        # Set legacy model reference
        self.model = self._processor.model_manager.model
    
    def predict(
        self,
        image: Union[str, np.ndarray],
        foler_path: Optional[str] = None  # Note: typo from v1.x
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict ear mask (legacy method).
        
        Args:
            image: Image path or numpy array
            foler_path: Folder path (unused, kept for compatibility)
            
        Returns:
            Tuple of (original_mask, resized_mask)
            
        .. deprecated:: 2.0.0
            Use ImageProcessor.process() instead.
        """
        if foler_path is not None:
            warnings.warn(
                "foler_path parameter is deprecated and ignored.",
                DeprecationWarning,
                stacklevel=2
            )
        
        # Ensure processor exists
        if self._processor is None:
            self.load_model(self.device)
        
        # Load image if path provided
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Failed to load image: {image}")
        
        # Get prediction
        result = self._processor.process(image)
        
        # Return in legacy format (both masks are the same in v2)
        return result.mask, result.mask


def process_image_mode(
    deviceId: int = 1,
    device: str = "cuda:0",
    folderPath: str = None
) -> None:
    """Legacy image mode function.
    
    .. deprecated:: 2.0.0
        Use CLI: earsegmentationai process-image
    """
    warnings.warn(
        "process_image_mode() is deprecated. Use CLI instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    if folderPath is None:
        raise ValueError("folderPath is required")
    
    processor = ImageProcessor(device=device)
    
    # Process single image or directory
    path = Path(folderPath)
    if path.is_file():
        result = processor.process(path, return_visualization=True)
        
        if "visualization" in result.metadata:
            cv2.imshow("Ear Segmentation", result.metadata["visualization"])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        # Process directory
        results = processor.process(path)
        print(f"Processed {len(results)} images")
        print(f"Detection rate: {results.detection_rate:.1f}%")


def process_camera_mode(
    deviceId: int = 1,
    device: str = "cuda:0",
    record: bool = False
) -> None:
    """Legacy camera mode function.
    
    .. deprecated:: 2.0.0
        Use CLI: earsegmentationai webcam
    """
    warnings.warn(
        "process_camera_mode() is deprecated. Use CLI instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    processor = VideoProcessor(device=device)
    
    output_path = "output.avi" if record else None
    
    stats = processor.process(
        deviceId,
        output_path=output_path,
        display=True
    )
    
    print(f"Processed {stats['frames_processed']} frames")
    print(f"Average FPS: {stats['average_fps']:.1f}")


# Legacy function aliases
run_camera_mode = process_camera_mode
run_image_mode = process_image_mode


# Migration helper
def migrate_v1_to_v2():
    """Print migration guide from v1.x to v2.0."""
    guide = """
    =====================================
    Migration Guide: v1.x to v2.0
    =====================================
    
    1. Model Loading:
       OLD: model = EarModel()
            model.download_models()
            model.load_model("cuda:0")
       
       NEW: processor = ImageProcessor(device="cuda:0")
            # Models are downloaded automatically
    
    2. Image Processing:
       OLD: mask_orig, mask_resized = model.predict(image_path)
       
       NEW: result = processor.process(image_path)
            mask = result.mask
            has_ear = result.has_ear
    
    3. Camera Mode:
       OLD: from camera_mode import process_camera_mode
            process_camera_mode(deviceId=0, device="cuda:0")
       
       NEW: processor = VideoProcessor(device="cuda:0")
            processor.process(0, display=True)
       
       OR:  earsegmentationai webcam --device-id 0
    
    4. CLI Commands:
       OLD: python -m earsegmentationai.main webcam-capture
       NEW: earsegmentationai webcam
       
       OLD: python -m earsegmentationai.main picture-capture
       NEW: earsegmentationai process-image
    
    For more details, see the documentation.
    """
    print(guide)


if __name__ == "__main__":
    migrate_v1_to_v2()