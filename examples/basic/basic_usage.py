"""Basic usage example for Ear Segmentation AI."""

import cv2
import numpy as np
from pathlib import Path

# Import the main classes
from earsegmentationai import ImageProcessor, VideoProcessor


def process_single_image():
    """Example: Process a single image."""
    print("=== Processing Single Image ===")
    
    # Initialize processor
    processor = ImageProcessor(device="cpu")  # Use "cuda:0" for GPU
    
    # Process image from file
    image_path = Path(__file__).parent / "0210.png"
    if image_path.exists():
        result = processor.process(
            image_path,
            return_probability=True,
            return_visualization=True
        )
        
        # Check results
        print(f"Ear detected: {result.has_ear}")
        print(f"Ear area: {result.ear_percentage:.2f}% of image")
        
        if result.has_ear:
            bbox = result.get_bounding_box()
            print(f"Bounding box: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}")
        
        # Display visualization if available
        if "visualization" in result.metadata:
            cv2.imshow("Ear Segmentation Result", result.metadata["visualization"])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print(f"Image not found: {image_path}")
        print("Creating a sample image...")
        
        # Create a sample image
        sample_image = np.ones((256, 256, 3), dtype=np.uint8) * 128
        cv2.circle(sample_image, (128, 128), 50, (255, 255, 255), -1)
        
        # Process numpy array
        result = processor.process(sample_image)
        print(f"Sample image - Ear detected: {result.has_ear}")


def process_image_directory():
    """Example: Process all images in a directory."""
    print("\n=== Processing Image Directory ===")
    
    # Initialize processor
    processor = ImageProcessor(device="cpu")
    
    # Process directory
    image_dir = Path(__file__).parent
    batch_result = processor.process(
        image_dir,
        save_results=True,
        output_dir=image_dir / "output"
    )
    
    # Print summary
    print(f"Total images processed: {len(batch_result)}")
    print(f"Detection rate: {batch_result.detection_rate:.1f}%")
    print(f"Average ear area: {batch_result.average_ear_area:.0f} pixels")
    
    # Show individual results
    for i, result in enumerate(batch_result):
        print(f"  Image {i}: Ear={'Yes' if result.has_ear else 'No'}, "
              f"Area={result.ear_percentage:.1f}%")


def process_webcam():
    """Example: Process webcam stream."""
    print("\n=== Processing Webcam Stream ===")
    print("Press 'q' to quit")
    
    # Initialize processor with frame skipping for performance
    processor = VideoProcessor(
        device="cpu",
        skip_frames=2,  # Process every 3rd frame
        smooth_masks=True  # Smooth between frames
    )
    
    # Define callback to print stats
    def frame_callback(result):
        if result.has_ear:
            print(f"Frame {result.metadata.get('frame_number', 0)}: "
                  f"Ear detected, area={result.ear_percentage:.1f}%")
    
    try:
        # Process webcam (camera ID 0)
        stats = processor.process(
            0,  # Camera ID
            display=True,
            callback=frame_callback,
            max_frames=300  # Stop after 300 frames (~10 seconds at 30fps)
        )
        
        # Print final statistics
        print("\nWebcam processing complete:")
        print(f"Frames processed: {stats['frames_processed']}")
        print(f"Average FPS: {stats['average_fps']:.1f}")
        print(f"Detection rate: {stats['detection_rate']:.1f}%")
        
    except Exception as e:
        print(f"Webcam processing failed: {e}")
        print("Make sure a webcam is connected and accessible")


def process_with_custom_threshold():
    """Example: Process with custom threshold."""
    print("\n=== Custom Threshold Example ===")
    
    # Initialize processor
    processor = ImageProcessor(device="cpu")
    
    # Create test image
    test_image = np.ones((256, 256, 3), dtype=np.uint8) * 100
    cv2.ellipse(test_image, (128, 128), (60, 40), 0, 0, 360, (255, 255, 255), -1)
    
    # Process with different thresholds
    thresholds = [0.3, 0.5, 0.7]
    
    for threshold in thresholds:
        processor.set_threshold(threshold)
        result = processor.process(test_image)
        
        print(f"Threshold {threshold}: Ear area = {result.ear_percentage:.2f}%")


def process_from_url():
    """Example: Process image from URL."""
    print("\n=== Processing Image from URL ===")
    
    # Initialize processor
    processor = ImageProcessor(device="cpu")
    
    # Example URL (replace with actual image URL)
    image_url = "https://example.com/ear_image.jpg"
    
    try:
        result = processor.process_from_url(
            image_url,
            return_visualization=True
        )
        
        print(f"URL image - Ear detected: {result.has_ear}")
        
    except Exception as e:
        print(f"Failed to process URL: {e}")
        print("Make sure the URL points to a valid image")


def main():
    """Run all examples."""
    print("Ear Segmentation AI - Examples")
    print("=" * 40)
    
    # Run examples
    process_single_image()
    process_image_directory()
    # process_webcam()  # Uncomment to test webcam
    process_with_custom_threshold()
    # process_from_url()  # Uncomment with valid URL
    
    print("\nExamples complete!")


if __name__ == "__main__":
    main()