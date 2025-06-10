# Image Processing API

## ImageProcessor

The main class for processing static images.

### Class Definition

```python
from earsegmentationai import ImageProcessor
```

### Constructor

```python
ImageProcessor(
    device: str = "cpu",
    threshold: float = 0.5,
    batch_size: int = 1
)
```

**Parameters:**
- `device` (str): Processing device ("cpu" or "cuda:0")
- `threshold` (float): Binary threshold for mask generation (0.0 to 1.0)
- `batch_size` (int): Batch size for processing multiple images

### Methods

#### process()

Process single or multiple images for ear segmentation.

```python
def process(
    input_data: Union[str, Path, np.ndarray, List[Union[str, Path, np.ndarray]]],
    return_probability: bool = False,
    return_visualization: bool = False,
    save_results: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
) -> Union[ProcessingResult, BatchProcessingResult]
```

**Parameters:**
- `input_data`: Input image(s) - can be:
  - Path to single image file
  - Path to directory containing images
  - Numpy array of single image
  - List of paths or numpy arrays
- `return_probability` (bool): Include probability maps in results
- `return_visualization` (bool): Include visualization images
- `save_results` (bool): Save results to disk
- `output_dir` (Optional[Union[str, Path]]): Output directory for saved results

**Returns:**
- `ProcessingResult` for single image
- `BatchProcessingResult` for multiple images

### ProcessingResult

Result object for single image processing.

**Attributes:**
- `success` (bool): Whether processing was successful
- `mask` (np.ndarray): Binary segmentation mask
- `probability_map` (Optional[np.ndarray]): Raw probability map
- `visualization` (Optional[np.ndarray]): Visualization with overlay
- `num_ears` (int): Number of ears detected (0 or 1)
- `confidence` (float): Detection confidence score
- `processing_time` (float): Processing time in seconds
- `ear_percentage` (float): Percentage of image covered by ear
- `bounding_box` (Optional[Tuple[int, int, int, int]]): Ear bounding box (x, y, w, h)

### BatchProcessingResult

Result object for batch processing.

**Attributes:**
- `results` (List[ProcessingResult]): Individual results for each image
- `total_images` (int): Total number of images processed
- `successful_images` (int): Number of successfully processed images
- `failed_images` (int): Number of failed images
- `total_time` (float): Total processing time
- `average_time` (float): Average time per image
- `detection_rate` (float): Percentage of images with ears detected

## Examples

### Basic Usage

```python
from earsegmentationai import ImageProcessor

# Initialize processor
processor = ImageProcessor(device="cpu")

# Process single image
result = processor.process("path/to/image.jpg")
if result.success:
    print(f"Ear detected: {result.num_ears > 0}")
    print(f"Confidence: {result.confidence:.2f}")
```

### Batch Processing

```python
# Process directory
results = processor.process("path/to/images/")
print(f"Processed {results.total_images} images")
print(f"Detection rate: {results.detection_rate:.1f}%")

# Process list of images
image_list = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = processor.process(image_list)
```

### Advanced Usage

```python
# Process with all options
result = processor.process(
    "image.jpg",
    return_probability=True,
    return_visualization=True,
    save_results=True,
    output_dir="output/"
)

# Access probability map
if result.probability_map is not None:
    prob_map = result.probability_map
    
# Access visualization
if result.visualization is not None:
    cv2.imwrite("visualization.jpg", result.visualization)
```

### Error Handling

```python
try:
    result = processor.process("image.jpg")
    if not result.success:
        print("Processing failed")
except Exception as e:
    print(f"Error: {e}")
```