"""Main CLI application for Ear Segmentation AI."""

from pathlib import Path
from typing import Optional

import typer
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from earsegmentationai.__version__ import __version__
from earsegmentationai.api.image import ImageProcessor
from earsegmentationai.api.video import VideoProcessor
from earsegmentationai.utils.logging import setup_logging

# Create Typer app
app = typer.Typer(
    name="earsegmentationai",
    help="Ear Segmentation AI - Detect and segment ears in images and videos",
    add_completion=True,
)

# Create console for rich output
console = Console()

# Setup logging
logger = setup_logging()


@app.command()
def version():
    """Show version information."""
    print(
        Panel(
            f"[bold blue]Ear Segmentation AI[/bold blue]\n"
            f"Version: [green]{__version__}[/green]",
            title="Version Info",
            expand=False,
        )
    )


@app.command()
def process_image(
    path: Path = typer.Argument(
        ...,
        help="Path to image file or directory",
        exists=True,
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for results",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        "-d",
        help="Processing device (cpu, cuda:0)",
    ),
    threshold: float = typer.Option(
        0.5,
        "--threshold",
        "-t",
        help="Binary threshold for mask generation",
        min=0.0,
        max=1.0,
    ),
    save_mask: bool = typer.Option(
        False,
        "--save-mask",
        help="Save segmentation mask",
    ),
    save_visualization: bool = typer.Option(
        False,
        "--save-viz",
        help="Save visualization image",
    ),
    batch_size: int = typer.Option(
        1,
        "--batch-size",
        "-b",
        help="Batch size for processing multiple images",
        min=1,
    ),
):
    """Process image(s) for ear segmentation."""
    try:
        # Create processor
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task("Initializing model...", total=None)
            processor = ImageProcessor(
                device=device, threshold=threshold, batch_size=batch_size
            )

        # Configure output
        save_results = save_mask or save_visualization
        if save_results and output_dir is None:
            output_dir = Path.cwd() / "output"
            console.print(
                f"[yellow]No output directory specified, using: {output_dir}[/yellow]"
            )

        # Process
        console.print(f"[blue]Processing: {path}[/blue]")

        result = processor.process(
            path,
            return_visualization=save_visualization,
            save_results=save_results,
            output_dir=output_dir,
        )

        # Display results
        if hasattr(result, "__len__"):  # Batch result
            # Create summary table
            table = Table(title="Processing Results")
            table.add_column("File", style="cyan")
            table.add_column("Ear Detected", style="green")
            table.add_column("Area %", style="yellow")

            for r in result:
                filename = r.metadata.get("filename", "unknown")
                detected = "✓" if r.has_ear else "✗"
                area = f"{r.ear_percentage:.1f}" if r.has_ear else "-"
                table.add_row(filename, detected, area)

            console.print(table)

            # Print summary
            console.print("\n[bold]Summary:[/bold]")
            console.print(f"Total images: {len(result)}")
            console.print(
                f"Detection rate: [green]{result.detection_rate:.1f}%[/green]"
            )

        else:  # Single result
            # Display single result
            if result.has_ear:
                console.print("[green]✓ Ear detected![/green]")
                console.print(f"Area: {result.ear_percentage:.2f}% of image")

                bbox = result.get_bounding_box()
                if bbox:
                    console.print(
                        f"Bounding box: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}"
                    )
            else:
                console.print("[red]✗ No ear detected[/red]")

        if save_results:
            console.print(f"\n[green]Results saved to: {output_dir}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def process_video(
    source: str = typer.Argument(
        ...,
        help="Video source (file path, camera ID, or URL)",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output video file path",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        "-d",
        help="Processing device (cpu, cuda:0)",
    ),
    threshold: float = typer.Option(
        0.5,
        "--threshold",
        "-t",
        help="Binary threshold for mask generation",
        min=0.0,
        max=1.0,
    ),
    skip_frames: int = typer.Option(
        0,
        "--skip-frames",
        "-s",
        help="Number of frames to skip between predictions",
        min=0,
    ),
    no_display: bool = typer.Option(
        False,
        "--no-display",
        help="Don't display video window",
    ),
    max_frames: Optional[int] = typer.Option(
        None,
        "--max-frames",
        "-m",
        help="Maximum number of frames to process",
        min=1,
    ),
):
    """Process video stream for ear segmentation."""
    try:
        # Parse source
        if source.isdigit():
            source = int(source)
            source_type = "camera"
        elif source.startswith(("http://", "https://", "rtsp://")):
            source_type = "stream"
        else:
            source = Path(source)
            if not source.exists():
                console.print(f"[red]Video file not found: {source}[/red]")
                raise typer.Exit(1)
            source_type = "file"

        # Create processor
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task("Initializing model...", total=None)
            processor = VideoProcessor(
                device=device,
                threshold=threshold,
                skip_frames=skip_frames,
            )

        # Process video
        console.print(f"[blue]Processing {source_type}: {source}[/blue]")
        if not no_display:
            console.print("[yellow]Press 'q' to quit[/yellow]")

        stats = processor.process(
            source,
            output_path=output,
            display=not no_display,
            max_frames=max_frames,
        )

        # Display statistics
        console.print("\n[bold]Processing Statistics:[/bold]")
        console.print(f"Frames processed: {stats['frames_processed']}")
        console.print(
            f"Average FPS: [green]{stats['average_fps']:.1f}[/green]"
        )
        console.print(
            f"Detection rate: [green]{stats['detection_rate']:.1f}%[/green]"
        )
        console.print(f"Processing time: {stats['processing_time']:.1f}s")

        if output:
            console.print(f"\n[green]Output saved to: {output}[/green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Processing interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def webcam(
    device_id: int = typer.Option(
        0,
        "--device-id",
        "-i",
        help="Camera device ID",
        min=0,
    ),
    processing_device: str = typer.Option(
        "cpu",
        "--device",
        "-d",
        help="Processing device (cpu, cuda:0)",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output video file path",
    ),
    threshold: float = typer.Option(
        0.5,
        "--threshold",
        "-t",
        help="Binary threshold for mask generation",
        min=0.0,
        max=1.0,
    ),
    skip_frames: int = typer.Option(
        1,
        "--skip-frames",
        "-s",
        help="Number of frames to skip (for performance)",
        min=0,
    ),
):
    """Real-time ear segmentation from webcam."""
    console.print(f"[blue]Starting webcam capture (Camera {device_id})[/blue]")
    console.print("[yellow]Press 'q' to quit[/yellow]")

    # Use process_video command with camera ID
    process_video(
        source=str(device_id),
        output=output,
        device=processing_device,
        threshold=threshold,
        skip_frames=skip_frames,
        no_display=False,
        max_frames=None,
    )


@app.command()
def benchmark(
    image_path: Path = typer.Argument(
        ...,
        help="Path to test image",
        exists=True,
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        "-d",
        help="Processing device (cpu, cuda:0)",
    ),
    iterations: int = typer.Option(
        100,
        "--iterations",
        "-n",
        help="Number of iterations",
        min=1,
    ),
    warmup: int = typer.Option(
        10,
        "--warmup",
        "-w",
        help="Number of warmup iterations",
        min=0,
    ),
):
    """Benchmark model performance."""
    import time

    import numpy as np

    try:
        # Create processor
        console.print(f"[blue]Initializing model on {device}...[/blue]")
        processor = ImageProcessor(device=device)

        # Load image
        import cv2

        image = cv2.imread(str(image_path))
        if image is None:
            console.print(f"[red]Failed to load image: {image_path}[/red]")
            raise typer.Exit(1)

        console.print(f"Image shape: {image.shape}")

        # Warmup
        if warmup > 0:
            console.print(
                f"[yellow]Warming up ({warmup} iterations)...[/yellow]"
            )
            for _ in range(warmup):
                processor.process(image)

        # Benchmark
        console.print(
            f"[blue]Running benchmark ({iterations} iterations)...[/blue]"
        )

        times = []
        with Progress() as progress:
            task = progress.add_task("Benchmarking...", total=iterations)

            for _ in range(iterations):
                start_time = time.time()
                processor.process(image)
                elapsed = time.time() - start_time
                times.append(elapsed)
                progress.update(task, advance=1)

        # Calculate statistics
        times = np.array(times)
        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        fps = 1.0 / mean_time

        # Display results
        table = Table(title="Benchmark Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Device", device)
        table.add_row("Image Size", f"{image.shape[1]}x{image.shape[0]}")
        table.add_row("Iterations", str(iterations))
        table.add_row("Mean Time", f"{mean_time*1000:.2f} ms")
        table.add_row("Std Dev", f"{std_time*1000:.2f} ms")
        table.add_row("Min Time", f"{min_time*1000:.2f} ms")
        table.add_row("Max Time", f"{max_time*1000:.2f} ms")
        table.add_row("FPS", f"{fps:.1f}")

        console.print(table)

        # Model info
        model_info = processor.model_manager.get_model_info()
        console.print("\n[bold]Model Info:[/bold]")
        console.print(f"Architecture: {model_info['architecture']}")
        console.print(f"Parameters: {model_info.get('parameters', 'N/A'):,}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def download_model(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-download even if model exists",
    ),
):
    """Download the ear segmentation model."""
    try:
        from earsegmentationai.core.model import ModelManager

        console.print("[blue]Downloading ear segmentation model...[/blue]")

        manager = ModelManager()
        manager.load_model(force_download=force)

        console.print("[green]✓ Model downloaded successfully![/green]")
        console.print(f"Model path: {manager.config.model_path}")

    except Exception as e:
        console.print(f"[red]Error downloading model: {e}[/red]")
        raise typer.Exit(1)


# Backward compatibility aliases
picture_capture = process_image
video_capture = process_video
webcam_capture = webcam

# Register legacy commands
app.command("picture-capture")(picture_capture)
app.command("video-capture")(video_capture)
app.command("webcam-capture")(webcam_capture)


if __name__ == "__main__":
    app()
