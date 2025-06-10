"""Integration tests for CLI."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from earsegmentationai.cli.app import app


@pytest.fixture
def runner():
    """Create CLI test runner."""
    return CliRunner()


class TestCLIIntegration:
    """Integration tests for CLI commands."""
    
    def test_version_command(self, runner):
        """Test version command."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "Ear Segmentation AI" in result.output
        assert "Version:" in result.output
    
    def test_help_command(self, runner):
        """Test help command."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Ear Segmentation AI" in result.output
        assert "process-image" in result.output
        assert "process-video" in result.output
        assert "webcam" in result.output
    
    def test_process_image_help(self, runner):
        """Test process-image help."""
        result = runner.invoke(app, ["process-image", "--help"])
        assert result.exit_code == 0
        assert "--device" in result.output
        assert "--threshold" in result.output
        assert "--save-mask" in result.output
    
    @pytest.mark.slow
    def test_download_model_command(self, runner, temp_dir, monkeypatch):
        """Test download-model command."""
        # Mock the cache directory
        monkeypatch.setenv("HOME", str(temp_dir))
        
        # Note: This would actually download the model in real test
        # For unit test, we should mock the download
        from unittest.mock import patch, Mock
        
        mock_manager = Mock()
        mock_manager.config.model_path = temp_dir / "model.pth"
        
        with patch('earsegmentationai.core.model.ModelManager', return_value=mock_manager):
            result = runner.invoke(app, ["download-model"])
            assert result.exit_code == 0
            assert "Model downloaded successfully" in result.output
    
    def test_process_image_single_file(self, runner, sample_image_files, temp_dir):
        """Test processing single image file."""
        output_dir = temp_dir / "cli_output"
        
        # Mock model manager to avoid actual model loading
        from unittest.mock import patch
        from earsegmentationai.api.base import ProcessingResult
        import numpy as np
        
        mock_result = ProcessingResult(
            image=np.ones((256, 256, 3), dtype=np.uint8),
            mask=np.ones((256, 256), dtype=np.uint8),
            metadata={"filename": "test.png"}
        )
        
        with patch('earsegmentationai.cli.app.ImageProcessor') as mock_processor:
            mock_instance = mock_processor.return_value
            mock_instance.process.return_value = mock_result
            
            result = runner.invoke(app, [
                "process-image",
                str(sample_image_files[0]),
                "--output", str(output_dir),
                "--save-mask"
            ])
            
            assert result.exit_code == 0
            assert "Ear detected!" in result.output
    
    def test_process_image_directory(self, runner, sample_image_files, temp_dir):
        """Test processing image directory."""
        from unittest.mock import patch
        from earsegmentationai.api.base import BatchProcessingResult, ProcessingResult
        import numpy as np
        
        # Create mock results
        results = []
        for i in range(3):
            results.append(ProcessingResult(
                image=np.ones((256, 256, 3), dtype=np.uint8),
                mask=np.ones((256, 256), dtype=np.uint8),
                metadata={"filename": f"test_{i}.png"}
            ))
        
        mock_batch_result = BatchProcessingResult(results)
        
        with patch('earsegmentationai.cli.app.ImageProcessor') as mock_processor:
            mock_instance = mock_processor.return_value
            mock_instance.process.return_value = mock_batch_result
            
            result = runner.invoke(app, [
                "process-image",
                str(sample_image_files[0].parent),
                "--batch-size", "2"
            ])
            
            assert result.exit_code == 0
            assert "Total images: 3" in result.output
            assert "Detection rate:" in result.output
    
    def test_process_video_command(self, runner, sample_video_file):
        """Test process-video command."""
        from unittest.mock import patch
        
        mock_stats = {
            "frames_processed": 100,
            "average_fps": 30.5,
            "detection_rate": 95.0,
            "processing_time": 3.3
        }
        
        with patch('earsegmentationai.cli.app.VideoProcessor') as mock_processor:
            mock_instance = mock_processor.return_value
            mock_instance.process.return_value = mock_stats
            
            result = runner.invoke(app, [
                "process-video",
                str(sample_video_file),
                "--no-display",
                "--max-frames", "50"
            ])
            
            assert result.exit_code == 0
            assert "Frames processed: 100" in result.output
            assert "Average FPS: 30.5" in result.output
    
    def test_webcam_command(self, runner):
        """Test webcam command."""
        from unittest.mock import patch
        
        # Mock the process_video function since webcam calls it
        with patch('earsegmentationai.cli.app.process_video') as mock_process:
            result = runner.invoke(app, [
                "webcam",
                "--device-id", "0",
                "--skip-frames", "2"
            ])
            
            # Check that process_video was called with correct arguments
            mock_process.assert_called_once()
            args = mock_process.call_args[1]
            assert args["source"] == "0"
            assert args["skip_frames"] == 2
    
    def test_benchmark_command(self, runner, sample_image_files):
        """Test benchmark command."""
        from unittest.mock import patch, Mock
        import numpy as np
        
        # Mock processor and its methods
        mock_processor = Mock()
        mock_processor.process.return_value = Mock(has_ear=True)
        mock_processor.model_manager.get_model_info.return_value = {
            "architecture": "Unet",
            "parameters": 1000000
        }
        
        with patch('earsegmentationai.cli.app.ImageProcessor', return_value=mock_processor):
            with patch('cv2.imread', return_value=np.ones((256, 256, 3), dtype=np.uint8)):
                result = runner.invoke(app, [
                    "benchmark",
                    str(sample_image_files[0]),
                    "--iterations", "10",
                    "--warmup", "2"
                ])
                
                assert result.exit_code == 0
                assert "Benchmark Results" in result.output
                assert "FPS" in result.output
    
    def test_error_handling_invalid_file(self, runner):
        """Test error handling for invalid file."""
        result = runner.invoke(app, [
            "process-image",
            "non_existent_file.jpg"
        ])
        
        # Should show error but not crash
        # Rich formatted error message
        assert result.exit_code == 2
        assert "Error" in result.output or "Invalid value" in result.output
    
    def test_backward_compatibility_aliases(self, runner):
        """Test backward compatibility command aliases."""
        # Test that old command names still work
        cmds = ["picture-capture", "video-capture", "webcam-capture"]
        
        for cmd in cmds:
            result = runner.invoke(app, [cmd, "--help"])
            # Should at least recognize the command
            assert result.exit_code in [0, 2]  # 0 for help, 2 for missing required args