#!/usr/bin/env python3
import os
import sys
import tempfile
import fire
import whisper
import datetime
import logging
import ffmpeg
import time
import platform
import glob


def setup_logging(log_level):
    """
    Set up logging with the specified log level
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger("subtitles")


def format_timestamp(seconds):
    """
    Convert seconds to SRT timestamp format (HH:MM:SS,mmm)
    """
    milliseconds = int((seconds - int(seconds)) * 1000)
    time_obj = datetime.datetime.utcfromtimestamp(int(seconds))
    return f"{time_obj.strftime('%H:%M:%S')},{milliseconds:03d}"


def seconds_to_srt_timestamp(start_seconds, end_seconds):
    """
    Convert start and end times in seconds to SRT timestamp format
    """
    return f"{format_timestamp(start_seconds)} --> {format_timestamp(end_seconds)}"


def create_srt_file(segments, output_file, logger):
    """
    Create SRT file from whisper segments
    """
    logger.info(f"Creating SRT file with {len(segments)} segments")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, start=1):
            # Write segment index
            f.write(f"{i}\n")
            # Write timestamp
            timestamp = seconds_to_srt_timestamp(segment['start'], segment['end'])
            f.write(f"{timestamp}\n")
            # Write text
            f.write(f"{segment['text'].strip()}\n")
            # Write blank line between segments
            f.write("\n")
            
            logger.debug(f"Segment {i}: {timestamp} - {segment['text'].strip()}")
    
    logger.info(f"Subtitles saved to '{output_file}'")


def is_apple_silicon():
    """
    Detect if running on Apple Silicon (M1/M2/M3)
    """
    return platform.system() == "Darwin" and platform.machine().startswith(("arm", "aarch"))


def extract_subtitles(input_pattern, output_file=None, language="en", model_name="medium", log_level="INFO", use_gpu=None):
    """
    Extract subtitles from video files using ffmpeg and whisper.
    
    Args:
        input_pattern: Path or glob pattern for input video files (.avi, .mp4, etc.)
        output_file: Path to the output subtitle file (.srt). Only used when processing a single file.
        language: Language code for the subtitles (default: "en")
        model_name: Whisper model to use (tiny, base, small, medium, large) (default: "medium")
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) (default: "INFO")
        use_gpu: Whether to use GPU if available (True/False/None). If None, auto-detect Apple Silicon
    """
    # Start timing the execution
    start_time = time.time()
    
    logger = setup_logging(log_level)
    
    # Determine GPU usage
    if use_gpu is None:
        use_gpu = is_apple_silicon()
        if use_gpu:
            logger.info("Apple Silicon detected, enabling GPU acceleration")
    
    # Find all files matching the pattern
    input_files = glob.glob(input_pattern, recursive=True)
    
    if not input_files:
        logger.error(f"No files found matching pattern '{input_pattern}'")
        sys.exit(1)
    
    logger.info(f"Found {len(input_files)} file(s) matching pattern '{input_pattern}'")
    logger.info(f"Using Whisper model: {model_name}")
    logger.info(f"Language: {language}")
    logger.info(f"GPU acceleration: {'Enabled' if use_gpu else 'Disabled'}")
    
    # Check if output_file was specified with multiple input files
    if output_file is not None and len(input_files) > 1:
        logger.warning("Output file specified with multiple input files. Each file will use its own output name.")
        output_file = None
    
    # Load the model once for all files
    model_load_start = time.time()
    logger.info(f"Loading Whisper model '{model_name}'...")
    
    # Set device for GPU processing
    device = "cpu"  # Default to CPU for safety
    
    # Try to use GPU if requested
    if use_gpu:
        try:
            # Check for Apple Silicon first
            if is_apple_silicon():
                logger.info("Apple Silicon detected, will try MPS for GPU acceleration")
                # Due to MPS compatibility issues with some operations in Whisper,
                # we'll just stick with CPU for now
                logger.warning("MPS acceleration currently has compatibility issues with Whisper")
                logger.warning("Falling back to CPU for better compatibility")
                device = "cpu"
                
                # The following code is commented out due to compatibility issues
                # import torch
                # if torch.backends.mps.is_available():
                #    device = "mps"
                #    logger.info("Using Apple MPS (Metal Performance Shaders) for GPU acceleration")
            else:
                # Check for CUDA
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                    logger.info(f"Using CUDA GPU acceleration: {torch.cuda.get_device_name(0)}")
        except (ImportError, AttributeError, Exception) as e:
            logger.warning(f"GPU acceleration requested but not available: {e}")
            logger.warning("Falling back to CPU")
            device = "cpu"
    
    if device == "cpu":
        logger.info("Using CPU for processing")
    
    # Load the model with the specified device
    try:
        logger.info(f"Loading model on device: {device}")
        model = whisper.load_model(model_name, device=device)
        
        model_load_time = time.time() - model_load_start
        logger.debug(f"Model loaded in {model_load_time:.2f} seconds")
    except Exception as e:
        logger.warning(f"Failed to load model on {device}: {e}")
        logger.warning("Falling back to default device")
        model = whisper.load_model(model_name)
        
        model_load_time = time.time() - model_load_start
        logger.debug(f"Model loaded in {model_load_time:.2f} seconds")
    
    # Process each file
    successful_files = 0
    failed_files = 0
    
    for input_file in input_files:
        file_start_time = time.time()
        
        logger.info(f"\nProcessing file {successful_files + failed_files + 1}/{len(input_files)}: '{input_file}'")
        
        if not os.path.exists(input_file):
            logger.error(f"Input file '{input_file}' does not exist.")
            failed_files += 1
            continue
        
        # Determine output file name if not provided
        current_output_file = output_file
        if current_output_file is None:
            base, ext = os.path.splitext(input_file)
            current_output_file = f"{base}.srt"
            logger.info(f"Output file not specified, using '{current_output_file}'")
        
        # Create a temporary file for the extracted audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
            logger.debug(f"Created temporary audio file: {temp_audio_path}")
        
        try:
            # Extract audio using ffmpeg-python
            logger.info(f"Extracting audio from '{input_file}'...")
            try:
                logger.debug(f"Running FFmpeg to extract audio to {temp_audio_path}")
                
                # Use ffmpeg-python to extract audio
                (
                    ffmpeg
                    .input(input_file)
                    .output(
                        temp_audio_path,
                        format='wav',
                        acodec='pcm_s16le',
                        ac=1,
                        ar='16000'
                    )
                    .global_args('-y')  # Overwrite output files without asking
                    .global_args('-loglevel', 'error' if log_level.upper() in ["ERROR", "CRITICAL"] else 'info')
                    .run(capture_stdout=True, capture_stderr=True)
                )
                
                logger.debug("Audio extraction completed successfully")
            except ffmpeg.Error as e:
                logger.error(f"FFmpeg error: {e.stderr.decode('utf-8', errors='replace')}")
                failed_files += 1
                continue
            
            logger.info("Transcribing audio to subtitles...")
            # Always show progress bar unless log level is ERROR or higher
            show_progress = log_level.upper() not in ["ERROR", "CRITICAL"]
            transcribe_options = {
                "language": language,
                "verbose": show_progress,
                # Disable fp16 for CPU to avoid compatibility issues
                "fp16": device != "cpu"
            }
            logger.debug(f"Transcription options: {transcribe_options}")
            
            transcribe_start = time.time()
            result = model.transcribe(
                temp_audio_path,
                **transcribe_options
            )
            transcribe_time = time.time() - transcribe_start
            logger.debug(f"Transcription completed in {transcribe_time:.2f} seconds")
            
            logger.info("Transcription completed successfully")
            logger.debug(f"Number of segments in transcription: {len(result['segments'])}")
            
            # Create SRT file from transcription result
            create_srt_file(result["segments"], current_output_file, logger)
            successful_files += 1
            
            # Calculate and display the file execution time
            file_end_time = time.time()
            file_execution_time = file_end_time - file_start_time
            
            # Format the time nicely
            if file_execution_time < 60:
                time_str = f"{file_execution_time:.2f} seconds"
            elif file_execution_time < 3600:
                minutes = int(file_execution_time // 60)
                seconds = file_execution_time % 60
                time_str = f"{minutes} minutes and {seconds:.2f} seconds"
            else:
                hours = int(file_execution_time // 3600)
                minutes = int((file_execution_time % 3600) // 60)
                seconds = file_execution_time % 60
                time_str = f"{hours} hours, {minutes} minutes and {seconds:.2f} seconds"
            
            logger.info(f"File processed successfully in {time_str}")
        
        except Exception as e:
            logger.error(f"Error during subtitle extraction for '{input_file}': {e}", exc_info=True)
            failed_files += 1
        finally:
            # Clean up the temporary audio file
            if os.path.exists(temp_audio_path):
                logger.debug(f"Removing temporary audio file: {temp_audio_path}")
                os.unlink(temp_audio_path)
    
    # Calculate and display the total execution time
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Format the time nicely
    if execution_time < 60:
        time_str = f"{execution_time:.2f} seconds"
    elif execution_time < 3600:
        minutes = int(execution_time // 60)
        seconds = execution_time % 60
        time_str = f"{minutes} minutes and {seconds:.2f} seconds"
    else:
        hours = int(execution_time // 3600)
        minutes = int((execution_time % 3600) // 60)
        seconds = execution_time % 60
        time_str = f"{hours} hours, {minutes} minutes and {seconds:.2f} seconds"
    
    # Summary
    logger.info(f"\nSummary: Processed {len(input_files)} files in {time_str}")
    logger.info(f"  - Successful: {successful_files}")
    logger.info(f"  - Failed: {failed_files}")


if __name__ == "__main__":
    fire.Fire(extract_subtitles)
