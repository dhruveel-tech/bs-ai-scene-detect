"""
Advanced Scene Detection and Subdivision System
Production-ready version with comprehensive error handling and logging
"""

import subprocess
import csv
import os
import sys
import time
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from pathlib import Path
import re

import cv2
import numpy as np
import torch
import shutil
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# ============================================================================
# CONFIGURATION
# ============================================================================

# File paths
VIDEO_PATH = r"D:\SDNA\Scene_Detector\op_using_detect_content\ind_vs_pak\subclip\India vs Pakistan t20 subclip.mp4"
OUTPUT_DIR = r"D:\SDNA\Scene_Detector\final_op\ind_pak"
LOG_FILE = os.path.join(OUTPUT_DIR, "processing_log.txt")
TIME_FILE = os.path.join(OUTPUT_DIR, "processing_time.txt")

# Detection parameters
EXTRACT_INTERVAL_SECONDS = 1.5  # Frame extraction interval
MIN_IMAGES_PER_SCENE = 3  # Minimum images to extract per scene
MAX_IMAGES_PER_SCENE = 30  # Maximum images to extract per scene
SIMILARITY_THRESHOLD = 0.85  # Histogram similarity threshold
STRUCTURAL_THRESHOLD = 0.85  # SSIM threshold
EDGE_THRESHOLD = 0.75  # Edge detection threshold
MIN_SUBSCENE_DURATION = 2.5  # Minimum duration for a sub-scene (seconds)
CLIP_SIMILARITY_THRESHOLD = 0.90  # CLIP model similarity threshold

# Video encoding settings
VIDEO_CODEC = "libx264"
VIDEO_PRESET = "fast"
VIDEO_CRF = "18"
AUDIO_CODEC = "aac"
AUDIO_BITRATE = "128k"

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(output_dir: str) -> logging.Logger:
    """
    Configure logging for the application with both file and console handlers.
    
    Args:
        output_dir: Directory where log file will be stored
        
    Returns:
        Configured logger instance
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create logger
        logger = logging.getLogger("SceneDetector")
        logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # File handler - detailed logs
        file_handler = logging.FileHandler(
            os.path.join(output_dir, "processing_log.txt"),
            mode='w',
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(funcName)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        # Console handler - important messages only
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
        
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        raise

# Initialize logger
logger = setup_logging(OUTPUT_DIR)

# ============================================================================
# CLIP MODEL SETUP
# ============================================================================

def load_clip_model() -> Tuple[Optional[CLIPModel], Optional[CLIPProcessor]]:
    """
    Load CLIP model and processor for semantic scene detection.
    
    Returns:
        Tuple of (model, processor) or (None, None) if loading fails
    """
    try:
        logger.info("Loading CLIP model...")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        logger.info("CLIP model loaded successfully")
        return model, processor
    except Exception as e:
        logger.error(f"Failed to load CLIP model: {e}")
        logger.warning("Continuing without CLIP-based detection")
        return None, None

# Load CLIP model
CLIP_MODEL, CLIP_PROCESSOR = load_clip_model()

# ============================================================================
# CLIP FEATURE EXTRACTION
# ============================================================================

def get_features(img_path: str) -> Optional[torch.Tensor]:
    """
    Extract CLIP features from an image.
    
    Args:
        img_path: Path to the image file
        
    Returns:
        Normalized feature tensor or None if extraction fails
    """
    try:
        if CLIP_MODEL is None or CLIP_PROCESSOR is None:
            return None
            
        image = Image.open(img_path).convert("RGB")
        inputs = CLIP_PROCESSOR(images=image, return_tensors="pt")
        
        with torch.no_grad():
            features = CLIP_MODEL.get_image_features(**inputs)
        
        # Normalize features
        return features[0] / features[0].norm()
        
    except Exception as e:
        logger.error(f"Failed to extract features from {img_path}: {e}")
        return None

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Calculate cosine similarity between two tensors.
    
    Args:
        a: First tensor
        b: Second tensor
        
    Returns:
        Cosine similarity score
    """
    try:
        return torch.dot(a, b).item()
    except Exception as e:
        logger.error(f"Failed to calculate cosine similarity: {e}")
        return 0.0

# ============================================================================
# SCENE DETECTION WITH SCENEDETECT
# ============================================================================

def run_scenedetect(video_path: str, output_dir: str) -> Optional[str]:
    """
    Run PySceneDetect to split video and generate scene CSV.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory for output files
        
    Returns:
        Path to images directory or None if operation fails
    """
    try:
        images_dir = os.path.join(output_dir, "scenes_images")
        video_dir = os.path.join(output_dir, "scenes_videos")

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)

        logger.info("Detecting scenes and splitting video...")
        
        # Split video into scenes
        result = subprocess.run(
            [
                "scenedetect", "-i", video_path,
                "--output", video_dir,
                "detect-hash",
                "split-video"
            ],
            check=True,
            capture_output=True,
            text=True
        )
        logger.debug(f"Scene split output: {result.stdout}")

        logger.info("Generating scene list CSV...")
        
        # Generate scene list CSV
        result = subprocess.run(
            [
                "scenedetect", "-i", video_path,
                "--output", images_dir,
                "detect-hash",
                "list-scenes"
            ],
            check=True,
            capture_output=True,
            text=True
        )
        logger.debug(f"Scene list output: {result.stdout}")

        return images_dir
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Scenedetect command failed: {e.stderr}")
        return None
    except FileNotFoundError:
        logger.error("Scenedetect not found. Please install: pip install scenedetect[opencv]")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in run_scenedetect: {e}")
        return None

# ============================================================================
# IMAGE SIMILARITY CALCULATIONS
# ============================================================================

def calculate_histogram_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate color histogram similarity using correlation method.
    
    Args:
        img1: First image array (BGR)
        img2: Second image array (BGR)
        
    Returns:
        Similarity score between 0 and 1
    """
    try:
        # Convert to HSV color space for better color comparison
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        
        # Calculate 3D histograms
        hist1 = cv2.calcHist([hsv1], [0, 1, 2], None, [8, 8, 8], 
                            [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1, 2], None, [8, 8, 8], 
                            [0, 256, 0, 256, 0, 256])
        
        # Normalize histograms
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        # Compare using correlation method
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
        
    except Exception as e:
        logger.error(f"Failed to calculate histogram similarity: {e}")
        return 0.0

def calculate_structural_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate structural similarity using PSNR-based metric.
    Detects layout and composition changes.
    
    Args:
        img1: First image array (BGR)
        img2: Second image array (BGR)
        
    Returns:
        Similarity score between 0 and 1
    """
    try:
        # Resize for faster computation
        img1_small = cv2.resize(img1, (320, 240))
        img2_small = cv2.resize(img2, (320, 240))
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1_small, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2_small, cv2.COLOR_BGR2GRAY)
        
        # Calculate mean squared error
        mse = np.mean((gray1.astype(float) - gray2.astype(float)) ** 2)
        
        if mse == 0:
            return 1.0
        
        # Convert to PSNR and normalize to similarity score
        max_pixel_value = 255.0
        psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
        similarity = min(1.0, psnr / 50.0)  # Normalize to [0, 1]
        
        return similarity
        
    except Exception as e:
        logger.error(f"Failed to calculate structural similarity: {e}")
        return 0.0

def calculate_edge_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate edge detection similarity using Canny edge detector.
    Detects composition and object boundary changes.
    
    Args:
        img1: First image array (BGR)
        img2: Second image array (BGR)
        
    Returns:
        Similarity score between 0 and 1 (IoU of edges)
    """
    try:
        # Resize for faster computation
        img1_small = cv2.resize(img1, (320, 240))
        img2_small = cv2.resize(img2, (320, 240))
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1_small, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2_small, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection
        edges1 = cv2.Canny(gray1, 100, 200)
        edges2 = cv2.Canny(gray2, 100, 200)
        
        # Calculate intersection over union of edges
        intersection = np.logical_and(edges1, edges2).sum()
        union = np.logical_or(edges1, edges2).sum()
        
        if union == 0:
            return 1.0
        
        return intersection / union
        
    except Exception as e:
        logger.error(f"Failed to calculate edge similarity: {e}")
        return 0.0

# ============================================================================
# ADVANCED SUBSCENE DETECTION
# ============================================================================

def detect_subscenes_advanced(
    frames_data: List[Dict], 
    min_duration: float = MIN_SUBSCENE_DURATION
) -> List[Tuple[int, int]]:
    """
    Advanced sub-scene detection using multiple similarity metrics.
    Combines histogram, structural, and edge similarity with look-ahead confirmation.
    
    Args:
        frames_data: List of frame dictionaries with 'image', 'timestamp', etc.
        min_duration: Minimum duration for a valid sub-scene (seconds)
        
    Returns:
        List of (start_index, end_index) tuples for each sub-scene
    """
    try:
        if len(frames_data) < 2:
            logger.warning("Not enough frames for subscene detection")
            return [(0, len(frames_data) - 1)]
        
        logger.info(f"Analyzing {len(frames_data)} frames for sub-scene detection...")
        
        # Calculate all similarity metrics between consecutive frames
        similarities = []
        for i in range(1, len(frames_data)):
            try:
                prev_img = frames_data[i-1]['image']
                curr_img = frames_data[i]['image']
                
                hist_sim = calculate_histogram_similarity(prev_img, curr_img)
                struct_sim = calculate_structural_similarity(prev_img, curr_img)
                edge_sim = calculate_edge_similarity(prev_img, curr_img)
                
                # Weighted average (histogram is most important for scene changes)
                combined_sim = (hist_sim * 0.5) + (struct_sim * 0.3) + (edge_sim * 0.2)
                
                similarities.append({
                    'index': i,
                    'histogram': hist_sim,
                    'structural': struct_sim,
                    'edge': edge_sim,
                    'combined': combined_sim,
                    'timestamp': frames_data[i]['timestamp']
                })
                
            except Exception as e:
                logger.error(f"Error calculating similarity for frame {i}: {e}")
                # Use high similarity as fallback to avoid false breaks
                similarities.append({
                    'index': i,
                    'histogram': 0.95,
                    'structural': 0.95,
                    'edge': 0.95,
                    'combined': 0.95,
                    'timestamp': frames_data[i]['timestamp']
                })
        
        # Find scene breaks with look-ahead confirmation
        subscenes = []
        current_start = 0
        i = 0
        
        while i < len(similarities):
            sim = similarities[i]
            frame_idx = sim['index']
            
            # Check if metrics indicate a significant change
            is_histogram_change = sim['histogram'] < SIMILARITY_THRESHOLD
            is_structural_change = sim['structural'] < STRUCTURAL_THRESHOLD
            is_edge_change = sim['edge'] < EDGE_THRESHOLD
            is_combined_change = sim['combined'] < 0.78
            
            # Vote-based system: need 2/3 metrics to agree or combined score low
            change_votes = sum([is_histogram_change, is_structural_change, is_edge_change])
            
            if (change_votes >= 2) or is_combined_change:
                # Check if current sub-scene meets minimum duration
                duration = frames_data[frame_idx - 1]['timestamp'] - frames_data[current_start]['timestamp']
                
                if duration >= min_duration:
                    # Look ahead to confirm sustained change (not just a flash/transition)
                    confirmed = True
                    if i + 2 < len(similarities):
                        try:
                            future_img = frames_data[min(frame_idx + 2, len(frames_data) - 1)]['image']
                            current_img = frames_data[frame_idx]['image']
                            
                            future_hist_sim = calculate_histogram_similarity(current_img, future_img)
                            
                            # If future frames are also different, might be transition effect
                            if future_hist_sim < 0.70:
                                confirmed = False
                                logger.debug(f"Potential transition at {sim['timestamp']:.2f}s - not confirmed")
                        except Exception as e:
                            logger.error(f"Error in look-ahead confirmation: {e}")
                    
                    if confirmed:
                        # Create sub-scene break
                        subscenes.append((current_start, frame_idx - 1))
                        logger.info(
                            f"Sub-scene break at {sim['timestamp']:.2f}s | "
                            f"Hist:{sim['histogram']:.2f} Struct:{sim['structural']:.2f} "
                            f"Edge:{sim['edge']:.2f} Combined:{sim['combined']:.2f}"
                        )
                        current_start = frame_idx
            
            i += 1
        
        # Add the final sub-scene
        subscenes.append((current_start, len(frames_data) - 1))
        
        # Post-process: merge very short sub-scenes with neighbors
        merged_subscenes = []
        for i, (start, end) in enumerate(subscenes):
            duration = frames_data[end]['timestamp'] - frames_data[start]['timestamp']
            
            if duration < min_duration and len(merged_subscenes) > 0:
                # Merge with previous sub-scene
                prev_start, prev_end = merged_subscenes[-1]
                merged_subscenes[-1] = (prev_start, end)
                logger.info(f"Merged short sub-scene ({duration:.1f}s) with previous")
            else:
                merged_subscenes.append((start, end))
        
        final_subscenes = merged_subscenes if merged_subscenes else subscenes
        logger.info(f"Detected {len(final_subscenes)} sub-scenes")
        
        return final_subscenes
        
    except Exception as e:
        logger.error(f"Critical error in detect_subscenes_advanced: {e}")
        # Return single scene as fallback
        return [(0, len(frames_data) - 1)]

# ============================================================================
# IMAGE EXTRACTION AND ANALYSIS
# ============================================================================

def extract_and_analyze_images(
    video_path: str, 
    output_dir: str, 
    interval_seconds: float = EXTRACT_INTERVAL_SECONDS
) -> List[Dict]:
    """
    Extract images from video and detect sub-scenes with advanced analysis.
    
    Args:
        video_path: Path to input video
        output_dir: Output directory for images
        interval_seconds: Time interval between extracted frames
        
    Returns:
        List of sub-scene information dictionaries
    """
    try:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        images_dir = os.path.join(output_dir, "scenes_images")
        subscenes_dir = os.path.join(output_dir, "subscenes_videos")
        
        os.makedirs(subscenes_dir, exist_ok=True)
        
        csv_path = os.path.join(images_dir, f"{base_name}-Scenes.csv")
        
        # Validate CSV exists
        if not os.path.exists(csv_path):
            logger.error(f"Scene CSV not found: {csv_path}")
            return []
        
        # Read scene CSV
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if len(lines) < 2:
                    logger.error("Scene CSV is empty or malformed")
                    return []
                    
                cleaned_csv = lines[1:]  # Skip header comment line
                reader = csv.DictReader(cleaned_csv)
                scenes = list(reader)
                
            logger.info(f"Found {len(scenes)} scenes in CSV")
            
        except Exception as e:
            logger.error(f"Failed to read scene CSV: {e}")
            return []
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video properties: {fps:.2f} FPS, {total_frames} total frames")
        logger.info(f"Extracting frames every {interval_seconds}s")
        
        all_subscenes_info = []
        
        # Process each scene
        for scene_idx, scene in enumerate(scenes, start=1):
            try:
                start_frame = int(scene["Start Frame"])
                end_frame = int(scene["End Frame"])
                start_time = float(scene["Start Time (seconds)"])
                end_time = float(scene["End Time (seconds)"])
                scene_duration = float(scene["Length (seconds)"])
                
                logger.info(f"Processing Scene {scene_idx:03d}: {scene_duration:.2f}s")
                
                # Calculate number of images to extract
                num_images = max(
                    MIN_IMAGES_PER_SCENE, 
                    min(MAX_IMAGES_PER_SCENE, int(scene_duration / interval_seconds) + 1)
                )
                
                # Extract frames from this scene
                frames_data = []
                current_time = start_time
                img_idx = 1
                
                while current_time <= end_time and len(frames_data) < num_images:
                    frame_num = int(start_frame + ((current_time - start_time) * fps))
                    
                    # Ensure frame number is valid
                    if frame_num >= total_frames:
                        logger.warning(f"Frame {frame_num} exceeds video length")
                        break
                    
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, frame = cap.read()
                    
                    if ret and frame is not None:
                        frames_data.append({
                            'image': frame.copy(),
                            'frame_num': frame_num,
                            'timestamp': current_time,
                            'index': img_idx
                        })
                        img_idx += 1
                    else:
                        logger.warning(f"Failed to read frame {frame_num}")
                    
                    current_time += interval_seconds
                
                logger.info(f"Extracted {len(frames_data)} frames from scene {scene_idx}")
                
                # Detect sub-scenes if we have enough frames
                if len(frames_data) > 0:
                    subscene_boundaries = detect_subscenes_advanced(frames_data, MIN_SUBSCENE_DURATION)
                    
                    logger.info(f"Scene {scene_idx} divided into {len(subscene_boundaries)} sub-scenes")
                    
                    # Save images and info for each sub-scene
                    for sub_idx, (start_idx, end_idx) in enumerate(subscene_boundaries, start=1):
                        try:
                            sub_start_time = frames_data[start_idx]['timestamp']
                            sub_end_time = frames_data[end_idx]['timestamp']
                            sub_duration = sub_end_time - sub_start_time
                            
                            logger.info(
                                f"  Sub-scene {sub_idx}: {sub_start_time:.1f}s → {sub_end_time:.1f}s "
                                f"({sub_duration:.1f}s, {end_idx-start_idx+1} frames)"
                            )
                            
                            # Save images for this sub-scene
                            for i in range(start_idx, end_idx + 1):
                                frame_data = frames_data[i]
                                img_name = f"{base_name}-Scene-{scene_idx:03d}-Sub-{sub_idx:02d}-{i-start_idx+1:02d}.jpg"
                                img_path = os.path.join(images_dir, img_name)
                                
                                cv2.imwrite(img_path, frame_data['image'])
                            
                            # Store sub-scene info
                            all_subscenes_info.append({
                                'scene_num': scene_idx,
                                'subscene_num': sub_idx,
                                'start_time': sub_start_time,
                                'end_time': sub_end_time,
                                'duration': sub_duration,
                                'num_images': end_idx - start_idx + 1,
                                'frames': frames_data[start_idx:end_idx + 1]
                            })
                            
                        except Exception as e:
                            logger.error(f"Failed to process sub-scene {sub_idx} of scene {scene_idx}: {e}")
                
            except Exception as e:
                logger.error(f"Failed to process scene {scene_idx}: {e}")
                continue
        
        cap.release()
        
        logger.info(f"Analyzed {len(scenes)} scenes into {len(all_subscenes_info)} sub-scenes")
        return all_subscenes_info
        
    except Exception as e:
        logger.error(f"Critical error in extract_and_analyze_images: {e}")
        return []

# ============================================================================
# VIDEO CREATION
# ============================================================================

def create_subscene_videos(
    video_path: str, 
    output_dir: str, 
    base_name: str, 
    subscenes_info: List[Dict]
) -> bool:
    """
    Create sub-scene video clips from original video using FFmpeg.
    
    Args:
        video_path: Path to original video
        output_dir: Output directory
        base_name: Base name for output files
        subscenes_info: List of sub-scene information
        
    Returns:
        True if all videos created successfully, False otherwise
    """
    try:
        subscenes_dir = os.path.join(output_dir, "subscenes_videos")
        os.makedirs(subscenes_dir, exist_ok=True)
        
        success_count = 0
        total_count = len(subscenes_info)
        
        logger.info(f"Creating {total_count} sub-scene videos...")

        for sub in subscenes_info:
            try:
                scene_num = sub["scene_num"]
                subscene_num = sub["subscene_num"]
                start_time = sub["start_time"]
                end_time = sub["end_time"]
                duration = end_time - start_time

                output_video = os.path.join(
                    subscenes_dir,
                    f"{base_name}-Scene-{scene_num:03d}-Sub-{subscene_num:02d}.mp4"
                )

                # FFmpeg command for accurate video cutting with re-encoding
                cmd = [
                    "ffmpeg",
                    "-y",  # Overwrite output files
                    "-ss", f"{start_time}",  # Seek to start time
                    "-i", video_path,  # Input file
                    "-t", f"{duration}",  # Duration
                    "-c:v", VIDEO_CODEC,  # Video codec
                    "-preset", VIDEO_PRESET,  # Encoding preset
                    "-crf", VIDEO_CRF,  # Quality level
                    "-c:a", AUDIO_CODEC,  # Audio codec
                    "-b:a", AUDIO_BITRATE,  # Audio bitrate
                    output_video
                ]

                logger.debug(f"Creating: {os.path.basename(output_video)}")
                
                result = subprocess.run(
                    cmd, 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=300  # 5 minute timeout per video
                )
                
                if result.returncode == 0 and os.path.exists(output_video):
                    success_count += 1
                else:
                    logger.error(f"FFmpeg failed for {output_video}: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                logger.error(f"FFmpeg timeout for scene {scene_num} sub {subscene_num}")
            except Exception as e:
                logger.error(f"Failed to create video for scene {scene_num} sub {subscene_num}: {e}")
        
        logger.info(f"Created {success_count}/{total_count} sub-scene videos")
        return success_count == total_count
        
    except Exception as e:
        logger.error(f"Critical error in create_subscene_videos: {e}")
        return False

# ============================================================================
# FILE RENAMING
# ============================================================================

def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to HH-MM-SS.mmm timestamp format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted timestamp string
    """
    try:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        millisecs = int((secs - int(secs)) * 1000)
        return f"{hours:02d}-{minutes:02d}-{int(secs):02d}.{millisecs:03d}"
    except Exception as e:
        logger.error(f"Failed to format timestamp {seconds}: {e}")
        return "00-00-00.000"

def rename_subscene_files(
    output_dir: str, 
    base_name: str, 
    subscenes_info: List[Dict]
) -> bool:
    """
    Rename sub-scene files with timestamps for better organization.
    
    Args:
        output_dir: Output directory
        base_name: Base name for files
        subscenes_info: List of sub-scene information
        
    Returns:
        True if all renames successful, False otherwise
    """
    try:
        images_dir = os.path.join(output_dir, "scenes_images")
        subscenes_dir = os.path.join(output_dir, "subscenes_videos")
        
        logger.info("Renaming files with timestamps...")
        
        success_count = 0
        total_operations = 0
        
        for sub_info in subscenes_info:
            try:
                scene_num = sub_info['scene_num']
                subscene_num = sub_info['subscene_num']
                start_tc = format_timestamp(sub_info['start_time'])
                end_tc = format_timestamp(sub_info['end_time'])
                
                # Rename video file
                # old_video = os.path.join(
                #     subscenes_dir,
                #     f"{base_name}-Scene-{scene_num:03d}-Sub-{subscene_num:02d}.mp4"
                # )
                # new_video = os.path.join(
                #     subscenes_dir,
                #     f"{base_name}-Scene-{scene_num:03d}-Sub-{subscene_num:02d}_{start_tc}_to_{end_tc}.mp4"
                # )
                
                # total_operations += 1
                # if os.path.exists(old_video):
                #     try:
                #         os.rename(old_video, new_video)
                #         success_count += 1
                #         logger.debug(f"Renamed video: {os.path.basename(new_video)}")
                #     except Exception as e:
                #         logger.error(f"Failed to rename video {old_video}: {e}")
                # else:
                #     logger.warning(f"Video file not found: {old_video}")
                
                # Rename image files
                for i, frame_data in enumerate(sub_info['frames'], start=1):
                    timestamp = format_timestamp(frame_data['timestamp'])
                    old_img = os.path.join(
                        images_dir,
                        f"{base_name}-Scene-{scene_num:03d}-Sub-{subscene_num:02d}-{i:02d}.jpg"
                    )
                    new_img = os.path.join(
                        images_dir,
                        f"{base_name}-Scene-{scene_num:03d}-Sub-{subscene_num:02d}_{timestamp}.jpg"
                    )
                    
                    total_operations += 1
                    if os.path.exists(old_img):
                        try:
                            os.rename(old_img, new_img)
                            success_count += 1
                        except Exception as e:
                            logger.error(f"Failed to rename image {old_img}: {e}")
                    else:
                        logger.warning(f"Image file not found: {old_img}")
                        
            except Exception as e:
                logger.error(f"Failed to rename files for scene {sub_info.get('scene_num')} sub {sub_info.get('subscene_num')}: {e}")
        
        logger.info(f"Renamed {success_count}/{total_operations} files")
        return success_count == total_operations
        
    except Exception as e:
        logger.error(f"Critical error in rename_subscene_files: {e}")
        return False

# ============================================================================
# CLIP-BASED SUBSCENE DETECTION
# ============================================================================

def detect_subscenes_with_clip(images_dir: str) -> Optional[Dict[int, List[str]]]:
    """
    Detect sub-scenes using CLIP model for semantic similarity.
    
    Args:
        images_dir: Directory containing extracted images
        
    Returns:
        Dictionary mapping subscene IDs to lists of image paths, or None if failed
    """
    try:
        if CLIP_MODEL is None or CLIP_PROCESSOR is None:
            logger.warning("CLIP model not available, skipping CLIP-based detection")
            return None
        
        # Get all image paths and sort them
        image_paths = sorted([
            os.path.join(images_dir, f) 
            for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])

        if not image_paths:
            logger.error("No images found in directory")
            return None

        logger.info(f"Processing {len(image_paths)} images with CLIP model...")

        # Extract features for all images
        embeddings = []
        for i, img_path in enumerate(image_paths):
            if i % 10 == 0:
                logger.debug(f"Extracting features: {i}/{len(image_paths)}")
            
            features = get_features(img_path)
            if features is None:
                logger.error(f"Failed to extract features from {img_path}")
                return None
            embeddings.append(features)

        # Detect scene boundaries based on similarity
        subscene_id = 0
        subscenes = {subscene_id: [image_paths[0]]}

        for i in range(1, len(embeddings)):
            sim = cosine_similarity(embeddings[i-1], embeddings[i])
            logger.debug(f"CLIP similarity {i}: {sim:.4f}")

            # If similarity drops below threshold, start new subscene
            if sim < CLIP_SIMILARITY_THRESHOLD: 
                subscene_id += 1
                subscenes[subscene_id] = []
                logger.info(f"New sub-scene detected at image {i} (similarity: {sim:.4f})")

            subscenes[subscene_id].append(image_paths[i])

        logger.info(f"CLIP detected {len(subscenes)} sub-scenes")
        return subscenes
        
    except Exception as e:
        logger.error(f"Failed in detect_subscenes_with_clip: {e}")
        return None

def save_subscenes(subscenes, output_dir):
    output = os.path.join(output_dir, "final_scene")

    if os.path.exists(output):
        shutil.rmtree(output)
    os.makedirs(output)

    global_subscene_index = {}  # tracks subscene numbering per main scene

    for sid, imgs in subscenes.items():
        folder = os.path.join(output, f"subscene_{sid+1}")
        os.makedirs(folder, exist_ok=True)

        # Extract main scene number from first image
        first_base = os.path.basename(imgs[0])
        try:
            main_scene = first_base.split("Scene-")[1].split("-Sub")[0]
        except:
            main_scene = f"{sid+1:03d}"

        # If this is the first folder for this main scene → start at Sub-01
        if main_scene not in global_subscene_index:
            global_subscene_index[main_scene] = 1
        else:
            global_subscene_index[main_scene] += 1

        subscene_id = global_subscene_index[main_scene]
        sub_str = f"{subscene_id:02d}"

        # Process images
        for img in imgs:
            base = os.path.basename(img)
            timestamp = base.rsplit("_", 1)[-1]  # keep original timestamp

            new_name = f"Scene-{main_scene}_Sub-{sub_str}_{timestamp}"
            shutil.copy(img, os.path.join(folder, new_name))

    print("\n✔ Final Sub-Scenes Saved →", output)
    return output

# ============================================================================
# FILE ORGANIZATION
# ============================================================================

def move_files_and_remove_subfolders(parent_folder: str) -> bool:
    """
    Move all files from subfolders to parent folder and remove empty subfolders.
    Skips duplicate files to avoid overwriting.
    
    Args:
        parent_folder: Parent directory to consolidate files into
        
    Returns:
        True if operation completed successfully, False otherwise
    """
    try:
        if not os.path.exists(parent_folder):
            logger.error(f"Parent folder does not exist: {parent_folder}")
            return False
        
        logger.info("Moving files from subfolders to parent directory...")
        
        moved_count = 0
        skipped_count = 0
        
        # Move files from subfolders to parent
        for root, dirs, files in os.walk(parent_folder):
            # Skip the parent folder itself
            if root == parent_folder:
                continue

            for file in files:
                try:
                    src = os.path.join(root, file)
                    dst = os.path.join(parent_folder, file)

                    # Skip if file already exists in destination
                    if os.path.exists(dst):
                        logger.debug(f"Skipped (already exists): {file}")
                        skipped_count += 1
                        continue

                    shutil.move(src, dst)
                    logger.debug(f"Moved: {file}")
                    moved_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to move file {file}: {e}")

        # Remove empty subfolders
        removed_count = 0
        for root, dirs, files in os.walk(parent_folder, topdown=False):
            # Skip the parent folder itself
            if root == parent_folder:
                continue
                
            try:
                if not os.listdir(root):  # Folder is empty
                    os.rmdir(root)
                    logger.debug(f"Removed empty folder: {os.path.basename(root)}")
                    removed_count += 1
            except Exception as e:
                logger.error(f"Failed to remove folder {root}: {e}")

        logger.info(
            f"File organization complete: {moved_count} moved, "
            f"{skipped_count} skipped, {removed_count} folders removed"
        )
        return True
        
    except Exception as e:
        logger.error(f"Critical error in move_files_and_remove_subfolders: {e}")
        return False

# ============================================================================
# VALIDATION AND CLEANUP
# ============================================================================

def validate_input_video(video_path: str) -> bool:
    """
    Validate input video file exists and is readable.
    
    Args:
        video_path: Path to video file
        
    Returns:
        True if video is valid, False otherwise
    """
    try:
        if not os.path.exists(video_path):
            logger.error(f"Video file does not exist: {video_path}")
            return False
        
        if not os.path.isfile(video_path):
            logger.error(f"Path is not a file: {video_path}")
            return False
        
        # Try to open video with OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video file: {video_path}")
            return False
        
        # Check if video has frames
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        if frame_count <= 0 or fps <= 0:
            logger.error(f"Invalid video properties: {frame_count} frames, {fps} fps")
            return False
        
        logger.info(f"Video validation passed: {frame_count} frames at {fps:.2f} fps")
        return True
        
    except Exception as e:
        logger.error(f"Error validating video: {e}")
        return False

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function with comprehensive error handling."""
    
    try:
        start_time = time.time()
        
        logger.info("=" * 70)
        logger.info("ADVANCED SCENE SUBDIVISION SYSTEM - PRODUCTION VERSION")
        logger.info("=" * 70)
        logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Video: {VIDEO_PATH}")
        logger.info(f"Output: {OUTPUT_DIR}")
        
        # Step 0: Validate environment
        logger.info("\n" + "=" * 70)
        logger.info("STEP 0: VALIDATION")
        logger.info("=" * 70)
        
        if not validate_input_video(VIDEO_PATH):
            logger.error("Input video validation failed. Aborting.")
            return False
        
        # Step 1: Scene detection
        logger.info("\n" + "=" * 70)
        logger.info("STEP 1: SCENE DETECTION")
        logger.info("=" * 70)
        
        images_dir = run_scenedetect(VIDEO_PATH, OUTPUT_DIR)
        if images_dir is None:
            logger.error("Scene detection failed. Aborting.")
            return False
        
        logger.info(f"Scene detection completed. Images directory: {images_dir}")
        
        # Step 2: Extract images and detect sub-scenes
        logger.info("\n" + "=" * 70)
        logger.info("STEP 2: IMAGE EXTRACTION AND SUB-SCENE DETECTION")
        logger.info("=" * 70)
        
        subscenes_info = extract_and_analyze_images(
            VIDEO_PATH, 
            OUTPUT_DIR, 
            EXTRACT_INTERVAL_SECONDS
        )
        
        if not subscenes_info:
            logger.error("No sub-scenes detected. Aborting.")
            return False
        
        logger.info(f"Detected {len(subscenes_info)} sub-scenes")
        
        # Step 3: Create sub-scene videos
        logger.info("\n" + "=" * 70)
        logger.info("STEP 3: VIDEO CREATION")
        logger.info("=" * 70)
        
        video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
        
        # if not create_subscene_videos(VIDEO_PATH, OUTPUT_DIR, video_name, subscenes_info):
        #     logger.warning("Some videos failed to create, but continuing...")
        
        # Step 4: Rename files with timestamps
        logger.info("\n" + "=" * 70)
        logger.info("STEP 4: FILE RENAMING")
        logger.info("=" * 70)
        
        if not rename_subscene_files(OUTPUT_DIR, video_name, subscenes_info):
            logger.warning("Some files failed to rename, but continuing...")
        
        # Step 5: CLIP-based detection (optional)
        logger.info("\n" + "=" * 70)
        logger.info("STEP 5: CLIP-BASED SCENE DETECTION")
        logger.info("=" * 70)
        
        if CLIP_MODEL is not None:
            subscenes = detect_subscenes_with_clip(images_dir)
            
            if subscenes:
                final_scene_dir = save_subscenes(subscenes, OUTPUT_DIR)
                
                if final_scene_dir:
                    if not move_files_and_remove_subfolders(final_scene_dir):
                        logger.warning("File organization had issues, but continuing...")
                else:
                    logger.warning("Failed to save CLIP subscenes, but continuing...")
            else:
                logger.warning("CLIP detection failed, but basic detection completed")
        else:
            logger.info("CLIP model not available, skipping semantic detection")

        if os.path.exists(images_dir):
            shutil.rmtree(images_dir)
            
        scene_images = os.path.join(OUTPUT_DIR, "scenes_images")
        os.rename(final_scene_dir, scene_images)

        # Calculate and save processing time
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info("\n" + "=" * 70)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Save processing time to file
        try:
            with open(TIME_FILE, 'w', encoding='utf-8') as f:
                f.write(f"Processing Time Report\n")
                f.write(f"{'=' * 50}\n")
                f.write(f"Video: {VIDEO_PATH}\n")
                f.write(f"Start: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"End: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n")
                f.write(f"Sub-scenes detected: {len(subscenes_info)}\n")
            logger.info(f"Processing time saved to: {TIME_FILE}")
        except Exception as e:
            logger.error(f"Failed to save processing time: {e}")
        
        logger.info("All operations completed successfully!")
        return True
        
    except KeyboardInterrupt:
        logger.warning("\nProcessing interrupted by user")
        return False
    except Exception as e:
        logger.critical(f"Unexpected error in main: {e}", exc_info=True)
        return False

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)