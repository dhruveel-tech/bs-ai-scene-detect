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
VIDEO_PATH = r"D:\SDNA\Scene_Detector\scene_subdivision\Mard\mard.mp4"
OUTPUT_DIR = r"D:\SDNA\Scene_Detector\final_op\Mard"

# ============================================================================
# DETECTION PARAMETERS
# ============================================================================

# Interval (in seconds) at which frames are extracted from the video.
# A smaller value gives more frames but increases processing time.
EXTRACT_INTERVAL_SECONDS = 2

# Minimum number of images to extract per detected scene.
# Ensures at least a few representative frames even if a scene is short.
MIN_IMAGES_PER_SCENE = 3

# Maximum number of images allowed per detected scene.
# Prevents excessive frame extraction in long scenes.
MAX_IMAGES_PER_SCENE = 30

# Histogram similarity threshold (0–1).
# Frames with histogram similarity above this value are considered "too similar", and will be skipped to avoid duplicate visual content.
SIMILARITY_THRESHOLD = 0.85

# Structural Similarity Index (SSIM) threshold (0–1).
# Measures structural similarity between frames. Higher threshold means only significantly different frames will be kept.
STRUCTURAL_THRESHOLD = 0.85

# Edge detection threshold (0–1).
# Compares edge maps to eliminate frames with nearly identical edge structures.
EDGE_THRESHOLD = 0.75

# Minimum duration (in seconds) for a sub-scene.
# Prevents extremely short or noisy sub-scene splits.
MIN_SUBSCENE_DURATION = 2.5

# CLIP model similarity threshold (0–1).
# Used for semantic comparison of frames. Frames with similarity ABOVE this threshold are considered to belong to the same sub-scene.
CLIP_SIMILARITY_THRESHOLD = 0.90

# PySceneDetect command to use for initial scene detection.
# 'detect-hash' = uses content-aware hashing for fast and accurate detection.
SCENE_DETECT_COMMAND = 'detect-hash'

# ============================================================================
# SUPPORTED VIDEO FORMATS
# ============================================================================

# Comprehensive list of supported video formats
SUPPORTED_VIDEO_FORMATS = {
    # Common formats
    '.mp4': 'MPEG-4 Video',
    '.avi': 'Audio Video Interleave',
    '.mov': 'QuickTime Movie',
    '.mkv': 'Matroska Video',
    '.webm': 'WebM Video',
    '.flv': 'Flash Video',
    '.wmv': 'Windows Media Video',
    '.m4v': 'iTunes Video',
    '.mpg': 'MPEG Video',
    '.mpeg': 'MPEG Video',
    '.3gp': '3GPP Video',
    '.3g2': '3GPP2 Video',
    '.ogv': 'Ogg Video',
    '.vob': 'DVD Video Object',
    '.ts': 'MPEG Transport Stream',
    '.mts': 'AVCHD Video',
    '.m2ts': 'Blu-ray Video',
    '.f4v': 'Flash MP4 Video',
    '.asf': 'Advanced Systems Format',
    '.rm': 'RealMedia',
    '.rmvb': 'RealMedia Variable Bitrate',
    '.divx': 'DivX Video',
    '.dv': 'Digital Video',
}

# ============================================================================
# VIDEO FORMAT DETECTION AND VALIDATION
# ============================================================================

def get_video_format_info(video_path: str) -> Dict[str, any]:
    info = {
        'extension': '',
        'format_name': 'Unknown',
        'codec': 'Unknown',
        'is_supported': False,
        'can_read': False,
        'fps': 0.0,
        'frame_count': 0,
        'duration': 0.0,
        'width': 0,
        'height': 0,
        'size_mb': 0.0
    }
    
    try:
        # Get file extension
        _, ext = os.path.splitext(video_path)
        ext = ext.lower()
        info['extension'] = ext
        
        # Check if format is in our supported list
        if ext in SUPPORTED_VIDEO_FORMATS:
            info['format_name'] = SUPPORTED_VIDEO_FORMATS[ext]
            info['is_supported'] = True
        
        # Get file size
        if os.path.exists(video_path):
            info['size_mb'] = os.path.getsize(video_path) / (1024 * 1024)
        
        # Try to get codec info using FFprobe
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                 '-show_entries', 'stream=codec_name,width,height,r_frame_rate,nb_frames',
                 '-of', 'csv=p=0', video_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout:
                parts = result.stdout.strip().split(',')
                if len(parts) >= 4:
                    info['codec'] = parts[0]
                    info['width'] = int(parts[1]) if parts[1] else 0
                    info['height'] = int(parts[2]) if parts[2] else 0
                    
                    # Parse frame rate (e.g., "30/1" -> 30.0)
                    if '/' in parts[3]:
                        num, den = parts[3].split('/')
                        info['fps'] = float(num) / float(den)
                    else:
                        info['fps'] = float(parts[3])
                    
                    if len(parts) >= 5 and parts[4]:
                        info['frame_count'] = int(parts[4])
        except:
            pass
        
        # Try to open with OpenCV
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            info['can_read'] = True
            
            # Get properties from OpenCV if not already set
            if info['fps'] == 0.0:
                info['fps'] = cap.get(cv2.CAP_PROP_FPS)
            if info['frame_count'] == 0:
                info['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if info['width'] == 0:
                info['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            if info['height'] == 0:
                info['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate duration
            if info['fps'] > 0 and info['frame_count'] > 0:
                info['duration'] = info['frame_count'] / info['fps']
            
            cap.release()
        
        return info
        
    except Exception as e:
        print(f"Error getting video format info: {e}")
        return info

def validate_video_format(video_path: str) -> Tuple[bool, str]:
    try:
        # Check file exists
        if not os.path.exists(video_path):
            return False, "Video file does not exist"
        
        if not os.path.isfile(video_path):
            return False, "Path is not a file"
        
        # Get format info
        info = get_video_format_info(video_path)
        
        # Print video information
        print("\n" + "=" * 70)
        print("VIDEO FORMAT INFORMATION")
        print("=" * 70)
        print(f"File: {os.path.basename(video_path)}")
        print(f"Format: {info['format_name']} ({info['extension']})")
        print(f"Codec: {info['codec']}")
        print(f"Resolution: {info['width']}x{info['height']}")
        print(f"Frame Rate: {info['fps']:.2f} fps")
        print(f"Total Frames: {info['frame_count']:,}")
        print(f"Duration: {info['duration']:.2f} seconds ({info['duration']/60:.2f} minutes)")
        print(f"File Size: {info['size_mb']:.2f} MB")
        print(f"OpenCV Compatible: {'Yes ✓' if info['can_read'] else 'No ✗'}")
        print("=" * 70 + "\n")
        
        # Validate
        if not info['can_read']:
            return False, f"Cannot read video file. Format may not be supported by OpenCV."
        
        if info['frame_count'] <= 0:
            return False, "Video has no frames"
        
        if info['fps'] <= 0:
            return False, "Invalid frame rate"
        
        # Warning for uncommon formats
        if not info['is_supported']:
            print(f"⚠ WARNING: '{info['extension']}' is not in the common formats list.")
            print(f"   The system will attempt to process it anyway.")
            print(f"   If you encounter issues, consider converting to MP4/MKV/AVI first.\n")
        
        return True, f"Video format validated: {info['format_name']}"
        
    except Exception as e:
        return False, f"Video validation failed: {str(e)}"

def convert_video_if_needed(video_path: str, output_dir: str) -> Tuple[str, bool]:
    try:
        info = get_video_format_info(video_path)
        
        # Check if conversion is recommended
        problematic_formats = ['.rm', '.rmvb', '.vob', '.divx', '.dv']
        needs_conversion = (
            not info['can_read'] or 
            info['extension'] in problematic_formats
        )
        
        if not needs_conversion:
            return video_path, False
        
        print(f"\n{'=' * 70}")
        print(f"VIDEO CONVERSION REQUIRED")
        print(f"{'=' * 70}")
        print(f"Format '{info['extension']}' may have compatibility issues.")
        print(f"Converting to MP4 for better compatibility...")
        
        # Create converted filename
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        converted_path = os.path.join(output_dir, f"{base_name}_converted.mp4")
        
        # Convert using FFmpeg
        cmd = [
            'ffmpeg', '-i', video_path,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-y',  # Overwrite
            converted_path
        ]
        
        print("Converting... (this may take a few minutes)")
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode == 0 and os.path.exists(converted_path):
            print(f"✓ Conversion successful: {converted_path}")
            print(f"{'=' * 70}\n")
            return converted_path, True
        else:
            print(f"✗ Conversion failed: {result.stderr}")
            print(f"Attempting to process original file anyway...")
            print(f"{'=' * 70}\n")
            return video_path, False
            
    except Exception as e:
        print(f"Conversion error: {e}")
        return video_path, False

# ============================================================================
# CLIP MODEL SETUP
# ============================================================================

def load_clip_model() -> Tuple[Optional[CLIPModel], Optional[CLIPProcessor]]:
    try:
        print("Loading CLIP model...")
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # print("Using device:", device)
        # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("CLIP model loaded successfully")
        return model, processor
    except Exception as e:
        print(f"Failed to load CLIP model: {e}")
        print("Continuing without CLIP-based detection")
        return None, None

# Load CLIP model
CLIP_MODEL, CLIP_PROCESSOR = load_clip_model()

# ============================================================================
# CLIP FEATURE EXTRACTION
# ============================================================================

def get_features(img_path: str) -> Optional[torch.Tensor]:
    try:
        if CLIP_MODEL is None or CLIP_PROCESSOR is None:
            return None
            
        image = Image.open(img_path).convert("RGB")
        inputs = CLIP_PROCESSOR(images=image, return_tensors="pt")
        # inputs = CLIP_PROCESSOR(images=image, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            features = CLIP_MODEL.get_image_features(**inputs)
        
        # Normalize features
        return features[0] / features[0].norm()
        
    except Exception as e:
        print(f"Failed to extract features from {img_path}: {e}")
        return None

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    try:
        return torch.dot(a, b).item()
    except Exception as e:
        print(f"Failed to calculate cosine similarity: {e}")
        return 0.0

# ============================================================================
# SCENE DETECTION WITH SCENEDETECT
# ============================================================================

def run_scenedetect(video_path: str, output_dir: str) -> Optional[str]:
    try:
        images_dir = os.path.join(output_dir, "scenes_images")
        video_dir = os.path.join(output_dir, "scenes_videos")

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)

        print("Detecting scenes and splitting video...")
        
        # Split video into scenes
        result = subprocess.run(
            [
                "scenedetect", "-i", video_path,
                "--output", video_dir,
                SCENE_DETECT_COMMAND,
                "split-video"
            ],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Scene split output: {result.stdout}")

        print("Generating scene list CSV...")
        
        # Generate scene list CSV
        result = subprocess.run(
            [
                "scenedetect", "-i", video_path,
                "--output", images_dir,
                SCENE_DETECT_COMMAND,
                "list-scenes"
            ],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Scene list output: {result.stdout}")

        return images_dir
        
    except subprocess.CalledProcessError as e:
        print(f"Scenedetect command failed: {e.stderr}")
        return None
    except FileNotFoundError:
        print("Scenedetect not found. Please install: pip install scenedetect[opencv]")
        return None
    except Exception as e:
        print(f"Unexpected error in run_scenedetect: {e}")
        return None

# ============================================================================
# IMAGE SIMILARITY CALCULATIONS
# ============================================================================

def calculate_histogram_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
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
        print(f"Failed to calculate histogram similarity: {e}")
        return 0.0

def calculate_structural_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
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
        print(f"Failed to calculate structural similarity: {e}")
        return 0.0

def calculate_edge_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
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
        print(f"Failed to calculate edge similarity: {e}")
        return 0.0

# ============================================================================
# ADVANCED SUBSCENE DETECTION
# ============================================================================

def detect_subscenes_advanced(
    frames_data: List[Dict], 
    min_duration: float = MIN_SUBSCENE_DURATION
) -> List[Tuple[int, int]]:
    try:
        if len(frames_data) < 2:
            print("Not enough frames for subscene detection")
            return [(0, len(frames_data) - 1)]
        
        print(f"Analyzing {len(frames_data)} frames for sub-scene detection...")
        
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
                print(f"Error calculating similarity for frame {i}: {e}")
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
                                # print(f"Potential transition at {sim['timestamp']:.2f}s - not confirmed")
                        except Exception as e:
                            print(f"Error in look-ahead confirmation: {e}")
                    
                    if confirmed:
                        # Create sub-scene break
                        subscenes.append((current_start, frame_idx - 1))
                        print(
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
                print(f"Merged short sub-scene ({duration:.1f}s) with previous")
            else:
                merged_subscenes.append((start, end))
        
        final_subscenes = merged_subscenes if merged_subscenes else subscenes
        # print(f"Detected {len(final_subscenes)} sub-scenes")
        
        return final_subscenes
        
    except Exception as e:
        print(f"Critical error in detect_subscenes_advanced: {e}")
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
    try:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        images_dir = os.path.join(output_dir, "scenes_images")
        # subscenes_dir = os.path.join(output_dir, "subscenes_videos")
        
        # os.makedirs(subscenes_dir, exist_ok=True)
        
        csv_path = os.path.join(images_dir, f"{base_name}-Scenes.csv")
        
        # Validate CSV exists
        if not os.path.exists(csv_path):
            print(f"Scene CSV not found: {csv_path}")
            return []
        
        # Read scene CSV
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if len(lines) < 2:
                    print("Scene CSV is empty or malformed")
                    return []
                    
                cleaned_csv = lines[1:]  # Skip header comment line
                reader = csv.DictReader(cleaned_csv)
                scenes = list(reader)
                
            print(f"Found {len(scenes)} scenes in CSV")
            
        except Exception as e:
            print(f"Failed to read scene CSV: {e}")
            return []
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {fps:.2f} FPS, {total_frames} total frames")
        print(f"Extracting frames every {interval_seconds}s")
        
        all_subscenes_info = []
        
        # Process each scene
        for scene_idx, scene in enumerate(scenes, start=1):
            try:
                start_frame = int(scene["Start Frame"])
                end_frame = int(scene["End Frame"])
                start_time = float(scene["Start Time (seconds)"])
                end_time = float(scene["End Time (seconds)"])
                scene_duration = float(scene["Length (seconds)"])
                
                print(f"-----------------------------------------\nProcessing Scene {scene_idx:03d}: {scene_duration:.2f}s")
                
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
                        print(f"Frame {frame_num} exceeds video length")
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
                        print(f"Failed to read frame {frame_num}")
                    
                    current_time += interval_seconds
                
                print(f"Extracted {len(frames_data)} frames from scene {scene_idx}")
                
                # Detect sub-scenes if we have enough frames
                if len(frames_data) > 0:
                    subscene_boundaries = detect_subscenes_advanced(frames_data, MIN_SUBSCENE_DURATION)
                    
                    print(f"Scene {scene_idx} divided into {len(subscene_boundaries)} sub-scenes")
                    
                    # Save images and info for each sub-scene
                    for sub_idx, (start_idx, end_idx) in enumerate(subscene_boundaries, start=1):
                        try:
                            sub_start_time = frames_data[start_idx]['timestamp']
                            sub_end_time = frames_data[end_idx]['timestamp']
                            sub_duration = sub_end_time - sub_start_time
                            
                            print(
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
                            print(f"Failed to process sub-scene {sub_idx} of scene {scene_idx}: {e}")
                
            except Exception as e:
                print(f"Failed to process scene {scene_idx}: {e}")
                continue
        
        cap.release()
        
        print(f"Analyzed {len(scenes)} scenes into {len(all_subscenes_info)} sub-scenes")
        return all_subscenes_info
        
    except Exception as e:
        print(f"Critical error in extract_and_analyze_images: {e}")
        return []

# ============================================================================
# FILE RENAMING
# ============================================================================

def format_timestamp(seconds: float) -> str:
    try:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        millisecs = int((secs - int(secs)) * 1000)
        return f"{hours:02d}-{minutes:02d}-{int(secs):02d}.{millisecs:03d}"
    except Exception as e:
        print(f"Failed to format timestamp {seconds}: {e}")
        return "00-00-00.000"

def rename_subscene_files(
    output_dir: str, 
    base_name: str, 
    subscenes_info: List[Dict]
) -> bool:
    try:
        images_dir = os.path.join(output_dir, "scenes_images")
        # subscenes_dir = os.path.join(output_dir, "subscenes_videos")
        
        print("Renaming files with timestamps...")
        
        success_count = 0
        total_operations = 0
        
        for sub_info in subscenes_info:
            try:
                scene_num = sub_info['scene_num']
                subscene_num = sub_info['subscene_num']

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
                            print(f"Failed to rename image {old_img}: {e}")
                    else:
                        print(f"Image file not found: {old_img}")
                        
            except Exception as e:
                print(f"Failed to rename files for scene {sub_info.get('scene_num')} sub {sub_info.get('subscene_num')}: {e}")
        
        print(f"Renamed {success_count}/{total_operations} files")
        return success_count == total_operations
        
    except Exception as e:
        print(f"Critical error in rename_subscene_files: {e}")
        return False

# ============================================================================
# CLIP-BASED SUBSCENE DETECTION
# ============================================================================

def detect_subscenes_with_clip(images_dir: str) -> Optional[Dict[int, List[str]]]:
    try:
        if CLIP_MODEL is None or CLIP_PROCESSOR is None:
            print("CLIP model not available, skipping CLIP-based detection")
            return None
        
        # Get all image paths and sort them
        image_paths = sorted([
            os.path.join(images_dir, f) 
            for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])

        if not image_paths:
            print("No images found in directory")
            return None

        print(f"Processing {len(image_paths)} images with CLIP model...")

        # Extract features for all images
        embeddings = []
        for i, img_path in enumerate(image_paths):
            if i % 10 == 0:
                print(f"Extracting features: {i}/{len(image_paths)}")
            
            features = get_features(img_path)
            if features is None:
                print(f"Failed to extract features from {img_path}")
                return None
            embeddings.append(features)

        # Detect scene boundaries based on similarity
        subscene_id = 0
        subscenes = {subscene_id: [image_paths[0]]}

        for i in range(1, len(embeddings)):
            sim = cosine_similarity(embeddings[i-1], embeddings[i])
            # print(f"CLIP similarity {i}: {sim:.4f}")

            # If similarity drops below threshold, start new subscene
            if sim < CLIP_SIMILARITY_THRESHOLD: 
                subscene_id += 1
                subscenes[subscene_id] = []
                # print(f"New sub-scene detected at image {i} (similarity: {sim:.4f})")

            subscenes[subscene_id].append(image_paths[i])

        print(f"CLIP detected {len(subscenes)} sub-scenes")
        return subscenes
        
    except Exception as e:
        print(f"Failed in detect_subscenes_with_clip: {e}")
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
    try:
        if not os.path.exists(parent_folder):
            print(f"Parent folder does not exist: {parent_folder}")
            return False
        
        print("Moving files from subfolders to parent directory...")
        
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
                        print(f"Skipped (already exists): {file}")
                        skipped_count += 1
                        continue

                    shutil.move(src, dst)
                    # print(f"Moved: {file}")
                    moved_count += 1
                    
                except Exception as e:
                    print(f"Failed to move file {file}: {e}")

        # Remove empty subfolders
        removed_count = 0
        for root, dirs, files in os.walk(parent_folder, topdown=False):
            # Skip the parent folder itself
            if root == parent_folder:
                continue
                
            try:
                if not os.listdir(root):  # Folder is empty
                    os.rmdir(root)
                    # print(f"Removed empty folder: {os.path.basename(root)}")
                    removed_count += 1
            except Exception as e:
                print(f"Failed to remove folder {root}: {e}")

        print(
            f"File organization complete: {moved_count} moved, "
            f"{skipped_count} skipped, {removed_count} folders removed"
        )
        return True
        
    except Exception as e:
        print(f"Critical error in move_files_and_remove_subfolders: {e}")
        return False

# ============================================================================
# VALIDATION AND CLEANUP
# ============================================================================

def validate_input_video(video_path: str) -> bool:
    try:
        if not os.path.exists(video_path):
            print(f"Video file does not exist: {video_path}")
            return False
        
        if not os.path.isfile(video_path):
            print(f"Path is not a file: {video_path}")
            return False
        
        # Try to open video with OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open video file: {video_path}")
            return False
        
        # Check if video has frames
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        if frame_count <= 0 or fps <= 0:
            print(f"Invalid video properties: {frame_count} frames, {fps} fps")
            return False
        
        print(f"Video validation passed: {frame_count} frames at {fps:.2f} fps")
        return True
        
    except Exception as e:
        print(f"Error validating video: {e}")
        return False

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# def main():
#     """Main execution function with comprehensive error handling."""
    
#     try:
#         start_time = time.time()
        
#         print("=" * 70)
#         print("ADVANCED SCENE SUBDIVISION SYSTEM")
#         print("=" * 70)
#         print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#         print(f"Video: {VIDEO_PATH}")
#         print(f"Output: {OUTPUT_DIR}")
        
#         # Step 0: Validate environment
#         print("\n" + "=" * 70)
#         print("STEP 0: VALIDATION")
#         print("=" * 70)
        
#         if not validate_input_video(VIDEO_PATH):
#             print("Input video validation failed. Aborting.")
#             return False
        
#         # Step 1: Scene detection
#         print("\n" + "=" * 70)
#         print("STEP 1: SCENE DETECTION")
#         print("=" * 70)
#         scene_start_time = time.time()
#         images_dir = run_scenedetect(VIDEO_PATH, OUTPUT_DIR)
#         if images_dir is None:
#             print("Scene detection failed. Aborting.")
#             return False
        
#         print(f"Scene detection completed. Images directory: {images_dir}")
#         scene_end_time = time.time()
#         # Step 2: Extract images and detect sub-scenes
#         print("\n" + "=" * 70)
#         print("STEP 2: IMAGE EXTRACTION AND SUB-SCENE DETECTION")
#         print("=" * 70)
#         image_start_time = time.time()
#         subscenes_info = extract_and_analyze_images(
#             VIDEO_PATH, 
#             OUTPUT_DIR, 
#             EXTRACT_INTERVAL_SECONDS
#         )
        
#         if not subscenes_info:
#             print("No sub-scenes detected. Aborting.")
#             return False
        
#         print(f"Detected {len(subscenes_info)} sub-scenes")
#         image_end_time = time.time()
#         # Step 3: Create sub-scene videos
#         print("\n" + "=" * 70)
#         print("STEP 3: VIDEO CREATION")
#         print("=" * 70)
        
#         video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
        
#         # Step 4: Rename files with timestamps
#         print("\n" + "=" * 70)
#         print("STEP 4: FILE RENAMING")
#         print("=" * 70)

#         rename_start_time = time.time()
#         if not rename_subscene_files(OUTPUT_DIR, video_name, subscenes_info):
#             print("Some files failed to rename, but continuing...")
#         rename_end_time = time.time()
#         # Step 5: CLIP-based detection (optional)
#         print("\n" + "=" * 70)
#         print("STEP 5: CLIP-BASED SCENE DETECTION")
#         print("=" * 70)
#         clip_start_time = time.time()
#         if CLIP_MODEL is not None:
#             subscenes = detect_subscenes_with_clip(images_dir)
            
#             if subscenes:
#                 final_scene_dir = save_subscenes(subscenes, OUTPUT_DIR)
                
#                 if final_scene_dir:
#                     if not move_files_and_remove_subfolders(final_scene_dir):
#                         print("File organization had issues, but continuing...")
#                 else:
#                     print("Failed to save CLIP subscenes, but continuing...")
#             else:
#                 print("CLIP detection failed, but basic detection completed")
#         else:
#             print("CLIP model not available, skipping semantic detection")
#         print(f"Clip Total Time :- {time.time() - clip_start_time}")
#         print(f"scene detect Total Time :- { scene_end_time - scene_start_time}")
#         print(f"image Total Time :- { image_end_time - image_start_time}")
#         print(f"rename Total Time :- {rename_end_time - rename_start_time}")
#         if os.path.exists(images_dir):
#             shutil.rmtree(images_dir)
            
#         scene_images = os.path.join(OUTPUT_DIR, "scenes_images")
#         os.rename(final_scene_dir, scene_images)

#         end_time = time.time()
#         total_time = end_time - start_time
        
#         print("\n" + "=" * 70)
#         print("PROCESSING COMPLETE")
#         print("=" * 70)
#         print(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
#         print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
#         print("All operations completed successfully!")
#         return True
        
#     except KeyboardInterrupt:
#         print("\nProcessing interrupted by user")
#         return False
#     except Exception as e:
#         print(f"Unexpected error in main: {e}", exc_info=True)
#         return False


def main():
    """Main execution function with comprehensive error handling."""
    
    try:
        start_time = time.time()
        
        print("=" * 70)
        print("ADVANCED SCENE SUBDIVISION SYSTEM")
        print("=" * 70)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Video: {VIDEO_PATH}")
        print(f"Output: {OUTPUT_DIR}")

        # =====================================================================
        # STEP 1: VIDEO FORMAT VALIDATION (NEW)
        # =====================================================================
        print("\n" + "=" * 70)
        print("STEP 1: VIDEO FORMAT CHECK")
        print("=" * 70)

        ok, msg = validate_video_format(VIDEO_PATH)
        print(msg)

        if not ok:
            print("Video format is not ideal. Will attempt conversion.")
        else:
            print("Video format looks good.")

        # =====================================================================
        # STEP 2: VIDEO CONVERSION IF REQUIRED (NEW)
        # =====================================================================
        print("\n" + "=" * 70)
        print("STEP 2: CONVERSION IF REQUIRED")
        print("=" * 70)

        final_video_path, converted = convert_video_if_needed(VIDEO_PATH, OUTPUT_DIR)

        if converted:
            print(f"Using converted video: {final_video_path}")
        else:
            print("No conversion needed. Proceeding with original video.")

        # IMPORTANT: UPDATE LOCATION FOR REST OF PIPELINE
        working_video = final_video_path

        # Optional: Print final format info
        print("\nFINAL VIDEO FORMAT INFO:")
        info = get_video_format_info(working_video)
        print(info)

        # =====================================================================
        # STEP 2.1: BASIC VALIDATION ALREADY IN YOUR CODE
        # =====================================================================
        if not validate_input_video(working_video):
            print("Input video validation failed. Aborting.")
            return False

        # =====================================================================
        # STEP 3: SCENE DETECTION
        # =====================================================================
        print("\n" + "=" * 70)
        print("STEP 3: SCENE DETECTION")
        print("=" * 70)

        scene_start_time = time.time()
        images_dir = run_scenedetect(working_video, OUTPUT_DIR)

        if images_dir is None:
            print("Scene detection failed. Aborting.")
            return False
        
        print(f"Scene detection completed. Images directory: {images_dir}")
        scene_end_time = time.time()

        # =====================================================================
        # STEP 4: SUB-SCENE DETECTION
        # =====================================================================
        print("\n" + "=" * 70)
        print("STEP 4: IMAGE EXTRACTION AND SUB-SCENE DETECTION")
        print("=" * 70)

        image_start_time = time.time()
        subscenes_info = extract_and_analyze_images(
            working_video, 
            OUTPUT_DIR, 
            EXTRACT_INTERVAL_SECONDS
        )

        if not subscenes_info:
            print("No sub-scenes detected. Aborting.")
            return False
        
        print(f"Detected {len(subscenes_info)} sub-scenes")
        image_end_time = time.time()

        # =====================================================================
        # STEP 5: RENAME FILES
        # =====================================================================
        print("\n" + "=" * 70)
        print("STEP 5: FILE RENAMING")
        print("=" * 70)

        rename_start_time = time.time()
        video_name = os.path.splitext(os.path.basename(working_video))[0]

        if not rename_subscene_files(OUTPUT_DIR, video_name, subscenes_info):
            print("Some files failed to rename, but continuing...")

        rename_end_time = time.time()

        # =====================================================================
        # STEP 6: CLIP DETECTION (OPTIONAL)
        # =====================================================================
        print("\n" + "=" * 70)
        print("STEP 6: CLIP-BASED SCENE DETECTION")
        print("=" * 70)

        clip_start_time = time.time()

        if CLIP_MODEL is not None:
            subscenes = detect_subscenes_with_clip(images_dir)
            
            if subscenes:
                final_scene_dir = save_subscenes(subscenes, OUTPUT_DIR)
                
                if final_scene_dir:
                    if not move_files_and_remove_subfolders(final_scene_dir):
                        print("File organization had issues, but continuing...")
                else:
                    print("Failed to save CLIP subscenes, but continuing...")
            else:
                print("CLIP detection failed, but basic detection completed")
        else:
            print("CLIP model not available, skipping semantic detection")

        # =====================================================================
        # PERFORMANCE SUMMARY
        # =====================================================================
        print(f"Clip Total Time :- {time.time() - clip_start_time}")
        print(f"scene detect Total Time :- {scene_end_time - scene_start_time}")
        print(f"image Total Time :- {image_end_time - image_start_time}")
        print(f"rename Total Time :- {rename_end_time - rename_start_time}")

        # Cleanup temp folder
        if os.path.exists(images_dir):
            shutil.rmtree(images_dir)

        # Move final scene folder
        scene_images = os.path.join(OUTPUT_DIR, "scenes_images")
        os.rename(final_scene_dir, scene_images)

        total_time = time.time() - start_time

        print("\n" + "=" * 70)
        print("PROCESSING COMPLETE")
        print("=" * 70)
        print(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("All operations completed successfully!")
        
        return True
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        return False
    except Exception as e:
        print(f"Unexpected error in main: {e}", exc_info=True)
        return False


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        success = main()
    except Exception as e:
        print(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)