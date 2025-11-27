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
# CLIP MODEL SETUP
# ============================================================================

def load_clip_model(device, video_path) -> Tuple[Optional[CLIPModel], Optional[CLIPProcessor]]:
    try:
      
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print(f"[{video_path}] CLIP model loaded successfully")
        return model, processor
    except Exception as e:
        print(f"[{video_path}] Failed to load CLIP model: {e}")
        return None, None

# ============================================================================
# CLIP FEATURE EXTRACTION
# ============================================================================

def get_features(img_path: str, device, clip_model, clip_processor, video_path) -> Optional[torch.Tensor]:
    try:
        if clip_model is None or clip_processor is None:
            return None
            
        image = Image.open(img_path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            features = clip_model.get_image_features(**inputs)
        
        # Normalize features
        return features[0] / features[0].norm()
        
    except Exception as e:
        print(f"[{video_path}] Failed to extract features from {img_path}: {e}")
        return None

def cosine_similarity(a: torch.Tensor, b: torch.Tensor, video_path) -> float:
    try:
        return torch.dot(a, b).item()
    except Exception as e:
        print(f"[{video_path}] Failed to calculate cosine similarity: {e}")
        return 0.0

# ============================================================================
# SCENE DETECTION WITH SCENEDETECT
# ============================================================================

# def run_scenedetect(video_path: str, output_dir: str, scene_detect_command: str) -> Optional[str]:
#     try:
#         images_dir = os.path.join(output_dir, "scenes_images")
#         video_dir = os.path.join(output_dir, "scenes_videos")

#         def recreate_dir(path):
#             if os.path.exists(path):
#                 shutil.rmtree(path)
#             os.makedirs(path)

#         recreate_dir(images_dir)
#         recreate_dir(video_dir)
        
#         # Split video into scenes
#         result = subprocess.run(
#             [
#                 "scenedetect", "-i", video_path,
#                 "--output", video_dir,
#                 scene_detect_command,
#                 "split-video"
#             ],
#             check=True,
#             capture_output=True,
#             text=True
#         )
#         print(f"[{video_path}] Scene split output: {result.stdout}")
        
#         # Generate scene list CSV
#         result = subprocess.run(
#             [
#                 "scenedetect", "-i", video_path,
#                 "--output", images_dir,
#                 scene_detect_command,
#                 "list-scenes"
#             ],
#             check=True,
#             capture_output=True,
#             text=True
#         )
#         print(f"[{video_path}] Scene list output: {result.stdout}")

#         return images_dir
        
#     except subprocess.CalledProcessError as e:
#         print(f"[{video_path}] Scenedetect command failed: {e.stderr}")
#         return None
#     except FileNotFoundError:
#         print(f"[{video_path}] Scenedetect not found. Please install: pip install scenedetect[opencv]")
#         return None
#     except Exception as e:
#         print(f"[{video_path}] Unexpected error in run_scenedetect: {e}")
#         return None

def run_scenedetect(video_path: str, output_dir: str, scene_detect_command: str) -> Optional[str]:
    try:
        images_dir = os.path.join(output_dir, "scenes_images")
        video_dir = os.path.join(output_dir, "scenes_videos")
 
        def recreate_dir(path):
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
 
        recreate_dir(images_dir)
        recreate_dir(video_dir)
       
        # Split video into scenes
        result = subprocess.Popen(
            [
                "scenedetect", "-i", video_path,
                "--output", video_dir,
                scene_detect_command,
                "split-video"
            ],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,bufsize=0
        )
        while True:
            chunk = result.stdout.read(4096)
            if not chunk:
                break
 
            text = chunk.decode("utf-8", errors="ignore")
            print(f"[{video_path}] Scene split output: {text}")
       
        # Generate scene list CSV
        result = subprocess.Popen(
            [
                "scenedetect", "-i", video_path,
                "--output", images_dir,
                scene_detect_command,
                "list-scenes"
            ],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,bufsize=0
        )
        while True:
            chunk = result.stdout.read(4096)
            if not chunk:
                break
 
            text = chunk.decode("utf-8", errors="ignore")
            print(f"[{video_path}] Scene list output: {text}")
 
        return images_dir
       
    except subprocess.CalledProcessError as e:
        print(f"[{video_path}] Scenedetect command failed: {e.stderr}")
        return None
    except FileNotFoundError:
        print(f"[{video_path}] Scenedetect not found. Please install: pip install scenedetect[opencv]")
        return None
    except Exception as e:
        print(f"[{video_path}] Unexpected error in run_scenedetect: {e}")
        return None

# ============================================================================
# IMAGE SIMILARITY CALCULATIONS
# ============================================================================

def calculate_histogram_similarity(img1: np.ndarray, img2: np.ndarray, video_path) -> float:
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
        print(f"[{video_path}] Failed to calculate histogram similarity: {e}")
        return 0.0

def calculate_structural_similarity(img1: np.ndarray, img2: np.ndarray, video_path) -> float:
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
        print(f"[{video_path}] Failed to calculate structural similarity: {e}")
        return 0.0

def calculate_edge_similarity(img1: np.ndarray, img2: np.ndarray, video_path) -> float:
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
        print(f"[{video_path}] Failed to calculate edge similarity: {e}")
        return 0.0

# ============================================================================
# ADVANCED SUBSCENE DETECTION
# ============================================================================

def detect_subscenes_advanced(
    frames_data: List[Dict], 
    min_duration: float,
    similarity_threshold: float,
    structural_threshold: float,
    edge_threshold: float,
    video_path
) -> List[Tuple[int, int]]:
    try:
        if len(frames_data) < 2:
            return [(0, len(frames_data) - 1)]
           
        # Calculate all similarity metrics between consecutive frames
        similarities = []
        for i in range(1, len(frames_data)):
            try:
                prev_img = frames_data[i-1]['image']
                curr_img = frames_data[i]['image']
                
                hist_sim = calculate_histogram_similarity(prev_img, curr_img, video_path)
                struct_sim = calculate_structural_similarity(prev_img, curr_img, video_path)
                edge_sim = calculate_edge_similarity(prev_img, curr_img, video_path)
                
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
                print(f"[{video_path}] Error calculating similarity for frame {i}: {e}")
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
            is_histogram_change = sim['histogram'] < similarity_threshold
            is_structural_change = sim['structural'] < structural_threshold
            is_edge_change = sim['edge'] < edge_threshold
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
                            
                            future_hist_sim = calculate_histogram_similarity(current_img, future_img, video_path)
                            
                            # If future frames are also different, might be transition effect
                            if future_hist_sim < 0.70:
                                confirmed = False
                                # print(f"Potential transition at {sim['timestamp']:.2f}s - not confirmed")
                        except Exception as e:
                            print(f"[{video_path}] Error in look-ahead confirmation: {e}")
                    
                    if confirmed:
                        # Create sub-scene break
                        subscenes.append((current_start, frame_idx - 1))
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
            else:
                merged_subscenes.append((start, end))
        
        final_subscenes = merged_subscenes if merged_subscenes else subscenes

        return final_subscenes
        
    except Exception as e:
        print(f"[{video_path}] Critical error in detect_subscenes_advanced: {e}")
        # Return single scene as fallback
        return [(0, len(frames_data) - 1)]

# ============================================================================
# IMAGE EXTRACTION AND ANALYSIS
# ============================================================================

def extract_and_analyze_images(
    video_path: str, 
    output_dir: str, 
    interval_seconds: float,
    min_images_per_scene: int,
    max_images_per_scene: int,
    min_subscene_duration: float,
    similarity_threshold: float,
    structural_threshold: float,
    edge_threshold: float
) -> List[Dict]:
    try:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        images_dir = os.path.join(output_dir, "scenes_images")
        # subscenes_dir = os.path.join(output_dir, "subscenes_videos")
        
        # os.makedirs(subscenes_dir, exist_ok=True)
        
        csv_path = os.path.join(images_dir, f"{base_name}-Scenes.csv")
        
        # Validate CSV exists
        if not os.path.exists(csv_path):
            print(f"[{video_path}] Scene CSV not found: {csv_path}")
            return []
        
        # Read scene CSV
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if len(lines) < 2:
                    return []
                    
                cleaned_csv = lines[1:]  # Skip header comment line
                reader = csv.DictReader(cleaned_csv)
                scenes = list(reader)
                
        except Exception as e:
            print(f"[{video_path}] Failed to read scene CSV: {e}")
            return []
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[{video_path}] Failed to open video: {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[{video_path}] Video properties: {fps:.2f} FPS, {total_frames} total frames")
        print(f"[{video_path}] Extracting frames every {interval_seconds}s")
        
        all_subscenes_info = []
        
        # Process each scene
        for scene_idx, scene in enumerate(scenes, start=1):
            try:
                start_frame = int(scene["Start Frame"])
                end_frame = int(scene["End Frame"])
                start_time = float(scene["Start Time (seconds)"])
                end_time = float(scene["End Time (seconds)"])
                scene_duration = float(scene["Length (seconds)"])
                
                # Calculate number of images to extract
                num_images = max(
                    min_images_per_scene, 
                    min(max_images_per_scene, int(scene_duration / interval_seconds) + 1)
                )
                
                # Extract frames from this scene
                frames_data = []
                current_time = start_time
                img_idx = 1
                
                while current_time <= end_time and len(frames_data) < num_images:
                    frame_num = int(start_frame + ((current_time - start_time) * fps))
                    
                    # Ensure frame number is valid
                    if frame_num >= total_frames:
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
                        print(f"[{video_path}] Failed to read frame {frame_num}")
                    
                    current_time += interval_seconds
                
                print(f"[{video_path}] Extracted {len(frames_data)} frames from scene {scene_idx}")
                
                # Detect sub-scenes if we have enough frames
                if len(frames_data) > 0:
                    subscene_boundaries = detect_subscenes_advanced(frames_data, min_subscene_duration, similarity_threshold, structural_threshold, edge_threshold, video_path)
                    
                    # Save images and info for each sub-scene
                    for sub_idx, (start_idx, end_idx) in enumerate(subscene_boundaries, start=1):
                        try:
                            sub_start_time = frames_data[start_idx]['timestamp']
                            sub_end_time = frames_data[end_idx]['timestamp']
                            sub_duration = sub_end_time - sub_start_time
                            
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
                            print(f"[{video_path}] Failed to process sub-scene {sub_idx} of scene {scene_idx}: {e}")
                
            except Exception as e:
                print(f"[{video_path}] Failed to process scene {scene_idx}: {e}")
                continue
        
        cap.release()
        return all_subscenes_info
        
    except Exception as e:
        print(f"[{video_path}] Critical error in extract_and_analyze_images: {e}")
        return []

# ============================================================================
# FILE RENAMING
# ============================================================================

def format_timestamp(seconds: float, video_path) -> str:
    try:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        millisecs = int((secs - int(secs)) * 1000)
        return f"{hours:02d}-{minutes:02d}-{int(secs):02d}.{millisecs:03d}"
    except Exception as e:
        print(f"[{video_path}] Failed to format timestamp {seconds}: {e}")
        return "00-00-00.000"

def rename_subscene_files(
    output_dir: str, 
    base_name: str, 
    subscenes_info: List[Dict],
    video_path: str
) -> bool:
    try:
        images_dir = os.path.join(output_dir, "scenes_images")

        success_count = 0
        total_operations = 0
        
        for sub_info in subscenes_info:
            try:
                scene_num = sub_info['scene_num']
                subscene_num = sub_info['subscene_num']

                # Rename image files
                for i, frame_data in enumerate(sub_info['frames'], start=1):
                    timestamp = format_timestamp(frame_data['timestamp'], video_path)
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
                            print(f"[{video_path}] Failed to rename image {old_img}: {e}")
                    else:
                        print(f"[{video_path}] Image file not found: {old_img}")
                        
            except Exception as e:
                print(f"[{video_path}] Failed to rename files for scene {sub_info.get('scene_num')} sub {sub_info.get('subscene_num')}: {e}")
        
        print(f"[{video_path}] Renamed {success_count}/{total_operations} files")
        return success_count == total_operations
        
    except Exception as e:
        print(f"[{video_path}] Critical error in rename_subscene_files: {e}")
        return False

# ============================================================================
# CLIP-BASED SUBSCENE DETECTION
# ============================================================================

def detect_subscenes_with_clip(images_dir: str, clip_similarity_threshold: float, clip_model, clip_processor, device, video_path) -> Optional[Dict[int, List[str]]]:
    try:
        if clip_model is None or clip_processor is None:
            print(f"[{video_path}] CLIP model not available, skipping CLIP-based detection")
            return None
        
        # Get all image paths and sort them
        image_paths = sorted([
            os.path.join(images_dir, f) 
            for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])

        if not image_paths:
            print(f"[{video_path}] No images found in directory")
            return None

        print(f"[{video_path}] Processing {len(image_paths)} images with CLIP model...")

        # Extract features for all images
        embeddings = []
        for i, img_path in enumerate(image_paths):
            if i % 10 == 0:
                print(f"[{video_path}] Extracting features: {i}/{len(image_paths)}")
            
            features = get_features(img_path, device, clip_model, clip_processor, video_path)
            if features is None:
                print(f"[{video_path}] Failed to extract features from {img_path}")
                return None
            embeddings.append(features)

        # Detect scene boundaries based on similarity
        subscene_id = 0
        subscenes = {subscene_id: [image_paths[0]]}

        for i in range(1, len(embeddings)):
            sim = cosine_similarity(embeddings[i-1], embeddings[i], video_path)

            # If similarity drops below threshold, start new subscene
            if sim < clip_similarity_threshold: 
                subscene_id += 1
                subscenes[subscene_id] = []

            subscenes[subscene_id].append(image_paths[i])

        print(f"[{video_path}] CLIP detected {len(subscenes)} sub-scenes")
        return subscenes
        
    except Exception as e:
        print(f"[{video_path}] Failed in detect_subscenes_with_clip: {e}")
        return None

def save_subscenes(subscenes, output_dir, video_path):
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

    print(f"[{video_path}] \n✔ Final Sub-Scenes Saved →", output)
    return output

# ============================================================================
# FILE ORGANIZATION
# ============================================================================

def move_files_and_remove_subfolders(parent_folder: str, video_path) -> bool:
    try:
        if not os.path.exists(parent_folder):
            print(f"[{video_path}] Parent folder does not exist: {parent_folder}")
            return False
        
        print(f"[{video_path}] Moving files from subfolders to parent directory...")
        
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
                        print(f"[{video_path}] Skipped (already exists): {file}")
                        skipped_count += 1
                        continue

                    shutil.move(src, dst)
                    moved_count += 1
                    
                except Exception as e:
                    print(f"[{video_path}] Failed to move file {file}: {e}")

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
                print(f"[{video_path}] Failed to remove folder {root}: {e}")

        print(f"[{video_path}] File organization complete: {moved_count} moved, {skipped_count} skipped, {removed_count} folders removed")
        return True
        
    except Exception as e:
        print(f"[{video_path}] Critical error in move_files_and_remove_subfolders: {e}")
        return False

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(video_path, output_path) -> bool:
    """Main execution function with comprehensive error handling."""
    # ============================================================================
    # DETECTION PARAMETERS
    # ============================================================================

    # Interval (in seconds) at which frames are extracted from the video.
    # A smaller value gives more frames but increases processing time.
    extract_interval_seconds = 2

    # Minimum number of images to extract per detected scene.
    # Ensures at least a few representative frames even if a scene is short.
    min_images_per_scene = 3
    
    # Maximum number of images allowed per detected scene.
    # Prevents excessive frame extraction in long scenes.
    max_images_per_scene = 30

    # Histogram similarity threshold (0–1).
    # Frames with histogram similarity above this value are considered "too similar", and will be skipped to avoid duplicate visual content.
    similarity_threshold = 0.85
    
    # Structural Similarity Index (SSIM) threshold (0–1).
    # Measures structural similarity between frames. Higher threshold means only significantly different frames will be kept.
    structural_threshold = 0.85
    
    # Edge detection threshold (0–1).
    # Compares edge maps to eliminate frames with nearly identical edge structures.
    edge_threshold = 0.75

    # Minimum duration (in seconds) for a sub-scene.
    # Prevents extremely short or noisy sub-scene splits.
    min_subscene_duration = 2.5
    
    # CLIP model similarity threshold (0–1).
    # Used for semantic comparison of frames. Frames with similarity ABOVE this threshold are considered to belong to the same sub-scene.
    clip_similarity_threshold = 0.90
    
    # PySceneDetect command to use for initial scene detection.
    # 'detect-hash' = uses content-aware hashing for fast and accurate detection.
    scene_detect_command = 'detect-hash'
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # Load CLIP model
        clip_model, clip_processor = load_clip_model(device, video_path)
        
        images_dir = run_scenedetect(video_path, output_path, scene_detect_command)

        if images_dir is None:
            print(f"[{video_path}] Scene detection failed. Aborting.")
            return False
        
        print(f"[{video_path}] Scene detection completed. Images directory: {images_dir}")

        subscenes_info = extract_and_analyze_images(
            video_path, 
            output_path, 
            extract_interval_seconds,
            min_images_per_scene,
            max_images_per_scene,
            min_subscene_duration,
            similarity_threshold,
            structural_threshold,
            edge_threshold
        )

        if not subscenes_info:
            print(f"[{video_path}] No sub-scenes detected. Aborting.")
            return False

        video_name = os.path.splitext(os.path.basename(video_path))[0]

        if not rename_subscene_files(output_path, video_name, subscenes_info, video_path):
            print(f"[{video_path}] Some files failed to rename, but continuing...")

        if clip_model is not None:
            subscenes = detect_subscenes_with_clip(images_dir, clip_similarity_threshold, clip_model, clip_processor, device, video_path)
            
            if subscenes:
                final_scene_dir = save_subscenes(subscenes, output_path, video_path)
                
                if final_scene_dir:
                    if not move_files_and_remove_subfolders(final_scene_dir, video_path):
                        print(f"[{video_path}] File organization had issues, but continuing...")
                else:
                    print(f"[{video_path}] Failed to save CLIP subscenes, but continuing...")
            else:
                print(f"[{video_path}] CLIP detection failed, but basic detection completed")
        else:
            print(f"[{video_path}] CLIP model not available, skipping semantic detection")


        if os.path.exists(images_dir):
            shutil.rmtree(images_dir)

        # Move final scene folder
        scene_images = os.path.join(output_path, "scenes_images")
        os.rename(final_scene_dir, scene_images)
       
        return True

    except Exception as e:
        print(f"[{video_path}] Unexpected error in main: {e}")
        return False

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        # File paths
        video_path = r"D:\SDNA\Scene_Detector\scene_detection_samples\cristiano_ronaldo\free_kick\Cristiano_Ronaldos_Free_Kick.mp4"
        output_path = r"D:\SDNA\Scene_Detector\scene_detection_samples\cristiano_ronaldo\test"
        success = main(video_path, output_path)
    except Exception as e:
        print(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)