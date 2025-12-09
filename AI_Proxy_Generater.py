import os
import sys
import shutil
import csv
import cv2
import subprocess
import platform
import time
from datetime import datetime
from configparser import ConfigParser
import plistlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import uuid
import zipfile
import json
import argparse
import av
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import shutil
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

RUN_ID = str(uuid.uuid4()).upper()
DEBUG_CONFIG = {"DEBUG_FILEPATH" : f"/tmp/{RUN_ID}_proxy_generator_debug.out",
                "DEBUG_PRINT" : False,
                "DEBUG_TO_FILE" : False
                }

def create_config(config,processGuid):
    try:
        url = "http://192.168.3.44:5080/processCentral/config/upsert"

        if len(config_map['apikey']) == 0 or len(processGuid) ==0:
            debug_print("Skipping send progress.Failed to get required data.")
            return False

        header = { 'version' : '2', 'X-Server-Select' : 'api', 'apikey' : config_map['apikey'], 'Content-Type': 'application/json' }

        payload = {
            "processGuid": processGuid,
            "processName": "ProxyGenerator",
            "processType": "Transcode",
            "config": config
        }
            
        error = ""
        for i in range(5):
            try:    
                response = requests.post(url,headers=header,json=payload)
                error = response.text
                if response.status_code == 200:
                    return True
                else:
                    time.sleep(i ** 2)
            except Exception as e:
                time.sleep(i ** 2)
        else:        
            debug_print(f"Max try existed. Error :{error}")
            return False

    except Exception as e:
        debug_print(f"Failed to Send Data to Process Central:{e}")
        return False


def send_progress(status,description= None,file_path= None,percentage = None):
    try:
        url = "http://192.168.3.44:5080/processCentral/run/upsert"

        if len(config_map['apikey']) == 0 or len(config_map['process_id'])==0:
            debug_print("Skipping send progress.Failed to get required data.")
            return False

        header = { 'version' : '2', 'X-Server-Select' : 'api', 'apikey' : config_map['apikey'], 'Content-Type': 'application/json' }

        payload = {
            "processGuid": config_map['process_id'],
            "processName": config_map['process_name'],
            "processType": config_map['process_type'],
            "runGuid": RUN_ID,
            "hostname": "localhost",
            "status": status
        }

        if not percentage is None:
            payload['percentage'] = int(percentage)
        if not description is None:
            payload['description'] = description
        if not file_path is None:
            payload['detailedLogUrl'] = file_path
            
        error = ""
        for i in range(5):
            try:    
                response = requests.post(url,headers=header,json=payload)
                error = response.text
                if response.status_code == 200:
                    return True
                else:
                    time.sleep(i ** 2)
            except Exception as e:
                time.sleep(i ** 2)
        else:        
            debug_print(f"Max try existed. Error :{error}")
            return False

    except Exception as e:
        debug_print(f"Failed to Send Data to Process Central:{e}")
        return False


def loadConfigurationMap():
    config_map = {}

    is_linux=0
    if platform.system() == "Linux":
        DNA_CLIENT_SERVICES = '/etc/StorageDNA/DNAClientServices.conf'
        is_linux=1
    elif platform.system() == "Darwin":
        DNA_CLIENT_SERVICES = '/Library/Preferences/com.storagedna.DNAClientServices.plist'

    if not os.path.exists(DNA_CLIENT_SERVICES):
        print(f'Unable to find configuration file: {DNA_CLIENT_SERVICES}')
        return False
    
    nodeAPIKey = ""
    GFYServicePath = ""
    if is_linux == 1:
        config_parser = ConfigParser()
        config_parser.read(DNA_CLIENT_SERVICES)
        if config_parser.has_section('General') and config_parser.has_option('General','AiProxyConfigFolder') and config_parser.has_option('General','NodeAPIKey'):
            section_info = config_parser['General']
            nodeAPIKey = section_info['NodeAPIKey']
            GFYServicePath = section_info['AiProxyConfigFolder'] + "/AiProxy.conf"

    else:
        with open(DNA_CLIENT_SERVICES, 'rb') as fp:
            my_plist = plistlib.load(fp)
            nodeAPIKey= my_plist['NodeAPIKey']
            GFYServicePath = my_plist["AiProxyConfigFolder"] + "/AiProxy.conf"

    if not os.path.exists(GFYServicePath):
        err= "Unable to find cloud target file: " + GFYServicePath
        send_progress("Failed",err)
        sys.exit(err)

    config_parser = ConfigParser()
    config_parser.read(GFYServicePath)
    for section in config_parser.sections():
        config_map[section] = {}
        for key, value in config_parser.items(section):
            config_map[section][key] = value

    return config_map,nodeAPIKey

def debug_print(text_string):
    if DEBUG_CONFIG['DEBUG_PRINT'] == False:
        return

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    str = f"{formatted_datetime} {text_string}".strip()

    if DEBUG_CONFIG['DEBUG_TO_FILE'] == False:
        print(str)
    else:
        debug_file = open(DEBUG_CONFIG['DEBUG_FILEPATH'], "a",encoding='utf-8')
        debug_file.write(f'{str}\n')
        debug_file.close()

AUDIO_ALLOWED_FORMATS = {
    "mp3": {
        "codec": "libmp3lame",
        "ext": ".mp3"
    },
    "m4a": {
        "codec": "aac",
        "ext": ".m4a"
    },
    "wav": {
        "codec": "pcm_s16le",
        "ext": ".wav"
    }
}

# ============================================================================
# CLIP MODEL SETUP
# ============================================================================

def load_clip_model(device) -> Tuple[Optional[CLIPModel], Optional[CLIPProcessor]]:
    try:
      
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        debug_print(f"CLIP model loaded successfully")
        return model, processor,True
    except Exception as e:
        debug_print(f"Failed to load CLIP model: {e}")
        return None, None,f"Failed to load CLIP model: {e}"

# ============================================================================
# CLIP FEATURE EXTRACTION
# ============================================================================

def get_features(img_path: str, device, CLIP_MODEL, CLIP_PROCESSOR,video_path) -> Optional[torch.Tensor]:
    try:
        if CLIP_MODEL is None or CLIP_PROCESSOR is None:
            return None
            
        image = Image.open(img_path).convert("RGB")
        inputs = CLIP_PROCESSOR(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            features = CLIP_MODEL.get_image_features(**inputs)
        
        # Normalize features
        return features[0] / features[0].norm()
        
    except Exception as e:
        debug_print(f"[{video_path}] Failed to extract features from {img_path}: {e}")
        return None

def cosine_similarity(a: torch.Tensor, b: torch.Tensor,video_path) -> float:
    try:
        return torch.dot(a, b).item()
    except Exception as e:
        debug_print(f"[{video_path}] Failed to calculate cosine similarity: {e}")
        return 0.0

# ============================================================================
# SCENE DETECTION WITH SCENEDETECT
# ============================================================================

def run_scenedetect(video_path: str, output_dir: str, scene_detect_command: str) -> Optional[str]:
    try:
        images_dir = os.path.join(output_dir, "scenes_images")
        video_dir = os.path.join(output_dir, "scenes_videos")

        if scene_detect_command not in ['detect-content', 'detect-threshold', 'detect-hist', 'detect-adaptive', 'detect-hash']:
            return None, f"[{video_path}] Invalid scene detection command: {scene_detect_command}"
 
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
            debug_print(f"[{video_path}] Scene split output: {text}")
        
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
            debug_print(f"[{video_path}] Scene list output: {text}")

        return images_dir,True
        
    except subprocess.CalledProcessError as e:
        debug_print(f"[{video_path}] Scenedetect command failed: {e.stderr}")
        return None,f"[{video_path}] Scenedetect command failed: {e.stderr}"
    except FileNotFoundError:
        debug_print(f"[{video_path}] Scenedetect not found. Please install: pip install scenedetect[opencv]")
        return None,f"[{video_path}] Scenedetect not found. Please install: pip install scenedetect[opencv]"
    except Exception as e:
        debug_print(f"[{video_path}] Unexpected error in run_scenedetect: {e}")
        return None, f"[{video_path}] Unexpected error in run_scenedetect: {e}"

# ============================================================================
# IMAGE SIMILARITY CALCULATIONS
# ============================================================================

def calculate_histogram_similarity(img1: np.ndarray, img2: np.ndarray,video_path) -> float:
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
        debug_print(f"[{video_path}] Failed to calculate histogram similarity: {e}")
        return 0.0

def calculate_structural_similarity(img1: np.ndarray, img2: np.ndarray,video_path) -> float:
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
        debug_print(f"[{video_path}] Failed to calculate structural similarity: {e}")
        return 0.0

def calculate_edge_similarity(img1: np.ndarray, img2: np.ndarray,video_path) -> float:
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
        debug_print(f"[{video_path}] Failed to calculate edge similarity: {e}")
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
                
                hist_sim = calculate_histogram_similarity(prev_img, curr_img,video_path)
                struct_sim = calculate_structural_similarity(prev_img, curr_img,video_path)
                edge_sim = calculate_edge_similarity(prev_img, curr_img,video_path)
                
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
                debug_print(f"[{video_path}] Error calculating similarity for frame {i}: {e}")
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
                            
                            future_hist_sim = calculate_histogram_similarity(current_img, future_img,video_path)
                            
                            # If future frames are also different, might be transition effect
                            if future_hist_sim < 0.70:
                                confirmed = False
                                # debug_print(f"Potential transition at {sim['timestamp']:.2f}s - not confirmed")
                        except Exception as e:
                            debug_print(f"[{video_path}] Error in look-ahead confirmation: {e}")
                    
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
        debug_print(f"[{video_path}] Critical error in detect_subscenes_advanced: {e}")
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
        csv_path = os.path.join(images_dir, f"{base_name}-Scenes.csv")

        if not os.path.exists(csv_path):
            return [], f"[{video_path}] Scene CSV not found: {csv_path}"

        # ---- READ SCENE CSV ----
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            cleaned_csv = lines[1:]
            reader = csv.DictReader(cleaned_csv)
            scenes = list(reader)

        # ---- OPEN VIDEO ----
        container = av.open(video_path)
        stream = container.streams.video[0]
        fps = float(stream.average_rate)

        debug_print(f"[{video_path}] Extracting frames every {interval_seconds}s")

        all_subscenes_info = []

        # ---- FUNCTION: GET EXACT FRAME BY TIME ----
        def get_frame_at_time(time_sec):
            """Seek & decode frame accurately without RAM explosion."""
            tb = stream.time_base
            num = tb.numerator
            den = tb.denominator
            
            # convert time → pts
            target_pts = int(time_sec * den / num)

            container.seek(target_pts, any_frame=False, backward=True, stream=stream)

            for frame in container.decode(stream):
                if frame.pts is None:
                    continue

                frame_time = float(frame.pts * num / den)
                if frame_time >= time_sec:
                    return frame.to_ndarray(format="bgr24")

            return None

        # ---- PROCESS EACH SCENE ----
        for scene_idx, scene in enumerate(scenes, start=1):
            try:
                start_time = float(scene["Start Time (seconds)"])
                end_time = float(scene["End Time (seconds)"])
                duration = end_time - start_time

                # Number of images
                num_images = max(
                    min_images_per_scene,
                    min(max_images_per_scene, int(duration / interval_seconds) + 1)
                )

                frames_data = []
                time_cursor = start_time
                img_idx = 1

                while time_cursor <= end_time and len(frames_data) < num_images:
                    frame_img = get_frame_at_time(time_cursor)

                    if frame_img is None:
                        time_cursor += interval_seconds
                        continue

                    frames_data.append({
                        'image': frame_img,
                        'timestamp': time_cursor,
                        'index': img_idx
                    })

                    img_idx += 1
                    time_cursor += interval_seconds

                debug_print(f"[{video_path}] Extracted {len(frames_data)} frames from scene {scene_idx}")

                # ---- SUBSCENE DETECTION ----
                if frames_data:
                    subscene_bounds = detect_subscenes_advanced(
                        frames_data,
                        min_subscene_duration,
                        similarity_threshold,
                        structural_threshold,
                        edge_threshold,
                        video_path
                    )

                    for sub_idx, (start_i, end_i) in enumerate(subscene_bounds, start=1):
                        sub_start_time = frames_data[start_i]['timestamp']
                        sub_end_time = frames_data[end_i]['timestamp']
                        sub_duration = sub_end_time - sub_start_time

                        # Save images
                        for i in range(start_i, end_i + 1):
                            fd = frames_data[i]
                            img_arr = fd['image'][:, :, ::-1]  # BGR → RGB

                            img_name = (
                                f"{base_name}-Scene-{scene_idx:03d}-"
                                f"Sub-{sub_idx:02d}-{i-start_i+1:02d}.jpg"
                            )
                            img_path = os.path.join(images_dir, img_name)
                            Image.fromarray(img_arr, mode='RGB').save(img_path)

                        all_subscenes_info.append({
                            'scene_num': scene_idx,
                            'subscene_num': sub_idx,
                            'start_time': sub_start_time,
                            'end_time': sub_end_time,
                            'duration': sub_duration,
                            'num_images': end_i - start_i + 1,
                            'frames': frames_data[start_i:end_i + 1]
                        })

            except Exception as e:
                return [], f"[{video_path}] Scene {scene_idx} error: {e}"

        container.close()
        return all_subscenes_info, True

    except Exception as e:
        return [], f"[{video_path}] Critical error: {e}"


# ============================================================================
# FILE RENAMING
# ============================================================================

def format_timestamp(seconds: float,video_path) -> str:
    try:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        millisecs = int((secs - int(secs)) * 1000)
        return f"{hours:02d}-{minutes:02d}-{int(secs):02d}.{millisecs:03d}"
    except Exception as e:
        debug_print(f"[{video_path}] Failed to format timestamp {seconds}: {e}")
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
                    timestamp = format_timestamp(frame_data['timestamp'],video_path)
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
                            debug_print(f"[{video_path}] Failed to rename image {old_img}: {e}")
                    else:
                        debug_print(f"[{video_path}] Image file not found: {old_img}")
                        
            except Exception as e:
                debug_print(f"[{video_path}] Failed to rename files for scene {sub_info.get('scene_num')} sub {sub_info.get('subscene_num')}: {e}")
        
        debug_print(f"[{video_path}] Renamed {success_count}/{total_operations} files")
        return success_count == total_operations
        
    except Exception as e:
        debug_print(f"[{video_path}] Critical error in rename_subscene_files: {e}")
        return f"[{video_path}] Critical error in rename_subscene_files: {e}"

# ============================================================================
# CLIP-BASED SUBSCENE DETECTION
# ============================================================================

def detect_subscenes_with_clip(images_dir: str, clip_similarity_threshold: float, CLIP_MODEL, CLIP_PROCESSOR, device, video_path) -> Optional[Dict[int, List[str]]]:
    try:
        if CLIP_MODEL is None or CLIP_PROCESSOR is None:
            debug_print(f"[{video_path}] CLIP model not available, skipping CLIP-based detection")
            return None
        
        # Get all image paths and sort them
        image_paths = sorted([
            os.path.join(images_dir, f) 
            for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])

        if not image_paths:
            debug_print(f"[{video_path}] No images found in directory")
            return None

        debug_print(f"[{video_path}] Processing {len(image_paths)} images with CLIP model...")

        # Extract features for all images
        embeddings = []
        for i, img_path in enumerate(image_paths):
            if i % 10 == 0:
                debug_print(f"[{video_path}] Extracting features: {i}/{len(image_paths)}")
            
            features = get_features(img_path, device, CLIP_MODEL, CLIP_PROCESSOR,video_path)
            if features is None:
                debug_print(f"[{video_path}] Failed to extract features from {img_path}")
                return None
            embeddings.append(features)

        # Detect scene boundaries based on similarity
        subscene_id = 0
        subscenes = {subscene_id: [image_paths[0]]}

        for i in range(1, len(embeddings)):
            sim = cosine_similarity(embeddings[i-1], embeddings[i],video_path)

            # If similarity drops below threshold, start new subscene
            if sim < clip_similarity_threshold: 
                subscene_id += 1
                subscenes[subscene_id] = []

            subscenes[subscene_id].append(image_paths[i])

        debug_print(f"[{video_path}] CLIP detected {len(subscenes)} sub-scenes")
        return subscenes
        
    except Exception as e:
        debug_print(f"[{video_path}] Failed in detect_subscenes_with_clip: {e}")
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

    debug_print(f"[{video_path}] \n✔ Final Sub-Scenes Saved → {output}")
    return output

# ============================================================================
# FILE ORGANIZATION
# ============================================================================

def move_files_and_remove_subfolders(parent_folder: str, video_path) -> bool:
    try:
        if not os.path.exists(parent_folder):
            debug_print(f"[{video_path}] Parent folder does not exist: {parent_folder}")
            return f"[{video_path}] Parent folder does not exist: {parent_folder}"
        
        debug_print(f"[{video_path}] Moving files from subfolders to parent directory...")
        
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
                        debug_print(f"[{video_path}] Skipped (already exists): {file}")
                        skipped_count += 1
                        continue

                    shutil.move(src, dst)
                    moved_count += 1
                    
                except Exception as e:
                    debug_print(f"[{video_path}] Failed to move file {file}: {e}")

        # Remove empty subfolders
        removed_count = 0
        for root, dirs, files in os.walk(parent_folder, topdown=False):
            # Skip the parent folder itself
            if root == parent_folder:
                continue
                
            try:
                if not os.listdir(root):  # Folder is empty
                    os.rmdir(root)
                    # debug_print(f"Removed empty folder: {os.path.basename(root)}")
                    removed_count += 1
            except Exception as e:
                debug_print(f"[{video_path}] Failed to remove folder {root}: {e}")

        debug_print(f"[{video_path}] File organization complete: {moved_count} moved, {skipped_count} skipped, {removed_count} folders removed")
        return True
        
    except Exception as e:
        debug_print(f"[{video_path}] Critical error in move_files_and_remove_subfolders: {e}")
        return f"[{video_path}] Critical error in move_files_and_remove_subfolders: {e}"


def generate_scene_video_images(video_path, output_path) -> bool:
    openCV_cfg = config_map.get('openCV', {})
    clip_cfg = config_map.get('clip_embedding_model', {})
    scene_cfg = config_map.get('pyscene_detect', {})
    extract_interval_seconds = float(openCV_cfg.get('extract_interval_seconds', 2))
    min_images_per_scene = int(openCV_cfg.get('min_images_per_scene', 3))
    max_images_per_scene = int(openCV_cfg.get('max_images_per_scene', 30))
    similarity_threshold = float(openCV_cfg.get('similarity_threshold', 0.85))
    structural_threshold = float(openCV_cfg.get('structural_threshold', 0.85))
    edge_threshold = float(openCV_cfg.get('edge_threshold', 0.75))
    min_subscene_duration = float(openCV_cfg.get('min_subscene_duration', 2.5))
    clip_similarity_threshold = float(clip_cfg.get('clip_similarity_threshold', 0.90))
    scene_detect_command = scene_cfg.get('scene_detect_command') or 'detect-hash'
    gpu = int(clip_cfg.get('gpu', 0))
    final_scene_dir = None

    device = "cuda" if gpu == 1 and torch.cuda.is_available() else "cpu"
    try:
      
        images_dir,error_msg = run_scenedetect(video_path, output_path, scene_detect_command)

        if images_dir is None:
            debug_print(f"[{video_path}] Scene detection failed. Aborting.")
            return error_msg
        
        debug_print(f"[{video_path}] Scene detection completed. Images directory: {images_dir}")

        subscenes_info,subscenes_status_msg = extract_and_analyze_images(
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
            debug_print(f"[{video_path}] No sub-scenes detected. Aborting.")
            return subscenes_status_msg

        video_name = os.path.splitext(os.path.basename(video_path))[0]

        if not rename_subscene_files(output_path, video_name, subscenes_info, video_path):
            debug_print(f"[{video_path}] Some files failed to rename, but continuing...")
            
        if CLIP_MODEL is not None:
            subscenes = detect_subscenes_with_clip(images_dir, clip_similarity_threshold, CLIP_MODEL, CLIP_PROCESSOR, device, video_path)
            
            if subscenes:
                final_scene_dir = save_subscenes(subscenes, output_path, video_path)
                
                if final_scene_dir:
                    if not move_files_and_remove_subfolders(final_scene_dir, video_path):
                        debug_print(f"[{video_path}] File organization had issues, but continuing...")
                else:
                    debug_print(f"[{video_path}] Failed to save CLIP subscenes, but continuing...")
            else:
                debug_print(f"[{video_path}] CLIP detection failed, but basic detection completed")
        else:
            debug_print(f"[{video_path}] CLIP model not available, skipping semantic detection")

        scene_images = os.path.join(output_path, "scenes_images")
 
        if final_scene_dir and os.path.exists(final_scene_dir):
            if os.path.exists(images_dir):
                shutil.rmtree(images_dir)
            os.rename(final_scene_dir, scene_images)
        else:
            print(f"[{video_path}] No final scene directory to move.")
       
        return True

    except Exception as e:
        debug_print(f"[{video_path}] Unexpected error in main: {e}")
        return f"[{video_path}] Unexpected error in main: {e}"

def normalize_row(row):
    return {k.strip(): v for k, v in row.items()}

def ffprobe_info(input_file):
    info = {}
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,r_frame_rate,display_aspect_ratio",
            "-of", "default=noprint_wrappers=1",
            input_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        for line in result.stdout.strip().splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            info[key.strip()] = value.strip()

        return info
    except Exception as e:
        debug_print(f"Error while runnig ffprobe: {e}")
        return info
    
def generate_highres_from_r3d(file_path,config_map):
    gpu_enabled = int(config_map['red']['gpu'])
    new_file_path = os.path.splitext(file_path)[0]
    if len(config_map["cache_path"])!= 0:
        new_file_path = f"{config_map['cache_path']}/{new_file_path}"
    os.makedirs(new_file_path,exist_ok=True)
    command = [
        "REDline",
        "-i", file_path,
        "-o", new_file_path,
        "--format", "201",
        "--PRcodec", "0"
    ]

    if gpu_enabled == 1:
        command += [
            "--gpuPlatform", "2",
            "--enableGpuDecode",
            "--cudaDeviceIndexes", "0",
            "--numOclStreams", "8",
            "--decodeThreads", "128",
            "--verbose"
        ]

    debug_print(f"[{file_path}] Starting REDline command:{' '.join(command)}\n")
    result = subprocess.Popen(command,stdout=subprocess.PIPE, stderr=subprocess.STDOUT,bufsize=0)
    while True:
        chunk = result.stdout.read(4096)
        if not chunk:
            break

        text = chunk.decode("utf-8", errors="ignore")
        debug_print(f"[{file_path}] {text}")

    result.wait()
    debug_print(f"[{file_path}] Redline exited with code: {result.returncode}\n")
    if result.returncode != 0:
        return result.stderr.decode(errors='ignore')
    return True

def generate_proxy(src_path, dest_path,transcode_output,proxy_params,config_map):
    dest_file_name = os.path.basename(dest_path)
    desf_file_path = os.path.dirname(dest_path)
    tmp_file_name = f"generating_{dest_file_name}"
    tmp_full_path = os.path.join(desf_file_path,tmp_file_name)
    tmp_file_support = False
    r3d_result = False
    gpu_enabled = int(config_map.get('ffmpeg', {}).get('gpu', 0))
    proxy_frame_rate = config_map.get('video_proxy', {}).get('proxy_frame_rate', '')
    proxy_frame_size = config_map.get('video_proxy', {}).get('proxy_frame_size', '')
    proxy_aspect_ratio = config_map.get('video_proxy', {}).get('proxy_aspect_ratio', '')
    proxy_audio = config_map.get('video_proxy', {}).get('proxy_audio', '')
    proxy_resolution = config_map.get('video_proxy', {}).get('proxy_resolution', 1080)
    proxy_sprite_interval = config_map.get('frame_interval_proxy', {}).get('proxy_sprite_interval', '5')
    audio_proxy_output_type = config_map.get('audio_proxy', {}).get('proxy_output_type', 'mp3')
    audio_proxy_bitrate = config_map.get('audio_proxy', {}).get('proxy_bitrate', '320k')


    command = []
    if transcode_output == "video":
        tmp_file_support = True
        
        if os.path.splitext(src_path)[1].lower() == ".r3d":
            r3d_result = generate_highres_from_r3d(src_path,config_map)
            if r3d_result == True:
                src_path = os.path.splitext(src_path)[0] + ".mov"
                if len(config_map["cache_path"])!= 0:
                    src_path = f"{config_map['red']['cache_path']}/{src_path}"
                dest_path_parts = os.path.splitext(dest_path)
                if dest_path_parts[0].endswith("_001"):
                    dest_path = dest_path_parts[0][:-4] + dest_path_parts[1]
            else:
                return r3d_result
        if len(proxy_params)!=0:
            command = [
                    "ffmpeg", "-y",
                    "-i", str(src_path),
                    *proxy_params.split(),
                    str(tmp_full_path)
                ]
        else:
            info = ffprobe_info(src_path)
            frame_rate = proxy_frame_rate if len(proxy_frame_rate)!= 0 else eval(info.get('frame_rate', '25'))
            frame_size = proxy_frame_size if len(proxy_frame_size)!= 0 else f"{info.get('width', '1920')}:{info.get('height', '1080')}"
            width,height = frame_size.split(":")
            aspect_ratio = proxy_aspect_ratio if len(proxy_aspect_ratio)!=0 else info.get('display_aspect_ratio', '16:9').replace(":","/")
            audio = int(proxy_audio) if len(proxy_audio)!= 0  else 0

            if gpu_enabled == 1:
                vf_filter = f"hwupload_cuda,scale_cuda={width}:-1:format=nv12,hwdownload,format=nv12,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setpts={frame_rate}/25*PTS,setdar={aspect_ratio}"
                command = [
                    "ffmpeg",
                    "-hwaccel", "cuda",
                    "-i", str(src_path),
                    "-c:v", "h264_nvenc",
                    "-preset", "p3",
                    "-r", str(frame_rate),
                    "-vf", vf_filter,
                    "-movflags", "+faststart"
                ]
                if audio == 0:
                    command.append("-an")
                command.append(str(tmp_full_path))
            else:
                if len(proxy_frame_rate)!= 0 or len(proxy_frame_size)!= 0 or len(proxy_aspect_ratio)!=0 or len(proxy_audio)!= 0 :
                    command = [
                        "ffmpeg",
                        "-i", str(src_path),
                        "-c:v", "libx264",
                        "-preset", "veryfast",
                        "-r", str(frame_rate),
                        "-vf", f"scale={width}:-1,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setpts={frame_rate}/25*PTS,setdar={aspect_ratio}",
                        "-movflags", "+faststart",
                    ]
                    if audio == 0:
                        command.append("-an")
                    command.append(str(tmp_full_path))

                else:
                    command = [
                    "ffmpeg",
                    "-y",
                    "-i", str(src_path),
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-crf", "28",
                    "-c:a", "aac",
                    "-b:a", "128k",
                    "-vf", f"""scale='if(gt(ih,{int(proxy_resolution)}),-2,iw)':'if(gt(ih,{int(proxy_resolution)}),{int(proxy_resolution)},ih)',mpdecimate""",
                    "-vsync", "vfr",
                    str(tmp_full_path)
                ]

    elif transcode_output == "audio":
        tmp_file_support = True
        codec = AUDIO_ALLOWED_FORMATS[audio_proxy_output_type]["codec"]
        command = [
        "ffmpeg",
        "-y",
        "-i", str(src_path),
        "-vn",
        "-acodec", codec,
        ]

        if audio_proxy_output_type in ["mp3", "m4a"]:
            command += ["-b:a", audio_proxy_bitrate]

        command += [str(tmp_full_path)]
    
    elif transcode_output == "frame_interval":
        file_name_without_ext = os.path.splitext(dest_file_name)[0]
        image_path = f"{desf_file_path}/{file_name_without_ext}"
        if os.path.exists(image_path):
            shutil.rmtree(image_path)
        os.makedirs(image_path,exist_ok=True)
        patten_path = f"{image_path}/{file_name_without_ext}_%03d.jpg"
        tmp_file_support = False
        command = [
            "ffmpeg",
            "-i", str(src_path),
            "-vf", f"fps=1/{proxy_sprite_interval}",
            "-q:v", "2",
            str(patten_path)]
    
    elif transcode_output == "frame_scene":
        file_name_without_ext = os.path.splitext(dest_file_name)[0]
        path_without_ext = f"{desf_file_path}/{file_name_without_ext}"
        return generate_scene_video_images(src_path,path_without_ext)

    if not command:
        return f"No valid transcode command for output: {transcode_output}"
    
    debug_print(f"[{src_path}] Starting ffmpeg command:{' '.join(command)}\n")

    result = subprocess.Popen(command,stdout=subprocess.PIPE, stderr=subprocess.STDOUT,bufsize=0)
    while True:
        chunk = result.stdout.read(4096)
        if not chunk:
            break

        text = chunk.decode("utf-8", errors="ignore")
        debug_print(f"[{src_path}] {text}")

    result.wait()
    debug_print(f"[{src_path}] ffmpeg exited with code: {result.returncode}\n")

    if result.returncode != 0:
        if transcode_output == "image" or transcode_output == "sprite" or transcode_output == "scene":
            shutil.rmtree(image_path)
        if r3d_result == True:
            os.remove(src_path)
        if os.path.exists(tmp_full_path) and tmp_file_support == True:
            os.remove(tmp_full_path)
        debug_print(f"Error transcoding:{src_path}\nError :{result.stderr}")
        return result.stderr.decode(errors='ignore')
    if tmp_file_support == True:
        if os.path.exists(tmp_full_path):
            if transcode_output == "scene" or transcode_output == "image" or transcode_output == "face":
                shutil.rmtree(image_path)
            if r3d_result == True:
                os.remove(src_path)
            os.rename(tmp_full_path,dest_path)
        else:
            return "proxy_not_found"
    return True

def create_lck(file_path):
    lck_path = f"{file_path}.lck"
    with open(lck_path, 'w'):
        pass
    return lck_path

def remove_lck(lck_path):
    try:
        os.remove(lck_path)
    except Exception:
        pass

def move_file_to_completed(input_path,completed_file):
    completed_path = os.path.join(input_path,".completed")
    os.makedirs(completed_path,exist_ok=True)
    shutil.move(completed_file, completed_path)
    debug_print(f"Moved to completed: {completed_file} -> {completed_path}")

def move_failed_file(input_path,failed_file):
    failed_path = os.path.join(input_path,".failed")
    os.makedirs(failed_path,exist_ok=True)
    shutil.move(failed_file, failed_path)
    debug_print(f"Moved to Failed: {failed_file} -> {failed_path}")

def delete_file(input_path,file_path,is_failed,error_message=None):
    try:
        os.remove(file_path)
        debug_print(f"File Deleted Succesfully: {file_path}")
    except Exception as e:
        debug_print(f"Failed to Delete File {file_path} : {e}")
    if is_failed == True:
        create_failed_file(input_path,file_path,error_message)

def create_failed_file(input_path,file_path,error_message):
    file_name = os.path.basename(os.path.normpath(file_path))
    failed_path = os.path.join(input_path,".failed")
    os.makedirs(failed_path,exist_ok=True)
    failed_file_path = os.path.join(failed_path,file_name)
    with open(failed_file_path,'w') as f:
        f.write(error_message)

def file_changed(file_info):
    try:
        stat = os.stat(file_info["path"])
        if stat.st_size == file_info["size"] and stat.st_mtime == file_info["mtime"]:
            return False
        else:
            return True
    except Exception as e:
        return True

def get_csv_file_list(folder_path):
    csv_file_list = []
    try:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                if file_path.lower().endswith('.csv'):
                    csv_file_list.append(file_path)
                elif file_path.lower().endswith('.zip'):
                    output_path = os.path.dirname(os.path.normpath(file_path))
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(output_path)
                    for root, _, files in os.walk(output_path):
                        for file in files:
                            if file.lower().endswith('.csv'):
                                csv_file_list.append(os.path.join(root, file))
                    os.remove(file_path)
    except Exception as e:
        debug_print(f"Failed to get CSVs from folder:{folder_path} Error: {e}")
        return []
    return list(set(csv_file_list))

def scan_and_generate_proxy_files(config_map):
    debug_print("Started")
    send_progress("Started")
    completed = 0
    failed = 0
    include_ext_set = set()
    exclude_ext_set = set()
    input_path = config_map['General']['source_path']
    output_path = config_map['General']['target_path']
    include_ext = config_map['General']['include']
    exclude_ext = config_map['General']['exclude']
    proxy_params = config_map['ffmpeg']['params']
    output_res = config_map['video_proxy']['proxy_resolution'] if len(config_map['video_proxy']['proxy_resolution']) != 0 else '1080'
    mode = config_map['General']['output_mode'].lower() 
    threads = int(config_map['General']['threads']) if len(config_map['General']['threads']) != 0 else 5
    source_type = config_map['General']['source_type'].lower() 
    transcode_app = config_map['General']['transcode_app'].lower() 
    transcode_output_list = config_map['General']['transcode_output'].split(",")
    log_level = int(config_map['General']['log_level']) if len(config_map['General']['log_level']) != 0 else 0
    if log_level == 1:
        DEBUG_CONFIG['DEBUG_PRINT'] = True
        DEBUG_CONFIG['DEBUG_TO_FILE'] = True

    required_fields = {
    "Source_path": input_path,
    "Target_path": output_path,
    "Source_type": source_type,
    "Transcode_app": transcode_app,
    "Transcode_output": transcode_output_list,
    }

    send_progress("Processing")
    debug_print("Processing")
    missing = []
    for name, value in required_fields.items():
        if len(value) == 0:
            missing.append(name)
    if len(missing) != 0:
        print(f"Missing required options: {', '.join(missing)}")
        send_progress("Failed",f"Missing required options: {', '.join(missing)}")
        debug_print(f"Missing required options: {', '.join(missing)}")
        exit(1)

    if source_type == "folder":
        if len(mode) == 0:
            print("Missing required option: Output_mode (needed when Source_type is Folder).")
            send_progress("Failed","Missing required option: Output_mode (needed when Source_type is Folder).")
            debug_print("Missing required option: Output_mode (needed when Source_type is Folder).")
            exit(1)

    allowed_values = {
    "Source_type": (source_type, ["folder", "csv"]),
    "Transcode_app": (transcode_app, ["auto"]),
    "Transcode_output": (transcode_output_list, ["video", "audio", "frame_interval", "frame_scene"]),
    "Log_Level": (log_level, [0, 1]),
    }

    def fail_invalid(name, value, allowed):
        msg = f"Invalid {name}: {value}. Allowed values are: {', '.join(allowed)}."
        print(msg)
        send_progress("Failed", msg)
        debug_print(msg)
        exit(1)

    for name, (value, allowed) in allowed_values.items():
        values = value if isinstance(value, list) else [value]
        for v in values:
            if v not in allowed:
                fail_invalid(name, v, allowed)

    if source_type == "folder":
        if mode not in ["move", "delete","keep"]:
            print(f"Invalid Output_mode: {mode}. Allowed values are: move, delete,keep.")
            send_progress("Failed",f"Invalid Output_mode: {mode}. Allowed values are: move, delete,keep.")
            debug_print(f"Invalid Output_mode: {mode}. Allowed values are: move, delete,keep.")
            exit(1)

    for transcode_output in transcode_output_list:
        if transcode_output == "video" and len(output_res) == 0:
            print("Missing required option: Output_res (needed when generating video proxy).")
            send_progress("Failed","Missing required option: Output_res (needed when generating video proxy).")
            debug_print("Missing required option: Output_res (needed when generating video proxy).")
            exit(1)   

        if transcode_output == "frame_interval" and len(config_map['frame_interval_proxy']['proxy_sprite_interval']) == 0:
            print("Missing required option: Sprite_time (needed to generate the Frames).")
            send_progress("Failed","Missing required option: Sprite_time (needed to generate the Frames).")
            debug_print("Missing required option: Sprite_time (needed to generate the Frames).")
            exit(1)    

    if len(include_ext) != 0:
        include_ext_set = set(include_ext.split(","))

    if len(exclude_ext) != 0:
        exclude_ext_set = set(exclude_ext.split(","))
    files = []
    processed_rdc_folders = set()
    if source_type == "folder":
        for root, dirs, filenames in os.walk(input_path):
            dirs[:] = [d for d in dirs if d not in {".completed", ".failed"}]

            for f in filenames:
                if f.endswith(".lck"):
                    continue
                full_path = os.path.join(root, f)
                if os.path.exists(f"{full_path}.lck"):
                    continue

                ext = os.path.splitext(f)[1].lower()
                if not ext:
                    continue
                
                if include_ext_set and ext not in include_ext_set:
                    continue
                if ext in exclude_ext_set:
                    continue

                if ext == ".r3d":
                    rdc_folder = os.path.dirname(full_path)
                    if rdc_folder.lower().endswith(".rdc"):

                        if rdc_folder in processed_rdc_folders:
                            continue

                        processed_rdc_folders.add(rdc_folder)

                        all_r3d = sorted(
                            file for file in os.listdir(rdc_folder)
                            if file.lower().endswith(".r3d")
                        )
                        if not all_r3d:
                            continue
                        first_r3d_path = os.path.join(rdc_folder, all_r3d[0])
                        if os.path.exists(f"{first_r3d_path}.lck"):
                            continue
                        full_path = first_r3d_path
                        debug_print(f"Detected RED sequence: {full_path}")

                create_lck(full_path)
                stat = os.stat(full_path)
                files.append({
                    "path": os.path.normpath(full_path),
                    "mtime": stat.st_mtime,
                    "size": stat.st_size
            })

    elif source_type == "csv":
        filenames = get_csv_file_list(input_path)
        for filename in filenames:
            if filename.lower().endswith('.csv'):
                with open(filename, newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        row = normalize_row(row)
                        file_path = row['Filename']
                        if os.path.exists(file_path) and not file_path.endswith(".lck") and not os.path.exists(f"{file_path}.lck"):
                            ext = os.path.splitext(file_path)[1].lower()
                            if not ext:
                                continue
                            
                            if include_ext_set and ext not in include_ext_set:
                                continue
                            if ext in exclude_ext_set:
                                continue

                            if ext == ".r3d":
                                rdc_folder = os.path.dirname(file_path)
                                if rdc_folder.lower().endswith(".rdc"):

                                    if rdc_folder in processed_rdc_folders:
                                        continue

                                    processed_rdc_folders.add(rdc_folder)

                                    all_r3d = sorted(
                                        file for file in os.listdir(rdc_folder)
                                        if file.lower().endswith(".r3d")
                                    )
                                    if not all_r3d:
                                        continue
                                    first_r3d_path = os.path.join(rdc_folder, all_r3d[0])
                                    if os.path.exists(f"{first_r3d_path}.lck"):
                                        continue
                                    file_path = first_r3d_path
                                    debug_print(f"Detected RED sequence: {file_path}")

                            create_lck(file_path)
                            stat = os.stat(file_path)
                            files.append({"path": os.path.normpath(file_path), "mtime": stat.st_mtime, "size": stat.st_size})
                
                os.remove(filename)
    debug_print(f"Scanned Files {','.join([f['path'] for f in files])}")
    debug_print(f"Total File Scanned: {len(files)}")
    time.sleep(10)
    output_counts = {}
    progress_counts = {} 
    error_logs = {} 
    total_outputs_required = len(transcode_output_list)
    total_file_size = sum(int(f["size"]) for f in files)
    completed_file_size = 0
    completed = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = {}
        for file_info  in files:
            if file_changed(file_info) == True:
                debug_print(f"Skipping copying file: {file_info['path']}")
                remove_lck(f"{file_info['path']}.lck")
                continue

            file = file_info["path"]
            output_path = os.path.normpath(output_path)
            file_name = os.path.basename(file)
            for transcode_output in transcode_output_list:
                if transcode_app == "auto":
                    if transcode_output == "video":
                        file_name = os.path.splitext(file_name)[0] + ".mp4"
                    elif transcode_output == "audio":
                        audio_proxy_output_type = config_map['audio_proxy']['proxy_output_type']
                        if len(audio_proxy_output_type)!=0:
                            ext = AUDIO_ALLOWED_FORMATS[audio_proxy_output_type]['ext']
                            file_name = os.path.splitext(file_name)[0] + ext
                        else:
                            file_name = os.path.splitext(file_name)[0] + ".mp3"


                folder_path = os.path.normpath(f"{output_path}/{os.path.dirname(file)}")
                os.makedirs(folder_path,exist_ok=True)
                output_full_path = os.path.join(folder_path,file_name)

                if os.path.exists(output_full_path):
                    debug_print(f"Deleting exits file: {output_full_path}")
                    os.remove(output_full_path)
                
                if transcode_app == "auto":
                    futures[executor.submit(generate_proxy, file, output_full_path,transcode_output,proxy_params,config_map)] = file_info

        for future in as_completed(futures):
            source_file_details = futures[future]
            source_file = source_file_details['path']
            source_file_size = source_file_details['size']
            file_key = os.path.basename(source_file)
            lck_path = f"{source_file}.lck"

            progress_counts.setdefault(file_key, 0)
            output_counts.setdefault(file_key, 0)
            error_logs.setdefault(file_key, [])

            try:
                result = future.result()
                completed_file_size += source_file_size
                progress_percent = (completed_file_size / total_file_size) * 100
                send_progress("Processing",percentage=int(progress_percent))
                progress_counts[file_key] += 1
                debug_print(f"[{file_key}] Completed {progress_counts[file_key]}/{total_outputs_required}")

                if result is True:
                    output_counts[file_key] += 1
                else:
                    error_message = str(result).splitlines()[-1]
                    error_logs[file_key].append(error_message)
                    debug_print(f"[ERROR] {file_key}: {error_message}")
                    failed += 1

                if progress_counts[file_key] == total_outputs_required:

                    remove_lck(lck_path)

                    if output_counts[file_key] == total_outputs_required:
                        if mode == "move":
                            move_file_to_completed(input_path, source_file)
                        elif mode == "delete":
                            delete_file(input_path, source_file, False)
                        completed += 1
                    else:
                        combined_error = "\n".join(error_logs[file_key])
                        debug_print(f"[ERROR SUMMARY] {file_key}:\n{combined_error}")
                        if source_type == "folder":
                            if mode == "move":
                                move_failed_file(input_path, source_file)
                            elif mode == "delete":
                                delete_file(input_path, source_file, True, combined_error)
                        elif source_type == "csv":
                            create_failed_file(input_path, source_file, combined_error)

            except Exception as e:
                progress_counts[file_key] += 1
                error_msg = str(e)
                error_logs[file_key].append(error_msg)
                debug_print(f"[EXCEPTION] {file_key}: {error_msg}")
                failed += 1

                if progress_counts[file_key] == total_outputs_required:
                    remove_lck(lck_path)
                    combined_error = "\n".join(error_logs[file_key])
                    if source_type == "folder":
                        if mode == "move":
                            move_failed_file(input_path, source_file)
                        elif mode == "delete":
                            delete_file(input_path, source_file, True, combined_error)
                    elif source_type == "csv":
                        create_failed_file(input_path, source_file, combined_error)

    if log_level == 1:
        send_progress("Completed", f"COMPLETED|{completed}|FAILED|{failed}",DEBUG_CONFIG["DEBUG_FILEPATH"])
    else:
        send_progress("Completed", f"COMPLETED|{completed}|FAILED|{failed}")

    debug_print(f"Batch finished: COMPLETED={completed}, FAILED={failed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', '--process_id',default="A41B4F2D-1EFD-4EE4-8CA6-DE362CA70ECA",help='Process ID (Random UUID4)')
    args = parser.parse_args()

    config,api_key = loadConfigurationMap()
    config_map = config.copy()
    config_map["apikey"] = api_key
    config_map['process_id'] = args.process_id
    config_map['process_name'] = "ProxyGenerator"
    config_map['process_type'] = "Transcode"
    log_level = int(config_map['General']['log_level']) if len(config_map['General']['log_level']) != 0 else 0
    if log_level == 1:
        DEBUG_CONFIG['DEBUG_PRINT'] = True
        DEBUG_CONFIG['DEBUG_TO_FILE'] = True
    clip_cfg = config_map.get('clip_embedding_model', {})
    gpu = int(clip_cfg.get('gpu', 0))
    device = "cuda" if gpu == 1 and torch.cuda.is_available() else "cpu"
    CLIP_MODEL, CLIP_PROCESSOR,status_msg = load_clip_model(device)

    if create_config(config,args.process_id) != True:
        debug_print("Failed to Add config in Process Central.")
        print("Failed to Add config in Process Central.")
        exit(1)
    if not config_map:
        debug_print("Config Not Found.")
        send_progress("Failed",f"Config Not Found.")
        exit(1)

    scan_and_generate_proxy_files(config_map)
