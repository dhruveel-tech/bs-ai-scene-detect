import subprocess
import csv
import os
import time
import cv2
import numpy as np
from datetime import timedelta

video_path = r"D:\SDNA\Scene_Detector\scene_detection_samples\Real_Madrid\Real_Madrid.mp4"
output_dir = r"D:\SDNA\Scene_Detector\scene_subdivision\Real_Madrid"

# Configuration
EXTRACT_INTERVAL_SECONDS = 1.5  # Extract more frequently for better detection
MIN_IMAGES_PER_SCENE = 5
MAX_IMAGES_PER_SCENE = 100
SIMILARITY_THRESHOLD = 0.80  # Histogram similarity threshold
STRUCTURAL_THRESHOLD = 0.85  # SSIM threshold
EDGE_THRESHOLD = 0.75  # Edge detection threshold
MIN_SUBSCENE_DURATION = 2.5  # Minimum duration for a sub-scene


def run_scenedetect(video_path, output_dir):
    """Run scenedetect to split videos and generate scene CSV."""
    images_dir = os.path.join(output_dir, "scenes_images")
    video_dir = os.path.join(output_dir, "scenes_videos")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    print("🎬 Detecting scenes and splitting video...")
    subprocess.run([
        "scenedetect", "-i", video_path,
        "--output", video_dir,
        "detect-content",
        "split-video"
    ], check=True)

    print("📋 Generating scene list CSV...")
    subprocess.run([
        "scenedetect", "-i", video_path,
        "--output", images_dir,
        "detect-content",
        "list-scenes"
    ], check=True)


def calculate_histogram_similarity(img1, img2):
    """Calculate color histogram similarity."""
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    
    hist1 = cv2.calcHist([hsv1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    
    cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def calculate_structural_similarity(img1, img2):
    """Calculate structural similarity (SSIM) - detects layout changes."""
    # Resize for faster computation
    img1_small = cv2.resize(img1, (320, 240))
    img2_small = cv2.resize(img2, (320, 240))
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1_small, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_small, cv2.COLOR_BGR2GRAY)
    
    # Calculate mean squared error as a simple SSIM alternative
    mse = np.mean((gray1.astype(float) - gray2.astype(float)) ** 2)
    
    if mse == 0:
        return 1.0
    
    # Convert to similarity score (0-1 range)
    max_pixel_value = 255.0
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    similarity = min(1.0, psnr / 50.0)  # Normalize to 0-1
    
    return similarity


def calculate_edge_similarity(img1, img2):
    """Calculate edge detection similarity - detects composition changes."""
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


def detect_subscenes_advanced(frames_data, min_duration=2.5):
    """
    Advanced sub-scene detection using multiple metrics.
    Combines histogram, structural, and edge similarity.
    Uses look-ahead confirmation to avoid false positives.
    """
    if len(frames_data) < 2:
        return [(0, len(frames_data) - 1)]
    
    # Calculate all similarities
    similarities = []
    for i in range(1, len(frames_data)):
        prev_img = frames_data[i-1]['image']
        curr_img = frames_data[i]['image']
        
        hist_sim = calculate_histogram_similarity(prev_img, curr_img)
        struct_sim = calculate_structural_similarity(prev_img, curr_img)
        edge_sim = calculate_edge_similarity(prev_img, curr_img)
        
        # Weighted average (histogram is most important)
        combined_sim = (hist_sim * 0.5) + (struct_sim * 0.3) + (edge_sim * 0.2)
        
        similarities.append({
            'index': i,
            'histogram': hist_sim,
            'structural': struct_sim,
            'edge': edge_sim,
            'combined': combined_sim,
            'timestamp': frames_data[i]['timestamp']
        })
    
    # Find scene breaks with look-ahead confirmation
    subscenes = []
    current_start = 0
    i = 0
    
    while i < len(similarities):
        sim = similarities[i]
        frame_idx = sim['index']
        
        # Check if metrics indicate a change
        is_histogram_change = sim['histogram'] < SIMILARITY_THRESHOLD
        is_structural_change = sim['structural'] < STRUCTURAL_THRESHOLD
        is_edge_change = sim['edge'] < EDGE_THRESHOLD
        is_combined_change = sim['combined'] < 0.78
        
        change_votes = sum([is_histogram_change, is_structural_change, is_edge_change])
        
        if (change_votes >= 2) or is_combined_change:
            # Check duration of current sub-scene
            duration = frames_data[frame_idx - 1]['timestamp'] - frames_data[current_start]['timestamp']
            
            if duration >= min_duration:
                # Look ahead to confirm this is a sustained change (not just a flash/transition)
                confirmed = True
                if i + 2 < len(similarities):  # Check next 2 frames
                    # Compare frame_idx with frame_idx+2 to see if change persists
                    future_img = frames_data[min(frame_idx + 2, len(frames_data) - 1)]['image']
                    current_img = frames_data[frame_idx]['image']
                    
                    future_hist_sim = calculate_histogram_similarity(current_img, future_img)
                    
                    # If future frames are very similar to current, it's a real scene change
                    # If they're different again, it might be a transition/flash
                    if future_hist_sim < 0.70:  # Future is also different - might be transition
                        confirmed = False
                
                if confirmed:
                    # Split here: previous sub-scene ends at frame_idx-1
                    subscenes.append((current_start, frame_idx - 1))
                    print(f"   🔍 Break at {sim['timestamp']:.2f}s | Hist:{sim['histogram']:.2f} Struct:{sim['structural']:.2f} Edge:{sim['edge']:.2f} → Combined:{sim['combined']:.2f}")
                    current_start = frame_idx  # New sub-scene starts at the changed frame
        
        i += 1
    
    # Add the last sub-scene
    subscenes.append((current_start, len(frames_data) - 1))
    
    # Post-process: merge very short sub-scenes with neighbors
    merged_subscenes = []
    for i, (start, end) in enumerate(subscenes):
        duration = frames_data[end]['timestamp'] - frames_data[start]['timestamp']
        
        if duration < min_duration and len(merged_subscenes) > 0:
            # Merge with previous sub-scene
            prev_start, prev_end = merged_subscenes[-1]
            merged_subscenes[-1] = (prev_start, end)
            print(f"   ⚠️  Merged short sub-scene ({duration:.1f}s) with previous")
        else:
            merged_subscenes.append((start, end))
    
    return merged_subscenes if merged_subscenes else subscenes


def calculate_average_frame(frames, num_samples=5):
    """Calculate average frame from multiple samples to reduce noise."""
    if len(frames) <= num_samples:
        sample_frames = frames
    else:
        step = len(frames) // num_samples
        sample_frames = [frames[i * step] for i in range(num_samples)]
    
    avg_frame = np.mean([f.astype(np.float32) for f in sample_frames], axis=0)
    return avg_frame.astype(np.uint8)


def extract_and_analyze_images(video_path, output_dir, interval_seconds=1.5):
    """Extract images and detect sub-scenes with advanced analysis."""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    images_dir = os.path.join(output_dir, "scenes_images")
    video_dir = os.path.join(output_dir, "scenes_videos")
    subscenes_dir = os.path.join(output_dir, "subscenes_videos")
    
    os.makedirs(subscenes_dir, exist_ok=True)
    
    csv_path = os.path.join(images_dir, f"{base_name}-Scenes.csv")
    
    # Read CSV
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        cleaned_csv = lines[1:]
        reader = csv.DictReader(cleaned_csv)
        scenes = list(reader)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\n📹 Video FPS: {fps:.2f}")
    print(f"⏱️  Extracting images every {interval_seconds} seconds")
    print(f"🔍 Multi-metric detection enabled\n")
    
    all_subscenes_info = []
    
    for scene_idx, scene in enumerate(scenes, start=1):
        start_frame = int(scene["Start Frame"])
        end_frame = int(scene["End Frame"])
        start_time = float(scene["Start Time (seconds)"])
        end_time = float(scene["End Time (seconds)"])
        scene_duration = float(scene["Length (seconds)"])
        
        print(f"🎬 Scene {scene_idx:03d}: Duration {scene_duration:.2f}s")
        
        # Calculate number of images
        num_images = max(MIN_IMAGES_PER_SCENE, 
                        min(MAX_IMAGES_PER_SCENE, 
                            int(scene_duration / interval_seconds) + 1))
        
        # Extract frames
        frames_data = []
        current_time = start_time
        img_idx = 1
        
        while current_time <= end_time and len(frames_data) < num_images:
            frame_num = int(start_frame + ((current_time - start_time) * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if ret:
                frames_data.append({
                    'image': frame,
                    'frame_num': frame_num,
                    'timestamp': current_time,
                    'index': img_idx
                })
                img_idx += 1
            
            current_time += interval_seconds
        
        # Detect sub-scenes with advanced analysis
        if len(frames_data) > 0:
            subscene_boundaries = detect_subscenes_advanced(frames_data, MIN_SUBSCENE_DURATION)
            
            print(f"   📊 Divided into {len(subscene_boundaries)} sub-scenes")
            
            # Save images and info for each sub-scene
            for sub_idx, (start_idx, end_idx) in enumerate(subscene_boundaries, start=1):
                sub_start_time = frames_data[start_idx]['timestamp']
                sub_end_time = frames_data[end_idx]['timestamp']
                sub_duration = sub_end_time - sub_start_time
                
                # Select exactly 3 representative frames from the sub-scene
                total_frames = end_idx - start_idx + 1
                if total_frames <= 3:
                    # Use all available frames if 3 or fewer
                    selected_indices = list(range(start_idx, end_idx + 1))
                else:
                    # Select 3 evenly distributed frames: start, middle, end
                    selected_indices = [
                        start_idx,                          # First frame
                        start_idx + (end_idx - start_idx) // 2,  # Middle frame
                        end_idx                             # Last frame
                    ]
                
                print(f"      ├─ Sub {sub_idx}: {sub_start_time:.1f}s → {sub_end_time:.1f}s ({sub_duration:.1f}s, 3 images)")
                
                # Save only the 3 selected images
                selected_frames = []
                for img_num, idx in enumerate(selected_indices, start=1):
                    frame_data = frames_data[idx]
                    img_name = f"{base_name}-Scene-{scene_idx:03d}-Sub-{sub_idx:02d}-{img_num:02d}.jpg"
                    img_path = os.path.join(images_dir, img_name)
                    cv2.imwrite(img_path, frame_data['image'])
                    selected_frames.append(frame_data)
                
                # Store sub-scene info with only 3 frames
                all_subscenes_info.append({
                    'scene_num': scene_idx,
                    'subscene_num': sub_idx,
                    'start_time': sub_start_time,
                    'end_time': sub_end_time,
                    'duration': sub_duration,
                    'num_images': len(selected_frames),
                    'frames': selected_frames
                })
    
    cap.release()
    
    # Don't split videos here - will do it later based on images
    print(f"\n✅ Analyzed {len(scenes)} scenes into {len(all_subscenes_info)} sub-scenes")
    return all_subscenes_info


def format_timestamp(seconds):
    """Convert seconds to HH-MM-SS.mmm format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    millisecs = int((secs - int(secs)) * 1000)
    return f"{hours:02d}-{minutes:02d}-{int(secs):02d}.{millisecs:03d}"


def extract_videos_from_images(video_path, output_dir, base_name, subscenes_info):
    """Extract video clips based on saved image timestamps."""
    subscenes_dir = os.path.join(output_dir, "subscenes_videos")
    os.makedirs(subscenes_dir, exist_ok=True)
    
    print("\n✂️  Extracting video clips based on image timestamps...\n")
    
    for sub_info in subscenes_info:
        scene_num = sub_info['scene_num']
        subscene_num = sub_info['subscene_num']
        frames = sub_info['frames']
        
        # Get start and end times from first and last images
        start_time = frames[0]['timestamp']
        end_time = frames[-1]['timestamp']
        duration = end_time - start_time
        
        # Add small buffer for better video capture (0.1s before and after)
        buffer = 0.1
        video_start = max(0, start_time - buffer)
        video_end = end_time + buffer
        
        start_tc = format_timestamp(start_time)
        end_tc = format_timestamp(end_time)
        
        output_video = os.path.join(
            subscenes_dir,
            f"{base_name}-Scene-{scene_num:03d}-Sub-{subscene_num:02d}_{start_tc}_to_{end_tc}.mp4"
        )
        
        # Use ffmpeg with re-encoding for precise frame extraction
        try:
            subprocess.run([
                "ffmpeg",
                "-i", video_path,
                "-ss", str(video_start),
                "-to", str(video_end),
                "-c:v", "libx264",      # Re-encode video for precision
                "-c:a", "aac",          # Re-encode audio
                "-b:v", "2M",           # Video bitrate
                "-preset", "fast",       # Encoding speed
                "-y",                    # Overwrite
                output_video
            ], 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL,
            check=True)
            
            print(f"   ✅ Scene-{scene_num:03d}-Sub-{subscene_num:02d}: {start_tc} → {end_tc} ({duration:.1f}s, {len(frames)} frames)")
        
        except subprocess.CalledProcessError:
            print(f"   ❌ Failed to extract Scene-{scene_num:03d}-Sub-{subscene_num:02d}")
    
    print(f"\n✅ Extracted {len(subscenes_info)} video clips")


def rename_subscene_files(output_dir, base_name, subscenes_info):
    """Rename image files with timestamps."""
    images_dir = os.path.join(output_dir, "scenes_images")
    
    print("\n🔄 Renaming image files with timestamps...\n")
    
    for sub_info in subscenes_info:
        scene_num = sub_info['scene_num']
        subscene_num = sub_info['subscene_num']
        
        # Rename images
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
            
            if os.path.exists(old_img):
                os.rename(old_img, new_img)


# ---------------------------
# RUN EVERYTHING
# ---------------------------
if __name__ == "__main__":
    start_time = time.time()
    
    print("="*60)
    print("🎥 ADVANCED SCENE SUBDIVISION SYSTEM")
    print("="*60)
    
    # Step 1: Detect scenes
    run_scenedetect(video_path, output_dir)
    
    # Step 2: Extract images and detect sub-scenes
    subscenes_info = extract_and_analyze_images(video_path, output_dir, EXTRACT_INTERVAL_SECONDS)
    
    # Step 3: Rename image files with timestamps
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    rename_subscene_files(output_dir, video_name, subscenes_info)
    
    # Step 4: Extract video clips based on image timestamps
    extract_videos_from_images(video_path, output_dir, video_name, subscenes_info)
    
    end_time = time.time()
    print(f"\n{'='*60}")
    print(f"⏱️  Total time: {end_time - start_time:.2f} seconds")
    print(f"✅ Processing complete!")
    print(f"{'='*60}")