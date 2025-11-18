import subprocess
import csv
import os
import time
from datetime import timedelta

video_path = r"scene_detection_samples/Real_Madrid/Real_Madrid.mp4"
output_dir = r"D:\SDNA\Scene_Detector\test"

def run_scenedetect(video_path, output_dir, num_images_per_scene=None):
    """
    Run scenedetect with dynamic image extraction.
    If num_images_per_scene is None, it will be calculated based on scene duration.
    """
    images_dir = os.path.join(output_dir, "scenes_images")
    video_dir = os.path.join(output_dir, "scenes_videos")

    # Create folders if not exist
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    # Run scenedetect to split video
    subprocess.run([
        "scenedetect", "-i", video_path,
        "--output", video_dir,
        "split-video"
    ], check=True)

    # If dynamic image count, we need to process scenes individually
    if num_images_per_scene is None:
        # First get the scene list
        subprocess.run([
            "scenedetect", "-i", video_path,
            "--output", images_dir,
            "list-scenes"
        ], check=True)
        
        # Now extract images based on scene duration
        extract_dynamic_images(video_path, images_dir)
    else:
        # Run scenedetect with fixed number of images
        subprocess.run([
            "scenedetect", "-i", video_path,
            "--output", images_dir,
            "list-scenes", "save-images", "-n", str(num_images_per_scene)
        ], check=True)


def extract_dynamic_images(video_path, images_dir):
    """Extract images with dynamic count based on scene duration."""
    import cv2
    
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = os.path.join(images_dir, f"{base_name}-Scenes.csv")
    
    # Read CSV to get scene info
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        cleaned_csv = lines[1:]
        reader = csv.DictReader(cleaned_csv)
        scenes = list(reader)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    for i, scene in enumerate(scenes, start=1):
        start_frame = int(scene["Start Frame"])
        end_frame = int(scene["End Frame"])
        scene_duration = float(scene["Length (seconds)"])
        
        # Calculate number of images based on duration
        # Rule: 1 image per 2-3 seconds, minimum 3, maximum 15
        num_images = max(3, min(15, int(scene_duration / 2.5) + 1))
        
        # Calculate frame positions
        if num_images == 1:
            frame_positions = [start_frame]
        else:
            step = (end_frame - start_frame) / (num_images - 1)
            frame_positions = [int(start_frame + j * step) for j in range(num_images)]
        
        # Extract frames
        for idx, frame_num in enumerate(frame_positions, start=1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if ret:
                img_name = f"{base_name}-Scene-{i:03d}-{idx:02d}.jpg"
                img_path = os.path.join(images_dir, img_name)
                cv2.imwrite(img_path, frame)
                print(f"📸 Extracted: {img_name} (Scene: {scene_duration:.1f}s, Images: {num_images})")
    
    cap.release()


def sanitize_timecode(tc):
    """Convert 00:00:11.000 → 00-00-11.000 for filenames."""
    return tc.replace(":", "-")


def get_image_timestamps(start_frame, end_frame, fps=25, num_images=3):
    """Calculate timestamps for thumbnail images distributed across the scene."""
    timestamps = []
    
    if num_images == 1:
        frame_positions = [start_frame]
    else:
        step = (end_frame - start_frame) / (num_images - 1)
        frame_positions = [int(start_frame + i * step) for i in range(num_images)]
    
    for frame in frame_positions:
        seconds = frame / fps
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        millisecs = int((secs - int(secs)) * 1000)
        timestamps.append(f"{hours:02d}-{minutes:02d}-{int(secs):02d}.{millisecs:03d}")
    
    return timestamps


def rename_files_from_csv(output_dir, base_name):
    images_dir = os.path.join(output_dir, "scenes_images")
    video_dir = os.path.join(output_dir, "scenes_videos")
    
    csv_path = os.path.join(images_dir, f"{base_name}-Scenes.csv")
    
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        cleaned_csv = lines[1:]
        reader = csv.DictReader(cleaned_csv)
        scenes = list(reader)

    for i, scene in enumerate(scenes, start=1):
        start_tc = sanitize_timecode(scene["Start Timecode"])
        end_tc = sanitize_timecode(scene["End Timecode"])
        start_frame = int(scene["Start Frame"])
        end_frame = int(scene["End Frame"])

        # RENAME VIDEO
        old_video = os.path.join(video_dir, f"{base_name}-Scene-{i:03d}.mp4")
        new_video = os.path.join(
            video_dir,
            f"{base_name}-Scene-{i:03d}_{start_tc}_to_{end_tc}.mp4"
        )

        if os.path.exists(old_video):
            os.rename(old_video, new_video)
            print(f"🎥 Renamed: {os.path.basename(new_video)}")

        # RENAME IMAGES
        img_index = 1
        existing_images = []
        while True:
            old_img = os.path.join(
                images_dir,
                f"{base_name}-Scene-{i:03d}-{img_index:02d}.jpg"
            )
            if not os.path.exists(old_img):
                break
            existing_images.append(old_img)
            img_index += 1

        num_images = len(existing_images)
        if num_images > 0:
            # Get FPS from CSV or default to 25
            fps = 25  # You can extract this from video metadata if needed
            image_timestamps = get_image_timestamps(start_frame, end_frame, fps=fps, num_images=num_images)
            
            for img_path, timestamp in zip(existing_images, image_timestamps):
                new_img = os.path.join(
                    images_dir,
                    f"{base_name}-Scene-{i:03d}_{timestamp}.jpg"
                )
                os.rename(img_path, new_img)
                print(f"🖼 Renamed: {os.path.basename(new_img)}")


# ---------------------------
# RUN EVERYTHING
# ---------------------------
start_time = time.time()

# Use dynamic image extraction (None = auto-calculate based on duration)
run_scenedetect(video_path, output_dir, num_images_per_scene=None)

video_name = os.path.splitext(os.path.basename(video_path))[0]
rename_files_from_csv(output_dir, video_name)

end_time = time.time()
print(f"\n⏱️ Total time: {end_time - start_time:.2f} seconds")
print("✅ All videos & images renamed with timestamps!")