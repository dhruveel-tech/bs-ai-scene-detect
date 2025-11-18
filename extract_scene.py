import subprocess
import csv
import os
import time
from datetime import timedelta

video_path = r"D:\SDNA\Scene_Detector\Main_op\bicycle_kick\Cristiano_Ronaldo_bicycle_kick.mp4"
output_dir = r"D:\SDNA\Scene_Detector\Main_op\bicycle_kick"

def run_scenedetect(video_path, output_dir):
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

    # Run scenedetect to generate scene images + CSV
    subprocess.run([
        "scenedetect", "-i", video_path,
        "--output", images_dir,
        "list-scenes", "save-images"
    ], check=True)


def sanitize_timecode(tc):
    """Convert 00:00:11.000 → 00-00-11.000 for filenames."""
    return tc.replace(":", "-")


def get_image_timestamps(start_frame, end_frame, fps=25, num_images=3):
    """Calculate timestamps for thumbnail images distributed across the scene."""
    timestamps = []
    
    if num_images == 1:
        # Single image at the start
        frame_positions = [start_frame]
    else:
        # Distribute images evenly across the scene
        step = (end_frame - start_frame) / (num_images - 1)
        frame_positions = [int(start_frame + i * step) for i in range(num_images)]
    
    for frame in frame_positions:
        seconds = frame / fps
        # Format as HH-MM-SS.mmm (with milliseconds for uniqueness)
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
        # Skip first line ("Timecode List:")
        cleaned_csv = lines[1:]
        reader = csv.DictReader(cleaned_csv)
        scenes = list(reader)

    for i, scene in enumerate(scenes, start=1):
        start_tc = sanitize_timecode(scene["Start Timecode"])
        end_tc = sanitize_timecode(scene["End Timecode"])
        start_frame = int(scene["Start Frame"])
        end_frame = int(scene["End Frame"])

        # -------------------------------
        #  RENAME VIDEO
        # -------------------------------
        old_video = os.path.join(video_dir, f"{base_name}-Scene-{i:03d}.mp4")
        new_video = os.path.join(
            video_dir,
            f"{base_name}-Scene-{i:03d}_{start_tc}_to_{end_tc}.mp4"
        )

        if os.path.exists(old_video):
            os.rename(old_video, new_video)
            print(f"🎥 Renamed: {os.path.basename(new_video)}")
        else:
            print(f"⚠ Video missing: {old_video}")

        # -------------------------------
        #  RENAME IMAGES with individual timestamps
        # -------------------------------
        # First, count how many images exist for this scene
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

        # Calculate timestamps for each image
        num_images = len(existing_images)
        if num_images > 0:
            image_timestamps = get_image_timestamps(start_frame, end_frame, fps=25, num_images=num_images)
            
            # Rename each image with its specific timestamp
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
run_scenedetect(video_path, output_dir)

video_name = os.path.splitext(os.path.basename(video_path))[0]
rename_files_from_csv(output_dir, video_name)
end_time = time.time()
print(end_time - start_time)
print("\n✅ All videos & images renamed with timestamps!")