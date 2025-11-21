import re
import subprocess
import csv
import os
import time
import cv2
import numpy as np
from datetime import timedelta
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import shutil

# -----------------------------------------
# CONFIGURATION
# -----------------------------------------
video_path = r"D:\SDNA\Scene_Detector\final_op\apple\apple.mp4"
output_dir = r"D:\SDNA\Scene_Detector\final_op\apple"
file_name = r"D:\SDNA\Scene_Detector\final_op\apple\processing_time.txt"

# -----------------------------------------
# 1. SCENEDETECT: Extract images
# -----------------------------------------
def run_scenedetect(video_path, output_dir):

    images_dir = os.path.join(output_dir, "scenes_images")
    video_dir = os.path.join(output_dir, "scenes_videos")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    # print("Detecting scenes and splitting video...")
    # subprocess.run([
    #     "scenedetect", "-i", video_path,
    #     "--output", video_dir,
    #     "detect-hash",
    #     "split-video"
    # ], check=True)

    # print("Generating scene list CSV...")
    # subprocess.run([
    #     "scenedetect", "-i", video_path,
    #     "--output", images_dir,
    #     "detect-hash",
    #     "list-scenes"
    # ], check=True)

    # scene_videos = os.listdir(video_dir)
    # for video in scene_videos:
    #     video_path = os.path.join(video_dir, video)
    #     print("Extracting images from:", video_path)
    #     extract_scene_images(video_path, images_dir)

    return images_dir


# def run_scenedetect(video_path, output_dir):

#     images_dir = os.path.join(output_dir, "scenes_images")
#     video_dir = os.path.join(output_dir, "scenes_videos")

#     os.makedirs(images_dir, exist_ok=True)
#     os.makedirs(video_dir, exist_ok=True)

#     print("Detecting scenes and splitting video...")
#     subprocess.run([
#         "scenedetect", "-i", video_path,
#         "--output", video_dir,
#         "detect-content", "--threshold", "27",
#         "detect-threshold", "--threshold", "12",
#         "split-video"
#     ], check=True)

#     print("Generating scene list CSV...")
#     subprocess.run([
#         "scenedetect", "-i", video_path,
#         "--output", images_dir,
#         "detect-content", "--threshold", "27",
#         "detect-threshold", "--threshold", "12",
#         "list-scenes"
#     ], check=True)

#     scene_videos = os.listdir(video_dir)
#     for video in scene_videos:
#         video_path = os.path.join(video_dir, video)
#         print("Extracting images from:", video_path)
#         extract_scene_images(video_path, images_dir)

#     return images_dir

def extract_scene_images(video_path, images_dir):

    command = [
        "scenedetect",
        "-i", video_path,
        "detect-content", "--threshold", "28",
        "detect-adaptive", "--threshold", "3.2", "--min-content-val", "15",
        "--frame-window", "3",
        "detect-threshold", "--threshold", "12",
        "save-images", "--num-images", "8", "--quality", "95",
        "-o", images_dir,
    ]
    subprocess.run(command)


# ------------------------------------------------------------------------------------
# READ SCENEDETECT CSV AND RETURN SCENE LIST
# ------------------------------------------------------------------------------------
def read_scene_csv(csv_path):
    scenes = []

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Skip header → Start from 2nd row
    for row in rows[2:]:
        if not row or not row[0].isdigit():
            continue

        scene_num = int(row[0])
        start_time = float(row[3]) 
        end_time = float(row[6])   
        length_sec = float(row[9])  

        scenes.append({
            "scene": scene_num,
            "start": start_time,
            "end": end_time,
            "length": length_sec
        })

    return scenes

# ------------------------------------------------------------------------------------
# RENAME SCENE IMAGES BY TIMESTAMP
# ------------------------------------------------------------------------------------

def seconds_to_hhmmss(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}-{minutes:02d}-{secs:02d}"

def rename_scene_images(scene_info, images_dir):
    """
    For each scene:
    - Get images for that scene
    - Rename using HH-MM-SS timestamp
    - Delete old images after copy
    """

    for scene in scene_info:
        scene_id = scene["scene"]
        start_time = scene["start"]
        end_time = scene["end"]
        length_sec = scene["length"]

        # Pattern like sub-Scene-001
        pattern = f"Scene-{scene_id:03d}"

        # Collect images
        scene_imgs = sorted([
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if pattern in f and f.lower().endswith(".jpg")
        ])

        if not scene_imgs:
            print(f"⚠ No images found for Scene {scene_id}")
            continue

        print(f"👉 Scene {scene_id}: {len(scene_imgs)} images found.")

        # Output folder
        scene_output = os.path.join(images_dir, f"scene_{scene_id:03d}")
        os.makedirs(scene_output, exist_ok=True)

        # Calculate gap
        if len(scene_imgs) > 1:
            time_step = length_sec / (len(scene_imgs) - 1)
        else:
            time_step = 0

        # Rename each image
        for i, img in enumerate(scene_imgs):
            timestamp = start_time + (i * time_step)

            # Convert to HH-MM-SS
            timestamp_hms = seconds_to_hhmmss(timestamp)

            new_name = f"Scene_{scene_id:03d}_{timestamp_hms}.jpg"
            new_path = os.path.join(scene_output, new_name)

            shutil.copy(img, new_path)

        print(f"✔ Scene {scene_id} renaming done → {scene_output}")

        # -----------------------------------------
        # DELETE OLD IMAGES AFTER COPYING
        # -----------------------------------------
        for img in scene_imgs:
            try:
                os.remove(img)
            except Exception as e:
                print(f"Error deleting {img}: {e}")

        print(f"🗑 Old images removed for Scene {scene_id}")
        print("----------------------------------------------")

# -----------------------------------------
# 2. CLIP MODEL SETUP
# -----------------------------------------
print("Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_features(img_path):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    return features[0] / features[0].norm()


def cosine(a, b):
    return torch.dot(a, b).item()


# -----------------------------------------
# 3. CLIP SUBSCENE DETECTION
# -----------------------------------------
def detect_subscenes(images_dir):

    image_paths = sorted(
        [os.path.join(images_dir, f) for f in os.listdir(images_dir)
         if f.lower().endswith((".jpg", ".png"))]
    )

    print(f"Total extracted images: {len(image_paths)}")

    embeddings = [get_features(p) for p in image_paths]

    threshold = 0.90  

    subscene_id = 0
    subscenes = {subscene_id: [image_paths[0]]}

    for i in range(1, len(embeddings)):
        sim = cosine(embeddings[i-1], embeddings[i])
        print(f"Similarity {i}: {sim}")

        if sim < threshold: 
            subscene_id += 1
            subscenes[subscene_id] = []

        subscenes[subscene_id].append(image_paths[i])

    return subscenes


def move_files_and_remove_subfolders(parent_folder):
    """
    Moves all files from subfolders into the parent folder.
    - No renaming
    - No overwriting (duplicates are skipped)
    - Removes empty subfolders after moving
    """
    for root, dirs, files in os.walk(parent_folder):
        if root == parent_folder:
            continue

        for file in files:
            src = os.path.join(root, file)
            dst = os.path.join(parent_folder, file)

            if os.path.exists(dst):
                print(f"Skipped (already exists): {file}")
                continue

            shutil.move(src, dst)
            print(f"Moved: {src} → {dst}")

    for root, dirs, files in os.walk(parent_folder, topdown=False):
        if root == parent_folder:
            continue
        if not os.listdir(root):  # Folder empty
            os.rmdir(root)
            print(f"Removed empty folder: {root}")

    print("Completed moving files and removing subfolders.")


# -----------------------------------------
# 4. SAVE FINAL SUBSCENES
# -----------------------------------------
# def save_subscenes(subscenes):
#     output = r"D:\SDNA\Scene_Detector\scene_output_clip"

#     if os.path.exists(output):
#         shutil.rmtree(output)

#     os.makedirs(output)

#     for sid, imgs in subscenes.items():
#         folder = os.path.join(output, f"subscene_{sid}")
#         os.makedirs(folder, exist_ok=True)

#         scene_id_str = f"{sid:03d}"   # convert to 000, 001, 002...

#         for img in imgs:
#             base = os.path.basename(img)

#             # original example: Scene_001_00-00-00.jpg
#             parts = base.split("_")

#             if len(parts) < 3:
#                 new_name = base   # keep original if unexpected format
#             else:
#                 # keep timestamp part: parts[2] = "00-00-00.jpg"
#                 timestamp = parts[2]
#                 new_name = f"Scene_{scene_id_str}_{timestamp}"

#             shutil.copy(img, os.path.join(folder, new_name))

#     print("\n✔ Saved and Renamed Sub-Scenes using CLIP →", output)


def save_subscenes(subscenes):
    output = r"D:\SDNA\Scene_Detector\scene_output_clip"

    if os.path.exists(output):
        shutil.rmtree(output)

    os.makedirs(output)

    for sid, imgs in subscenes.items():
        folder = os.path.join(output, f"subscene_{sid}")
        os.makedirs(folder, exist_ok=True)

        scene_id_str = f"{sid:03d}"

        for img in imgs:
            base = os.path.basename(img)

            # Extract timestamp safely
            before, _, timestamp = base.rpartition("_")

            if timestamp == "":
                new_name = base  # fallback
            else:
                new_name = f"Scene_{scene_id_str}_{timestamp}"

            shutil.copy(img, os.path.join(folder, new_name))

    print("\n✔ Saved and Renamed Sub-Scenes using CLIP →", output)

# -----------------------------------------
# MAIN PIPELINE
# -----------------------------------------
if __name__ == "__main__":

    start_time = time.time()

    print("="*80)
    print("      ADVANCED SCENE + SUBSCENE DETECTION SYSTEM (SceneDetect + CLIP)")
    print("="*80)

    images_dir = run_scenedetect(video_path, output_dir)

    # csv_path = os.path.join(output_dir, "scenes_images", "Democrats_defend-Scenes.csv")
    
    # if not os.path.exists(csv_path):
    #     csv_path = os.path.join(output_dir, "Democrats_defend-Scenes.csv")

    # scene_data = read_scene_csv(csv_path)

    # rename_scene_images(scene_data, images_dir)

    # move_files_and_remove_subfolders(images_dir)

    subscenes = detect_subscenes(images_dir)

    save_subscenes(subscenes)

    end_time = time.time()

    # Save timing
    with open(file_name, 'w') as f:
        f.write(f"Total time: {end_time - start_time:.2f} seconds")

    print(f"\nTotal time: {end_time - start_time:.2f} seconds")
    print("Processing complete!")
    print("="*80)
