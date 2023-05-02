import os
import shutil

# Path to the directory where your folders are stored
base_dir = "/srv/beegfs02/scratch/rl_course/data/proj-diffuse-drive/dataset"

# Path to the directory where you want to store the merged images
merged_dir = "/srv/beegfs02/scratch/rl_course/data/proj-diffuse-drive/dataset/extracted_rgb_front_unconditioned"

# Loop through each folder in the base directory

for folder_name_weather in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name_weather)
    folder_path = os.path.join(folder_path, "data")
    # Only process folders, not files
    if not os.path.isdir(folder_path):
        continue
    
    for folder_name_route in os.listdir(folder_path):
        folder_path_current = os.path.join(folder_path, folder_name_route)
        folder_path_rgb_front = os.path.join(folder_path_current, "rgb_front")
        
        if not os.path.isdir(folder_path_rgb_front):
            continue
    
        # Loop through each file in the folder
        for file_name in os.listdir(folder_path_rgb_front):
            file_path = os.path.join(folder_path_rgb_front, file_name)
            
            # Only process image files
            if not file_name.endswith(".jpg"):
                continue
            
            # Copy the image to the merged directory with a unique name
            new_name = f"{folder_name_weather}_{folder_name_route}_{file_name}"
            new_path = os.path.join(merged_dir, new_name)
            shutil.copyfile(file_path, new_path)