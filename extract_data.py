import os
import shutil

original_dir = "/srv/beegfs02/scratch/rl_course/data/proj-diffuse-drive/dataset"
new_dir = "/scratch_net/biwidl216/rl_course_14/extracted_dataset"


def create_folder(new_folder_path):
    if not os.path.exists(new_folder_path):
        # If it doesn't exist, create it
        os.makedirs(new_folder_path)

create_folder(new_dir)
i = 0
for folder_name_weather in os.listdir(original_dir):
    folder_path = os.path.join(original_dir, folder_name_weather)
    folder_path = os.path.join(folder_path, "data")

    
    new_folder_path = os.path.join(new_dir, folder_name_weather)
    create_folder(new_folder_path)
    new_folder_path = os.path.join(new_folder_path, "data")
    create_folder(new_folder_path)
    # Only process folders, not files
    if not os.path.isdir(folder_path):
        continue
    
    for folder_name_route in os.listdir(folder_path):
        folder_path_current = os.path.join(folder_path, folder_name_route)
        folder_path_rgb_front = os.path.join(folder_path_current, "rgb_front")
        folder_path_measurements = os.path.join(folder_path_current, "measurements")
        ## MOD Minxuan: Add birdview folder
        folder_path_bev = os.path.join(folder_path_current, "birdview")

        if not os.path.isdir(folder_path_rgb_front) or not os.path.isdir(folder_path_measurements) or not os.path.isdir(folder_path_bev):
            continue
    
        folder_path_current_new = os.path.join(new_folder_path, folder_name_route)
        folder_path_rgb_front_new = os.path.join(folder_path_current_new, "rgb_front")
        folder_path_measurements_new = os.path.join(folder_path_current_new, "measurements")
        folder_path_bev_new = os.path.join(folder_path_current_new, "birdview")
        create_folder(folder_path_current_new)
        create_folder(folder_path_rgb_front_new)
        create_folder(folder_path_measurements_new)
        create_folder(folder_path_bev_new)

        # Loop through each file in the folder
        for file_name in os.listdir(folder_path_rgb_front):
            file_path_rgb_front = os.path.join(folder_path_rgb_front, file_name)
            
            file_name_no_suffix, _ = os.path.splitext(file_name)
            file_id = int(file_name_no_suffix)
            file_path_measurements = os.path.join(folder_path_measurements, file_name_no_suffix + ".json")
            ## MOD Minxuan: Add birdview
            file_path_bev = os.path.join(folder_path_bev, file_name)
            
            # Only process image files and ensure the corresponding measurements file exist
            if not file_name.endswith(".jpg") or not os.path.exists(file_path_measurements):
                continue

            file_path_rgb_front_new = os.path.join(folder_path_rgb_front_new, file_name)
            file_path_measurements_new = os.path.join(folder_path_measurements_new, file_name_no_suffix + ".json")
            file_path_bev_new = os.path.join(folder_path_bev_new, file_name)
            shutil.copy(file_path_rgb_front, file_path_rgb_front_new)
            shutil.copy(file_path_measurements, file_path_measurements_new)
            shutil.copy(file_path_bev, file_path_bev_new)
            i += 1
            print(i)