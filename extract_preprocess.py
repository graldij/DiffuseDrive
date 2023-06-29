from fileinput import filename
import os
import shutil
from PIL import Image
from torchvision import transforms
import numpy as np

original_dir = "/scratch_net/biwidl216/rl_course_14/extracted_dataset"
new_dir = "/scratch_net/biwidl216/rl_course_14/preprocessed_extracted_diffusedrive_dataset"


def create_folder(new_folder_path):
    if not os.path.exists(new_folder_path):
        # If it doesn't exist, create it
        os.makedirs(new_folder_path)
    
def apply_transforms(image):      
    img_size = 128 
    transform = transforms.Compose([
                transforms.Resize((img_size,img_size))
                # transforms.ToTensor(),
                # transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
            ])
    return transform(image)

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
        ## MOD Minxuan: add virdview for validation
        folder_path_bev = os.path.join(folder_path_current, "birdview")

        if not os.path.isdir(folder_path_rgb_front) or not os.path.isdir(folder_path_measurements) or not os.path.isdir(folder_path_bev):
            continue

        folder_path_current_new = os.path.join(new_folder_path, folder_name_route)
        folder_path_rgb_front_new = os.path.join(folder_path_current_new, "rgb_front")
        folder_path_measurements_new = os.path.join(folder_path_current_new, "measurements")
        ## MOD Minxuan: add birdview
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
            ## MOD Minxuan: add birdview
            file_path_bev = os.path.join(folder_path_bev, file_name)

            # Only process image files and ensure the corresponding measurements file exist
            if not file_name.endswith(".jpg") or not os.path.exists(file_path_measurements):
                continue

            file_path_rgb_front_new = os.path.join(folder_path_rgb_front_new, file_name_no_suffix + ".jpg")
            file_path_measurements_new = os.path.join(folder_path_measurements_new, file_name_no_suffix + ".json")
            file_path_bev_new = os.path.join(folder_path_bev_new, file_name_no_suffix + ".jpg")
            # extract jpg format to png
            im = Image.open(file_path_rgb_front)
            im = im.convert('RGB')
            im = apply_transforms(im)
            # print(im.numpy().shape)
            # im = Image.fromarray(im.numpy())
            im.save(file_path_rgb_front_new, quality=50)
            
            ## MOD Minxuan: convert bev images
            bev_im = Image.open(file_path_bev)
            bev_im = bev_im.convert('RGB')
            bev_im.save(file_path_bev_new, quality=75)
            
            loaded_im = Image.open(file_path_rgb_front_new)
            # loaded_im = loaded_im.convert('RGB')
            # breakpoint()
            if np.array(im).mean() - np.array(loaded_im).mean() == 0:
                print("same")
            # shutil.copy(filepath_rgb_front, file_path_rgb_front_new)
            shutil.copy(file_path_measurements, file_path_measurements_new)
            #shutil.copy(file_path_bev, file_path_bev_new)
            i += 1
            print(i)