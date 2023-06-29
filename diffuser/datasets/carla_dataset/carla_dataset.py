import datasets
import os
from PIL import Image
import json
import numpy
from datasets import load_dataset
import torch
from torchvision import transforms
import time

class CarlaDatasetConfig(datasets.BuilderConfig):
    def __init__(
            self, 
            name, 
            description, 
            base_dir, 
            img_buffer_size = 3, 
            waypoint_buffer_size = 3, 
            waypoint_prediction_size = 6, 
            img_future_size = 6,
            high_level_cmd_size = 6, 
            horizon = 10,
            is_valid = False,
            using_cmd = False,
            **kwargs
        ):
        """BuilderConfig for CarlaDataset.
        Args:
          name: name of config used
          description: 
          base_dir: directory to the dataset
          img_buffer_size: how many past images to load
          waypoint_buffer_size: how many past waypoints to load
          waypoint_prediction_size: how many future waypoints to load
          high_level_command_size: how many high level command from carla to load
          **kwargs: keyword arguments forwarded to super.
        """
        super(CarlaDatasetConfig, self).__init__(version=datasets.Version("1.0.0"), **kwargs)
        self.base_dir = base_dir
        self.img_buffer_size = img_buffer_size
        self.img_future_size = img_future_size
        self.waypoint_buffer_size = waypoint_buffer_size
        self.waypoint_prediction_size = waypoint_prediction_size
        self.high_level_cmd_size = high_level_cmd_size
        self.name = name
        self.horizon = horizon
        self.is_valid = is_valid
        self.using_cmd = using_cmd

## For Minxuan: path for validation dataset: extracted_diffusedrive_dataset_valid_png
class CarlaDataset(datasets.GeneratorBasedBuilder):
    "Carla Dataset"
    BUILDER_CONFIGS = [
        CarlaDatasetConfig(
            name="unconditioned",
            description="Image only datas for unconditioned diffusion",
            base_dir="/scratch_net/biwidl216/rl_course_14/preprocessed_extracted_diffusedrive_dataset",
            img_buffer_size = 0
        ),
        CarlaDatasetConfig(
            name="waypoint_imageConditioned",
            description="Data for imaged conditioned waypoint diffusion",
            base_dir="/scratch_net/biwidl216/rl_course_14/preprocessed_extracted_diffusedrive_dataset",
            img_buffer_size = 4,
            waypoint_buffer_size = 4,
            waypoint_prediction_size = 6,
            high_level_cmd_size = 1
        ),
        # [Note] This config is used to load the data
        CarlaDatasetConfig(
            name="decdiff",
            description="format for decdiff trainer",
            base_dir="/scratch_net/biwidl216/rl_course_14/preprocessed_extracted_diffusedrive_dataset",
            horizon = 12,
            img_buffer_size = 3,
            img_future_size = 0,
            waypoint_buffer_size = 3,
            waypoint_prediction_size = 8,
            high_level_cmd_size = 1, 
            is_valid = False,
            using_cmd = False,
        ),
        CarlaDatasetConfig(
            name="waypoint_unconditioned",
            description="format for decdiff trainer",
            base_dir="/scratch_net/biwidl216/rl_course_14/preprocessed_extracted_diffusedrive_dataset",
            horizon = 12,
            waypoint_buffer_size = 3,
            waypoint_prediction_size = 8,
            high_level_cmd_size = 1,
            is_valid = False,
            using_cmd = False,
        )
    ]

    def _info(self):
        if self.config.name == "unconditioned":
            return datasets.DatasetInfo(
                description="Image only data from carla",
                features=datasets.Features(
                    {
                        "current_image": datasets.Image()
                    }
                )
            )
        elif self.config.name =="waypoint_imageConditioned":
            past_waypoint_list = [datasets.Array2D(shape=(1,3), dtype=float)] * self.config.waypoint_buffer_size
            future_waypoint_list = [datasets.Array2D(shape=(1,3), dtype=float)] * self.config.waypoint_prediction_size
            high_level_cmd_list = [datasets.Array2D(shape=(1,3), dtype=float)] * self.config.high_level_cmd_size
            past_image_list = [datasets.Image()] * self.config.img_buffer_size

            features = { "past_waypoint_"+str(k):v for k,v in enumerate(past_waypoint_list)}
            features.update({"current_waypoint": datasets.Array2D(shape=(1,3), dtype="float")})
            features.update({"future_waypoint_"+str(k):v for k,v in enumerate(future_waypoint_list)})
            features.update({"high_level_cmd_"+str(k):v for k,v in enumerate(high_level_cmd_list)})
            features.update({"past_image_"+str(k):v for k,v in enumerate(past_image_list)})
            features.update({"current_image":datasets.Image()})

            return datasets.DatasetInfo(
                description="Dataset collected from carla",
                features=datasets.Features(features),
                supervised_keys=("waypoints","image")
            )
        elif self.config.name =="waypoint_unconditioned":
            features ={
                "actions": datasets.Sequence(datasets.Array2D(shape=(1,3), dtype=float), self.config.horizon)
            }
            ## MOD Minxuan: load bridview when using validation dataset
            ## MOD Minxuan: add command
            if self.config.using_cmd:
                #past_cmd_list = [datasets.Array2D(shape=(1,1), dtype=int)] * self.config.waypoint_buffer_size
                #features.update({"past_cmd_"+str(k):v for k,v in enumerate(past_cmd_list)})
                #features.update({"current_cmd": datasets.Array2D(shape=(1,1), dtype="int")})
                features.update({"cmd": datasets.Sequence(datasets.Array2D(shape=(1,1), dtype=int), self.config.waypoint_buffer_size)})

            if self.config.is_valid:
                features["birdview"] = datasets.Image()
                return datasets.DatasetInfo(
                description="Validation Dataset collected from carla",
                features=datasets.Features(features)
                )
            return datasets.DatasetInfo(
                description="Data with only waypoints",
                features=datasets.Features(features)
            )
        else:
            features ={
                "actions": datasets.Sequence(datasets.Array2D(shape=(1,3), dtype=float), self.config.horizon),
                "observations": datasets.Sequence(datasets.Image(), self.config.horizon)
            }
            ## MOD Minxuan: load bridview when using validation dataset
            ## MOD Minxuan: add cmd
            if self.config.using_cmd:
                #past_cmd_list = [datasets.Array2D(shape=(1,1), dtype=int)] * self.config.waypoint_buffer_size
                features.update({"cmd": datasets.Sequence(datasets.Array2D(shape=(1,1), dtype=int), self.config.waypoint_buffer_size)})

            if self.config.is_valid:
                features["birdview"] = datasets.Image()
                return datasets.DatasetInfo(
                description="Validation Dataset collected from carla",
                features=datasets.Features(features)
                )
            return datasets.DatasetInfo(
                description="Dataset collected from carla",
                features=datasets.Features(features)
                )


    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"base_dir": self.config.base_dir},
            )
        ]
       

    def _generate_examples(self, base_dir):
        if self.config.name == "unconditioned": 
            raise NotImplementedError
        
            base_dir = self.config.base_dir
            for folder_name_weather in os.listdir(base_dir):
                folder_path = os.path.join(base_dir, folder_name_weather)
                folder_path = os.path.join(folder_path, "data")
                # Only process folders, not files
                if not os.path.isdir(folder_path):
                    continue
            
                for folder_name_route in os.listdir(folder_path):
                    folder_path_current = os.path.join(folder_path, folder_name_route)
                    folder_path_rgb_front = os.path.join(folder_path_current, "rgb_front")
                    folder_path_measurements = os.path.join(folder_path_current, "measurements")
                    
                    if not os.path.isdir(folder_path_rgb_front) or not os.path.isdir(folder_path_measurements):
                        continue
            
                    # Loop through each file in the folder
                    for file_name in os.listdir(folder_path_rgb_front):
                        file_path_rgb_front = os.path.join(folder_path_rgb_front, file_name)
                        
                        file_name_no_suffix, _ = os.path.splitext(file_name)
                        # Only process image files
                        if not file_name.endswith(".jpg"):
                            continue

                        # print(file_path_rgb_front)
                        current_img = Image.open(file_path_rgb_front)

                        # Copy the image to the merged directory with a unique name
                        new_name = f"{folder_name_weather}_{folder_name_route}_{file_name_no_suffix}"
                        yield new_name, {
                            "current_image":{"path":file_path_rgb_front, "bytes": current_img}
                        }
        elif self.config.name == "decdiff":
            for folder_name_weather in os.listdir(base_dir):
                folder_path = os.path.join(base_dir, folder_name_weather)
                folder_path = os.path.join(folder_path, "data")
                # Only process folders, not files
                if not os.path.isdir(folder_path):
                    continue
                
                for folder_name_route in os.listdir(folder_path):
                    folder_path_current = os.path.join(folder_path, folder_name_route)
                    folder_path_rgb_front = os.path.join(folder_path_current, "rgb_front")
                    folder_path_measurements = os.path.join(folder_path_current, "measurements")
                    
                    if not os.path.isdir(folder_path_rgb_front) or not os.path.isdir(folder_path_measurements):
                        continue

                    ## MOD Minxuan: for validation dataset, also consider birdview image
                    is_valid = self.config.is_valid
                    using_cmd = self.config.using_cmd
                    if is_valid:
                        folder_path_bev = os.path.join(folder_path_current, "birdview")
                        if not os.path.isdir(folder_path_bev):
                            continue
                        
                    # MOD Minxuan: sequence reading in order, is_valid parameter used
                    measure_dir_list = os.listdir(folder_path_rgb_front)
                    
                    if is_valid:
                        measure_dir_list.sort()
                        
                    # Loop through each file in the folder
                    for file_name in measure_dir_list:
                        file_path_rgb_front = os.path.join(folder_path_rgb_front, file_name)
                        
                        file_name_no_suffix, _ = os.path.splitext(file_name)
                        file_id = int(file_name_no_suffix)
                        file_path_measurements = os.path.join(folder_path_measurements, file_name_no_suffix + ".json")
                        
                        ## MOD Minxuan: for validation dataset, also consider birdview filepath
                        if is_valid:
                            file_path_bev = os.path.join(folder_path_bev, file_name_no_suffix + ".jpg")
    
                        # Only process image files and ensure the corresponding measurements file exist
                        if not file_name.endswith(".jpg") or not os.path.exists(file_path_measurements):
                            continue

                        ## MOD Minxuan: add birdview
                        if is_valid:
                            current_bev = {"path":file_path_bev, "bytes":Image.open(file_path_bev)}
                        current_img = Image.open(file_path_rgb_front)
                        action_list, waypoint_flag = self.get_action_list(file_path_measurements, file_id, folder_path_measurements)
                        # TODO: change this high_level_cmd last dim is command not orientation
                        # high_level_cmd_ego = self.convert_high_level_cmd_to_ego(high_level_cmd_gps, current_waypoint_gps)
                        past_img, past_img_flag = self.get_past_img(file_id, folder_path_rgb_front)
                        future_img, future_img_flag = self.get_future_img(file_id, folder_path_rgb_front)

                        ## MOD Minxuan: add past_cmd_list (past+current) and current_cmd
                        if using_cmd:
                            cmd_list, cmd_flag = self.get_cmd_list(file_path_measurements, file_id, folder_path_measurements)

                        if waypoint_flag * past_img_flag * future_img_flag < 1:
                            continue
                        
                        image_list = []
                        for i in reversed(past_img):
                            image_list.append(i)
                        image_list.append(i)
                        for i in future_img:
                            image_list.append(i)

                        # Copy the image to the merged directory with a unique name
                        output= {"actions": action_list, "observations": image_list}
                        ## MOD Minxuan: add bev to output
                        if is_valid:
                            output["birdview"] = current_bev
                        if using_cmd:
                            output["cmd"] = cmd_list
                        new_name = f"{folder_name_weather}_{folder_name_route}_{file_name_no_suffix}"
                        yield new_name, output
        elif self.config.name == "waypoint_unconditioned":
            for folder_name_weather in os.listdir(base_dir):
                folder_path = os.path.join(base_dir, folder_name_weather)
                folder_path = os.path.join(folder_path, "data")
                # Only process folders, not files
                if not os.path.isdir(folder_path):
                    continue
                
                for folder_name_route in os.listdir(folder_path):
                    folder_path_current = os.path.join(folder_path, folder_name_route)
                    folder_path_measurements = os.path.join(folder_path_current, "measurements")
                    
                    if not os.path.isdir(folder_path_measurements):
                        continue
                    
                    ## MOD Minxuan: for validation dataset, also consider birdview image
                    is_valid = self.config.is_valid
                    using_cmd = self.config.using_cmd
                    if is_valid:
                        folder_path_bev = os.path.join(folder_path_current, "birdview")
                        if not os.path.isdir(folder_path_bev):
                            continue
                    # Loop through each file in the folder
                    # MOD Minxuan: sequence reading in order, is_valid parameter used here
                    measure_dir_list = os.listdir(folder_path_measurements)
                    is_valid = self.config.is_valid
                    if is_valid:
                        measure_dir_list.sort()
                    for file_name in measure_dir_list:
                        file_path_measurements = os.path.join(folder_path_measurements, file_name)
                        
                        file_name_no_suffix, _ = os.path.splitext(file_name)
                        file_id = int(file_name_no_suffix)
                        # Only process image files and ensure the corresponding measurements file exist
                        if not file_name.endswith(".json"):
                            continue
                        
                        ## MOD Minxuan: for validation dataset, also consider birdview filepath
                        if is_valid:
                            file_path_bev = os.path.join(folder_path_bev, file_name_no_suffix + ".jpg")
                            current_bev = {"path":file_path_bev, "bytes":Image.open(file_path_bev)}

                        
                        action_list, waypoint_flag = self.get_action_list(file_path_measurements, file_id, folder_path_measurements)
                        high_level_cmd_gps = self.get_high_level_cmd_gps(file_id, folder_path_measurements)
                        if using_cmd:
                            cmd_list, cmd_flag = self.get_cmd_list(file_path_measurements, file_id, folder_path_measurements)
                        if not waypoint_flag:
                            continue


                        # Copy the image to the merged directory with a unique name
                        output= {"actions": action_list}
                        ## MOD Minxuan: add bev to output
                        if is_valid:
                            output["birdview"] = current_bev
                        if using_cmd:
                            output["cmd"] = cmd_list

                        new_name = f"{folder_name_weather}_{folder_name_route}_{file_name_no_suffix}"
                        yield new_name, output
        else:
            raise NotImplementedError
    
            for folder_name_weather in os.listdir(base_dir):
                folder_path = os.path.join(base_dir, folder_name_weather)
                folder_path = os.path.join(folder_path, "data")
                # Only process folders, not files
                if not os.path.isdir(folder_path):
                    continue
                
                for folder_name_route in os.listdir(folder_path):
                    folder_path_current = os.path.join(folder_path, folder_name_route)
                    folder_path_rgb_front = os.path.join(folder_path_current, "rgb_front")
                    folder_path_measurements = os.path.join(folder_path_current, "measurements")
                    
                    if not os.path.isdir(folder_path_rgb_front) or not os.path.isdir(folder_path_measurements):
                        continue
                
                    # Loop through each file in the folder
                    # MOD Minxuan: sequence reading in order, is_valid parameter used
                    measure_dir_list = os.listdir(folder_path_rgb_front)
                    is_valid = self.config.is_valid
                    if is_valid:
                        measure_dir_list.sort()
                    for file_name in measure_dir_list:
                        file_path_rgb_front = os.path.join(folder_path_rgb_front, file_name)
                        
                        file_name_no_suffix, _ = os.path.splitext(file_name)
                        file_id = int(file_name_no_suffix)
                        file_path_measurements = os.path.join(folder_path_measurements, file_name_no_suffix + ".json")
                        
                        # Only process image files and ensure the corresponding measurements file exist
                        if not file_name.endswith(".jpg") or not os.path.exists(file_path_measurements):
                            continue

                        
                        current_img = Image.open(file_path_rgb_front)
                        current_json = json.load(open(file_path_measurements))

            
                        if (current_json.get('ego_x') is None):
                            current_waypoint_gps = numpy.array([current_json["gps_x"], current_json["gps_y"], current_json["theta"]])
                            current_waypoint_ego = numpy.array([0.0, 0.0, 0.0])

                            past_waypoint_gps = self.get_past_waypoint_gps(file_id, folder_path_measurements)
                            past_waypoint_ego = convert_gps_to_ego(past_waypoint_gps, current_waypoint_gps)
                            future_waypoint_gps = self.get_future_waypoint_gps(file_id, folder_path_measurements)
                            future_waypoint_ego = convert_gps_to_ego(future_waypoint_gps, current_waypoint_gps)
                            high_level_cmd_gps = self.get_high_level_cmd_gps(file_id, folder_path_measurements)
                        # TODO: change this high_level_cmd last dim is command not orientation
                        high_level_cmd_ego = self.convert_high_level_cmd_to_ego(high_level_cmd_gps, current_waypoint_gps)
                        past_img = self.get_past_img(file_id, folder_path_rgb_front)

                        output = {"past_waypoint_"+str(k):v for k,v in enumerate(past_waypoint_ego)}
                        output.update({"current_waypoint": current_waypoint_ego})
                        output.update({"future_waypoint_"+str(k):v for k,v in enumerate(future_waypoint_ego)})
                        output.update({"high_level_cmd_"+str(k):v for k,v in enumerate(high_level_cmd_ego)})
                        output.update({"past_image_"+str(k):v for k,v in enumerate(past_img)})
                        output.update({"current_image": {"path":file_path_rgb_front, "bytes": current_img}})

                        # Copy the image to the merged directory with a unique name
                        new_name = f"{folder_name_weather}_{folder_name_route}_{file_name_no_suffix}"
                        yield new_name, output

    def get_action_list(self, file_path_measurements, file_id, folder_path_measurements):
        current_json = json.load(open(file_path_measurements))
        # NOT in use, since at every waypoint the ego pose will change, so have to save the whole horizon for each data point.
        if (current_json.get('ego_x') is None):
            current_waypoint_gps = numpy.array([current_json["gps_x"], current_json["gps_y"], current_json["theta"]])
            current_waypoint_ego = numpy.array([0.0, 0.0, 0.0])

            past_waypoint_gps, past_waypoint_flag = self.get_past_waypoint_gps(file_id, folder_path_measurements)
            past_waypoint_ego = convert_gps_to_ego(past_waypoint_gps, current_waypoint_gps)
            future_waypoint_gps, future_waypoint_flag = self.get_future_waypoint_gps(file_id, folder_path_measurements)
            future_waypoint_ego = convert_gps_to_ego(future_waypoint_gps, current_waypoint_gps)

            if future_waypoint_flag * past_waypoint_flag < 1:
                return None, False

        else:
            current_waypoint_ego = numpy.array([0.0, 0.0, 0.0])
            past_waypoint_ego, past_waypoint_flag = self.get_past_waypoint_ego(file_id, folder_path_measurements)
            future_waypoint_ego, future_waypoint_flag = self.get_future_waypoint_ego(file_id, folder_path_measurements)

            if past_waypoint_flag * future_waypoint_flag < 1:
                return None, False

        action_list = []
        for i in reversed(past_waypoint_ego):
            action_list.append(i[0])
        action_list.append(current_waypoint_ego)
        for i in future_waypoint_ego:
            action_list.append(i[0])
        return action_list, True

    def get_past_waypoint_gps(self, file_id, folder_path):
        past_waypoint_gps = [None] * self.config.waypoint_buffer_size
        for t in range(self.config.waypoint_buffer_size):
            previous_file_id = file_id - t - 1
            previous_file_path_measurements = os.path.join(folder_path, str(previous_file_id).zfill(4) + ".json")
                        
            if not os.path.exists(previous_file_path_measurements):
                return past_waypoint_gps, False

            previous_json = json.load(open(previous_file_path_measurements))
            past_waypoint_gps[t] = numpy.array([previous_json["gps_x"], previous_json["gps_y"], previous_json["theta"]])
        
        return past_waypoint_gps, True

    def get_past_waypoint_ego(self, file_id, folder_path):
        past_waypoint_ego = [None] * self.config.waypoint_buffer_size
        for t in range(self.config.waypoint_buffer_size):
            previous_file_id = file_id - t - 1
            previous_file_path_measurements = os.path.join(folder_path, str(previous_file_id).zfill(4) + ".json")
                        
            if not os.path.exists(previous_file_path_measurements):
                return past_waypoint_ego, False

            previous_json = json.load(open(previous_file_path_measurements))
            if (previous_json.get('ego_x') is None):
                return past_waypoint_ego, False

            past_waypoint_ego[t] = numpy.array([previous_json["ego_x"], previous_json["ego_y"], previous_json["ego_theta"]])
        
        return past_waypoint_ego, True

    def get_future_waypoint_gps(self, file_id, folder_path):
        future_waypoint_gps = [None] * self.config.waypoint_prediction_size
        for t in range(self.config.waypoint_prediction_size):
            next_file_id = file_id + t + 1
            next_file_path_measurements = os.path.join(folder_path, str(next_file_id).zfill(4) + ".json")
                        
            if not os.path.exists(next_file_path_measurements):
                return future_waypoint_gps, False

            next_json = json.load(open(next_file_path_measurements))
            future_waypoint_gps[t] = numpy.array([next_json["gps_x"], next_json["gps_y"], next_json["theta"]])
        
        return future_waypoint_gps, True

    def get_future_waypoint_ego(self, file_id, folder_path):
        future_waypoint_ego = [None] * self.config.waypoint_prediction_size
        for t in range(self.config.waypoint_prediction_size):
            next_file_id = file_id + t + 1
            next_file_path_measurements = os.path.join(folder_path, str(next_file_id).zfill(4) + ".json")
                        
            if not os.path.exists(next_file_path_measurements):
                return future_waypoint_ego, False

            next_json = json.load(open(next_file_path_measurements))
            if (next_json.get('ego_x') is None):
                return future_waypoint_ego, False
            future_waypoint_ego[t] = numpy.array([next_json["ego_x"], next_json["ego_y"], next_json["ego_theta"]])
        
        return future_waypoint_ego, True

    def get_high_level_cmd_gps(self, file_id, folder_path):
        high_level_cmd_gps = [None] * self.config.high_level_cmd_size
        current_json_file = os.path.join(folder_path, str(file_id).zfill(4) + ".json")
        current_json = json.load(open(current_json_file))
        high_level_cmd_list = current_json["future_waypoints"]

        for t in range(self.config.high_level_cmd_size):
            if t == len(high_level_cmd_list):
                break
            
            high_level_cmd_gps[t] = numpy.array(high_level_cmd_list[t])

        return high_level_cmd_gps

    def get_past_img(self, file_id, folder_path):
        past_img = [None] * self.config.img_buffer_size
        for t in range(self.config.img_buffer_size):
            next_file_id = file_id - t - 1
            next_file_path_rgb_front = os.path.join(folder_path, str(next_file_id).zfill(4) + ".jpg")

            if not os.path.exists(next_file_path_rgb_front):
                return past_img, False

            past_img[t] = {"path":next_file_path_rgb_front, "bytes":Image.open(next_file_path_rgb_front)}
        
        return past_img, True
    
    def get_future_img(self, file_id, folder_path):
        future_img = [None] * self.config.img_future_size
        for t in range(self.config.img_future_size):
            next_file_id = file_id + t
            next_file_path_rgb_front = os.path.join(folder_path, str(next_file_id).zfill(4) + ".jpg")

            if not os.path.exists(next_file_path_rgb_front):
                return future_img, False

            future_img[t] = {"path":next_file_path_rgb_front, "bytes":Image.open(next_file_path_rgb_front)}
        
        return future_img, True



    def convert_high_level_cmd_to_ego(self, high_level_cmd, current_gps_waypoint):
        high_level_cmd_ego = [None] * len(high_level_cmd)
        for t in range(len(high_level_cmd)):
            if not isinstance(high_level_cmd[t], numpy.ndarray):
                break
            cmd = high_level_cmd[t][2]
            ego_waypoint = self.transform_2d_points(
                numpy.zeros((1,3)),
                numpy.pi / 2 - 0.0,
                -high_level_cmd[t][0],
                -high_level_cmd[t][1],
                numpy.pi / 2 - current_gps_waypoint[2],
                -current_gps_waypoint[0],
                -current_gps_waypoint[1],
            )
            high_level_cmd_ego[t] = numpy.array([ego_waypoint[0,0], ego_waypoint[0,1], cmd])
        return high_level_cmd_ego

    ## MOD Minxuan: Get current + past command
    def get_cmd_list(self, file_path_measurements, file_id, folder_path_measurements):
        current_json = json.load(open(file_path_measurements))
        current_cmd = numpy.array([current_json["command"]])

        past_cmd, past_cmd_flag = self.get_past_cmd(file_id, folder_path_measurements)
        future_cmd, future_cmd_flag = self.get_future_cmd(file_id, folder_path_measurements)

        if future_cmd_flag * past_cmd_flag < 1:
            return None, False

        cmd_list = []
        for i in reversed(past_cmd):
            cmd_list.append(i[0])
        cmd_list.append(current_cmd)
        return cmd_list, True

    def get_past_cmd(self, file_id, folder_path):
        past_cmd = [None] * self.config.waypoint_buffer_size
        for t in range(self.config.waypoint_buffer_size):
            prev_file_id = file_id - t - 1
            prev_file_path_measurements = os.path.join(folder_path, str(prev_file_id).zfill(4) + ".json")

            if not os.path.exists(prev_file_path_measurements):
                return past_cmd, False
            
            prev_json = json.load(open(prev_file_path_measurements))
            past_cmd[t] = numpy.array([prev_json["command"]]).reshape((1,1))
        return past_cmd, True
    
    def get_future_cmd(self, file_id, folder_path):
        future_cmd = [None] * self.config.waypoint_prediction_size
        for t in range(self.config.waypoint_prediction_size):
            next_file_id = file_id + t + 1
            next_file_path_measurements = os.path.join(folder_path, str(next_file_id).zfill(4) + ".json")

            if not os.path.exists(next_file_path_measurements):
                return future_cmd, False
            next_json = json.load(open(next_file_path_measurements))
            future_cmd[t] = numpy.array([next_json["command"]]).reshape((1,1))

        return future_cmd, True

def convert_gps_to_ego(gps_waypoints, current_gps_waypoint):
    ego_waypoint = [None] * len(gps_waypoints)
    for t in range(len(gps_waypoints)):
        if not isinstance(gps_waypoints[t], numpy.ndarray):
            break

        ego_waypoint[t] = transform_2d_points(
            numpy.zeros((1,3)),
            numpy.pi / 2 - gps_waypoints[t][2],
                -gps_waypoints[t][0],
                -gps_waypoints[t][1],
                numpy.pi / 2 - current_gps_waypoint[2],
                -current_gps_waypoint[0],
                -current_gps_waypoint[1],
        )
    return ego_waypoint

def transform_2d_points(xyz, r1, t1_x, t1_y, r2, t2_x, t2_y):
    """
    Build a rotation matrix and take the dot product.
    """
    # z value to 1 for rotation
    xy1 = xyz.copy()
    xy1[:, 2] = 1
    xy1_front = xy1.copy()
    xy1_front[:,0] += 1.0

    c, s = numpy.cos(r1), numpy.sin(r1)
    r1_to_world = numpy.matrix([[c, s, t1_x], [-s, c, t1_y], [0, 0, 1]])

    # np.dot converts to a matrix, so we explicitly change it back to an array
    world = numpy.asarray(r1_to_world @ xy1.T)
    world_front = numpy.asarray(r1_to_world @ xy1_front.T)

    c, s = numpy.cos(r2), numpy.sin(r2)
    r2_to_world = numpy.matrix([[c, s, t2_x], [-s, c, t2_y], [0, 0, 1]])
    world_to_r2 = numpy.linalg.inv(r2_to_world)

    out = numpy.asarray(world_to_r2 @ world).T
    out_front = numpy.asarray(world_to_r2 @ world_front).T
    # reset z-coordinate
    out[:, 2] = numpy.arctan2(out_front[:,1]-out[:,1], out_front[:,0] - out[:,0])

    return out


if __name__ == '__main__':
    data_set = load_dataset("/scratch_net/biwidl216/rl_course_14/project/our_approach/decision-diffuser/code/diffuser/datasets/carla_dataset", "decdiff", streaming=True, split="train")
    data_set.shuffle(seed=42, buffer_size=50)



    preprocess = transforms.Compose(
        [
            transforms.Resize((125,125)),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(20),
            # transforms.RandomPerspective(distortion_scale=0.1, p = 0.3),
            # transforms.RandomGrayscale(p = 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
        ]
    )
    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["current_image"]]
        return {"current_image": images}

    dataloader = torch.utils.data.DataLoader(
        data_set, batch_size = 10, num_workers= 1)
    # print(next(dataloader))

    for i in data_set.iter(1):
        print(i)
    #     break