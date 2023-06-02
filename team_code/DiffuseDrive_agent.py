import os
import json
import einops
import datetime
import pathlib
import time
# TODO Jacopo: imp is deprecated, substitute with importlib
import imp
import cv2
import carla
from collections import deque

import torch
import carla
import numpy as np
from PIL import Image

from torchvision import transforms
from leaderboard.autoagents import autonomous_agent
from team_code.utils import lidar_to_histogram_features, transform_2d_points
from team_code.planner import RoutePlanner
from team_code.DiffuseDrive_controller import DiffuseDriveController

import diffuser.utils as utils
from diffuser.datasets.carla_dataset.carla_dataset import convert_gps_to_ego

import math
import yaml
import pdb

try:
    import pygame
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")


SAVE_PATH = os.environ.get("SAVE_PATH", 'eval')
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class DisplayInterface(object):
    def __init__(self):
        self._width = 1200
        self._height = 600
        self._surface = None

        pygame.init()
        pygame.font.init()
        self._clock = pygame.time.Clock()
        self._display = pygame.display.set_mode(
            (self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("Human Agent")

    # TODO Marcus: have no clue of what this does
    def run_interface(self, input_data):
        rgb = input_data['rgb']
        rgb_left = input_data['rgb_left']
        rgb_right = input_data['rgb_right']
        rgb_focus = input_data['rgb_focus']
        map = input_data['map']
        surface = np.zeros((600, 1200, 3),np.uint8)
        surface[:, :800] = rgb
        surface[:400,800:1200] = map
        surface[400:600,800:1000] = input_data['map_t1']
        surface[400:600,1000:1200] = input_data['map_t2']
        surface[:150,:200] = input_data['rgb_left']
        surface[:150, 600:800] = input_data['rgb_right']
        surface[:150, 325:475] = input_data['rgb_focus']
        surface = cv2.putText(surface, input_data['control'], (20,580), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255), 1)
        surface = cv2.putText(surface, input_data['meta_infos'][0], (20,560), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255), 1)
        surface = cv2.putText(surface, input_data['meta_infos'][1], (20,540), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255), 1)
        surface = cv2.putText(surface, input_data['time'], (20,520), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255), 1)

        surface = cv2.putText(surface, 'Left  View', (40,135), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0), 2)
        surface = cv2.putText(surface, 'Focus View', (335,135), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0), 2)
        surface = cv2.putText(surface, 'Right View', (640,135), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0), 2)

        surface = cv2.putText(surface, 'Future Prediction', (940,420), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 2)
        surface = cv2.putText(surface, 't', (1160,385), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0), 2)
        surface = cv2.putText(surface, '0', (1170,385), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 2)
        surface = cv2.putText(surface, 't', (960,585), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0), 2)
        surface = cv2.putText(surface, '1', (970,585), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 2)
        surface = cv2.putText(surface, 't', (1160,585), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0), 2)
        surface = cv2.putText(surface, '2', (1170,585), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 2)

        surface[:150,198:202]=0
        surface[:150,323:327]=0
        surface[:150,473:477]=0
        surface[:150,598:602]=0
        surface[148:152, :200] = 0
        surface[148:152, 325:475] = 0
        surface[148:152, 600:800] = 0
        surface[430:600, 998:1000] = 255
        surface[0:600, 798:800] = 255
        surface[0:600, 1198:1200] = 255
        surface[0:2, 800:1200] = 255
        surface[598:600, 800:1200] = 255
        surface[398:400, 800:1200] = 255


        # display image
        self._surface = pygame.surfarray.make_surface(surface.swapaxes(0, 1))
        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))

        pygame.display.flip()
        pygame.event.get()
        return surface

    def _quit(self):
        pygame.quit()


def get_entry_point():
    return "DiffuseDriveAgent"

def get_config():
    
    # TODO Marcus: this probably won't work as the path is not correct. But cannot run it.
    class Parser(utils.Parser):
        dataset: str = 'carla-expert'
        config: str = 'config.carla'

    args = Parser().parse_args('diffusion')
    
    return args

def get_model(config, dataset):
    
    model_config = utils.Config(
        config.model,
        image_backbone = config.image_backbone,
        # savepath='model_config.pkl', # might be needed
        horizon=config.horizon,
        transition_dim=dataset.observation_dim + dataset.action_dim,
        cond_dim=dataset.observation_dim,
        dim_mults=config.dim_mults,
        returns_condition=config.returns_condition,
        dim=config.dim,
        condition_dropout=config.condition_dropout,
        calc_energy=config.calc_energy,
        device=config.device,
        past_image_cond = config.past_image_cond,
        image_backbone_freeze = config.image_backbone_freeze,
        using_cmd = config.using_cmd
        # attention??
    )
    
    diffusion_config = utils.Config(
        config.diffusion,
        savepath='diffusion_config.pkl',
        horizon=config.horizon,
        observation_dim=dataset.observation_dim,
        action_dim=dataset.action_dim,
        n_timesteps=config.n_diffusion_steps,
        loss_type=config.loss_type,
        clip_denoised=config.clip_denoised,
        predict_epsilon=config.predict_epsilon,
        hidden_dim=config.hidden_dim,
        ar_inv=config.ar_inv,
        train_only_inv=config.train_only_inv,
        ## loss weighting
        action_weight=config.action_weight,
        loss_weights=config.loss_weights,
        loss_discount=config.loss_discount,
        returns_condition=config.returns_condition,
        condition_guidance_w=config.condition_guidance_w,
        device=config.device,
    )
    
    model = model_config()

    diffusion = diffusion_config(model)
    
    return diffusion
    
def get_train_dataset(config):
    
    dataset_config = utils.Config(
        config.loader,
        savepath='dataset_config.pkl',
        env=config.dataset,
        horizon=config.horizon,
        normalizer=config.normalizer,
        preprocess_fns=config.preprocess_fns,
        use_padding=config.use_padding,
        max_path_length=config.max_path_length,
        include_returns=config.include_returns,
        returns_scale=config.returns_scale,
        discount=config.discount,
        termination_penalty=config.termination_penalty,
        past_image_cond = config.past_image_cond,
        waypoints_normalization = config.waypoints_normalization,
    )
    
    dataset = dataset_config()
    
    return dataset

class Resize2FixedSize:
    def __init__(self, size):
        self.size = size

    def __call__(self, pil_img):
        pil_img = pil_img.resize(self.size)
        return pil_img


def create_carla_rgb_transform(
    input_size, need_scale=True, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
):

    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = input_size
    tfl = []

    if isinstance(input_size, (tuple, list)):
        input_size_num = input_size[-1]
    else:
        input_size_num = input_size

    # if need_scale:
    #     if input_size_num == 112:
    #         tfl.append(Resize2FixedSize((170, 128)))
    #     elif input_size_num == 128:
    #         tfl.append(Resize2FixedSize((195, 146)))
    #     elif input_size_num == 224:
    #         tfl.append(Resize2FixedSize((341, 256)))
    #     elif input_size_num == 256:
    #         tfl.append(Resize2FixedSize((288, 288)))
    #     else:
    #         raise ValueError("Can't find proper crop size")
    # tfl.append(transforms.CenterCrop(img_size))
    tfl.append(Resize2FixedSize((128,128)))
    tfl.append(transforms.ToTensor())
    tfl.append(transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)))

    return transforms.Compose(tfl)


class DiffuseDriveAgent(autonomous_agent.AutonomousAgent):
    # TODO Jacopo: the path is probably still needed for compatibility with leaderboard_evaluator.py, but actually not used
    def setup(self, path_to_conf_file):
        # breakpoint()
        self._hic = DisplayInterface()
        self.lidar_processed = list()
        self.track = autonomous_agent.Track.SENSORS
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False
        self.skip_prediction_frame = 20
        self.past_pred_waypoints = None
        self.device = torch.device('cuda')
        
        # TODO Marcus in theory it should not matter much what size we input into the evaluation network as it is resized anyway. Also: the semantic segm. backbone we are using resizes the input anyway to a larger size.
        self.rgb_front_transform = create_carla_rgb_transform(224)
        
        self.rgb_left_transform = create_carla_rgb_transform(128)
        self.rgb_right_transform = create_carla_rgb_transform(128)
        self.rgb_center_transform = create_carla_rgb_transform(128, need_scale=False)

        # self.tracker = Tracker()

        self.past_timesteps = 3
        self.input_buffer = {
            "rgb": deque([None] * self.past_timesteps),
            "rgb_left": deque(),
            "rgb_right": deque(),
            "rgb_rear": deque(),
            "lidar": deque(),
            "gps": deque([None] * self.past_timesteps),
            "past_command": deque([None] * self.past_timesteps),
            "thetas": deque(),
        }

        # load the config file
        self.config = get_config()
        self.skip_frames = self.config.skip_frames
        self.controller = DiffuseDriveController(self.config)

        # normalization mean and std used for training
        # load train_dataset to get useful attributes and methods (e.g. normalization)
        self.train_dataset = get_train_dataset(self.config)
        self.waypoints_mean, self.waypoints_std = self.train_dataset.get_mean_std_waypoints()

        # load the model
        self.net = get_model(self.config, self.train_dataset)
        path_to_model_file = "/home/marcus/Documents/Semester2/RobotLearning/DiffuseDrive/trained_models/step_model_ema_cmd.pt"
        # print('load model: %s' % path_to_model_file)
        data = torch.load(path_to_model_file)
        self.net.load_state_dict(data['ema'])
        self.net.cuda()
        self.net.eval()
        self.net.to(self.device)
        
        self.softmax = torch.nn.Softmax(dim=1)
        
        self.prev_lidar = None
        self.prev_control = None
        self.prev_surround_map = None

        self.save_path = None
        
        self.past_image_cond = self.config.past_image_cond
        
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ["ROUTES"]).stem + "_"
            string += "_".join(
                map(
                    lambda x: "%02d" % x,
                    (now.month, now.day, now.hour, now.minute, now.second),
                )
            )

            print(string)

            self.save_path = pathlib.Path(SAVE_PATH) / string
            self.save_path.mkdir(parents=True, exist_ok=False)
            (self.save_path / "meta").mkdir(parents=True, exist_ok=False)

    def _init(self):
        self._route_planner = RoutePlanner(4.0, 50.0)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True

    def _get_position(self, tick_data):
        # TODO Marcus: probably here need to adapt to our coordinate system (ego)
        gps = tick_data["gps"]
        gps = (gps - self._route_planner.mean) * self._route_planner.scale
        return gps
    
    def get_past_waypoints(self):
        # TODO Marcus: get the 4 past waypoints used as condition for our model.
        # Not sure if this is needed or carla is able to give as data the past waypoints, so no need to store them in a buffer and retrieve them here.
        # There is already an input buffer, take a look at that.
        # raise NotImplementedError
        return self.input_buffer['gps']
    
    def get_past_images(self):
        # TODO Marcus: get the past images used as condition for our model.
        # Same as for waypoints, not sure if this is needed or carla is able to give as data the past images, so no need to store them in a buffer and retrieve them here. 
        # There is already an input buffer, take a look at that, maybe that is what we need.
        return self.input_buffer['rgb']
    
    def get_past_command(self):
        return self.input_buffer['past_command']
    
    
    def waypoints_normalizer(self, waypoints):   
        normalized_waypoints = (waypoints - self.waypoints_mean[:4]) / (self.waypoints_std[:4] + 1e-7)
        
        return normalized_waypoints
    
    
    def waypoints_denormalizer(self, normalized_waypoints):
        waypoints = normalized_waypoints * (self.waypoints_std[4:] + 1e-7) + self.waypoints_mean[4:]
        
        return waypoints
    
    def diffusedrive_image_processing(self, images):
        # TODO Marcus: normalize images. Same as done by the images_batch_norm function in training.py. Ideally we should try to avoid double code, i.e. refer to that function here.
        # raise NotImplementedError
        return images
    
    def carla2diffusedrive_data(self, input_data, past_image_cond, n_diff_trajectories = 1):
        # TODO Marcus: build the Batch object used for training
        # * normalize images
        # * normalized past waypoints
        # * need to check what format is coming from carla (I assume PIL)
        # * not sure how the "past waypoints" are handled by carla, e.g. if carla stores the past waypoints as input data and can then be used as input for our model, or if we need to have a buffer to store and retrieve them   
        # breakpoint()
        past_waypoints = self.get_past_waypoints()
        
        current_waypoint = np.array([input_data['gps'][0], input_data['gps'][1], input_data['compass']])
        condition = np.zeros((self.past_timesteps+1, 3))
        for t, past_waypoint in enumerate(past_waypoints):
            if past_waypoint is None:
                past_waypoint = current_waypoint.copy()
            past_waypoint_ego = convert_gps_to_ego([past_waypoint], current_waypoint)
            condition[t] = past_waypoint_ego[0]
        condition[-1] = np.zeros((1,3))

        # Normalize waypoints here
        norm_condition = self.waypoints_normalizer(condition)
        norm_condition = norm_condition[np.newaxis, :, : ]
        
        norm_condition = einops.repeat(norm_condition, 'b t d -> (repeat b) t d', repeat = n_diff_trajectories)
        
        norm_condition = torch.from_numpy(norm_condition)
        past_images = None
        
        if past_image_cond:
            current_image = input_data['rgb']
            past_images = self.get_past_images()

            norm_images = torch.zeros((1, self.past_timesteps+1, 3, 128, 128))
            for t, past_image in enumerate(past_images):
                if past_image is None:
                    past_image = current_image.clone()
                norm_images[:,t] = past_image
            norm_images[:,-1] = current_image

            # Image are already normalized 
            # norm_images = self.diffusedrive_image_processing(past_images) 
            
            norm_images = einops.repeat(norm_images, 'b t h w d -> (repeat b) t h w d', repeat = n_diff_trajectories)           
        

        if self.config.using_cmd:
            cond_cmd = [None] * (self.past_timesteps + 1)

            current_cmd = input_data["next_command"]
            past_cmd = self.get_past_command()
            for t, past_cmd in enumerate(past_cmd):
                if past_cmd is None:
                    past_cmd = current_cmd
                cond_cmd[t] = past_cmd
            cond_cmd[-1] = current_cmd
            
            cond_cmd = self.train_dataset.preprocess_cmd(cond_cmd)
            cond_cmd = torch.from_numpy(cond_cmd)
            cond_cmd = torch.unsqueeze(cond_cmd, 0)
            sample = (norm_condition, norm_images, cond_cmd)
        else:
            sample = (norm_condition, norm_images)

        # breakpoint()
        return sample
    
    
    def diffusedrive2carla_data(self, diffusedrive_data):
        # TODO Marcus
        # See render_samples function in training.py.
        # * discard past waypoints
        # * de-normalize waypoints
        
        # future waypoints
        norm_pred_waypoints = diffusedrive_data[:,4:,:]
        
        pred_waypoints = self.waypoints_denormalizer(norm_pred_waypoints)
        # raise NotImplementedError

        return pred_waypoints

    def sensors(self):
        return [
            {
                "type": "sensor.camera.rgb",
                "x": 1.3,
                "y": 0.0,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "width": 800,
                "height": 600,
                "fov": 100,
                "id": "rgb",
            },
            {
                "type": "sensor.camera.rgb",
                "x": 1.3,
                "y": 0.0,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -60.0,
                "width": 400,
                "height": 300,
                "fov": 100,
                "id": "rgb_left",
            },
            {
                "type": "sensor.camera.rgb",
                "x": 1.3,
                "y": 0.0,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 60.0,
                "width": 400,
                "height": 300,
                "fov": 100,
                "id": "rgb_right",
            },
            {
                "type": "sensor.lidar.ray_cast",
                "x": 1.3,
                "y": 0.0,
                "z": 2.5,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -90.0,
                "id": "lidar",
            },
            {
                "type": "sensor.other.imu",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.05,
                "id": "imu",
            },
            {
                "type": "sensor.other.gnss",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.01,
                "id": "gps",
            },
            {"type": "sensor.speedometer", "reading_frequency": 20, "id": "speed"},
        ]

    def tick(self, input_data):

        rgb = cv2.cvtColor(input_data["rgb"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_left = cv2.cvtColor(input_data["rgb_left"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_right = cv2.cvtColor(
            input_data["rgb_right"][1][:, :, :3], cv2.COLOR_BGR2RGB
        )
        gps = input_data["gps"][1][:2]
        speed = input_data["speed"][1]["speed"]
        compass = input_data["imu"][1][-1]
        if (
            math.isnan(compass) == True
        ):  # It can happen that the compass sends nan for a few frames
            compass = 0.0

        result = {
            "rgb": rgb,
            "rgb_left": rgb_left,
            "rgb_right": rgb_right,
            "gps": gps,
            "speed": speed,
            "compass": compass,
        }

        # breakpoint()
        pos = self._get_position(result)

        lidar_data = input_data['lidar'][1]
        result['raw_lidar'] = lidar_data

        lidar_unprocessed = lidar_data[:, :3]
        lidar_unprocessed[:, 1] *= -1
        full_lidar = transform_2d_points(
            lidar_unprocessed,
            np.pi / 2 - compass,
            -pos[0],
            -pos[1],
            np.pi / 2 - compass,
            -pos[0],
            -pos[1],
        )
        lidar_processed = lidar_to_histogram_features(full_lidar, crop=224)
        if self.step % 2 == 0 or self.step < 4:
            self.prev_lidar = lidar_processed
        result["lidar"] = self.prev_lidar

        result["gps"] = pos
        
        # TODO Marcus: here we plan the route and take next waypoint and action, I guess 
        next_wp, next_cmd = self._route_planner.run_step(pos)
        result["next_command"] = next_cmd.value
        result['measurements'] = [pos[0], pos[1], compass, speed]

        theta = compass + np.pi / 2
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        # TODO Marcus: not sure exactly why this is needed
        local_command_point = np.array([next_wp[0] - pos[0], next_wp[1] - pos[1]])
        local_command_point = R.T.dot(local_command_point)
        result["target_point"] = local_command_point

        return result

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        self.step += 1
        if self.step % self.skip_frames != 0 and self.step > 4:
            return self.prev_control

        tick_data = self.tick(input_data)

        velocity = tick_data["speed"]
        command = tick_data["next_command"]

        rgb = (
            self.rgb_front_transform(Image.fromarray(tick_data["rgb"]))
            .unsqueeze(0)
            .float()
        )
        rgb_left = (
            self.rgb_left_transform(Image.fromarray(tick_data["rgb_left"]))
            .unsqueeze(0)
            .float()
        )
        rgb_right = (
            self.rgb_right_transform(Image.fromarray(tick_data["rgb_right"]))
            .unsqueeze(0)
            .float()
        )
        rgb_center = (
            self.rgb_center_transform(Image.fromarray(tick_data["rgb"]))
            .unsqueeze(0)
            .float()
        )

        # TODO Marcus: what exactly are the categories of the one hot encoding?
        # TODO Marcus: might be interesting to feed also the velocity as input of our diffusion model?
        cmd_one_hot = [0, 0, 0, 0, 0, 0]
        cmd = command - 1
        cmd_one_hot[cmd] = 1
        cmd_one_hot.append(velocity)
        mes = np.array(cmd_one_hot)
        mes = torch.from_numpy(mes).float().unsqueeze(0).cuda()

        input_data = {}
        input_data["rgb"] = rgb
        input_data["rgb_left"] = rgb_left
        input_data["rgb_right"] = rgb_right
        input_data["rgb_center"] = rgb_center
        input_data["measurements"] = mes
        input_data["target_point"] = (
            torch.from_numpy(tick_data["target_point"]).float().cuda().view(1, -1)
        )
        input_data["lidar"] = (
            torch.from_numpy(tick_data["lidar"]).float().cuda().unsqueeze(0)
        )
        input_data['gps'] = tick_data['gps']
        input_data['compass'] = tick_data['compass']

        if self.config.using_cmd:
            input_data["next_command"] = np.array(tick_data["next_command"])

        # breakpoint()
        ################################## PREDICT FUTURE WAYPOINTS ##################################
        with torch.no_grad():
            
            # TODO Jacopo: how many trajectories should we sample? And how to handle them if more than one?
            # breakpoint()
            # TODO Marcus: implement carla-data to our expected input. diffusedrive_input should be a tuple of (conditions, images). See trainin.py at fuction render_samples.
            if self.past_pred_waypoints is None or self.step % self.skip_prediction_frame == 0:
                diffusedrive_input = self.carla2diffusedrive_data(input_data, self.past_image_cond)

                # forward pass samples trajectories (calling conditional_sample function)
                if self.config.using_cmd:
                    diffused_waypoints = self.net(cond = diffusedrive_input[0].to(self.device), images = diffusedrive_input[1].to(self.device), cmd = diffusedrive_input[2].to(self.device))
                else:
                    diffused_waypoints = self.net(cond = diffusedrive_input[0].to(self.device), images = diffusedrive_input[1].to(self.device))
                diffused_waypoints = diffused_waypoints.detach().cpu().numpy()
                # TODO Marcus: pred_waypoints should be non-normalized, with [0] being the current position (I think)
                pred_waypoints = self.diffusedrive2carla_data(diffused_waypoints)
                # TODO Marcus: not sure if this is needed
                pred_waypoints = pred_waypoints[0]
                # print(pred_waypoints)
            else:
                pred_waypoints = self.past_pred_waypoints
            
            self.past_pred_waypoints = pred_waypoints
            # pred_waypoints = np.array([[-5.0, -10.0]] * 10)
            # pred_waypoints[0] = np.zeros((1,2))
        ##############################################################################################


        # call the controller to compute the control out of the predicted waypoints
        steer, throttle, brake, meta_infos = self.controller.run_step(velocity, [pred_waypoints[0], pred_waypoints[2]])

        # TODO Jacopo: to check if this is needed.
        if brake < 0.05:
            brake = 0.0
        if brake > 0.1:
            throttle = 0.0

        # TODO Marcus: here you probably need to check if this works out of the box
        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)

        pred_waypoints = pred_waypoints.reshape(-1, 2)
        

        if self.step % 2 != 0 and self.step > 4:
            control = self.prev_control
        else:
            self.prev_control = control

        self.input_buffer['gps'].append(np.array([tick_data['gps'][0], tick_data['gps'][1], tick_data['compass']]))
        self.input_buffer['gps'].popleft()

        self.input_buffer['rgb'].append(input_data['rgb'])
        self.input_buffer['rgb'].popleft()

        if self.config.using_cmd:
            self.input_buffer['past_command'].append(input_data['next_command'])
            self.input_buffer['past_command'].popleft()

        tick_data["rgb_raw"] = tick_data["rgb"]
        tick_data["rgb_left_raw"] = tick_data["rgb_left"]
        tick_data["rgb_right_raw"] = tick_data["rgb_right"]

        # we probably do not need these resizings
        tick_data["rgb"] = cv2.resize(tick_data["rgb"], (800, 600))
        tick_data["rgb_left"] = cv2.resize(tick_data["rgb_left"], (200, 150))
        tick_data["rgb_right"] = cv2.resize(tick_data["rgb_right"], (200, 150))
        tick_data["rgb_focus"] = cv2.resize(tick_data["rgb_raw"][244:356, 344:456], (150, 150))
        
        
        tick_data["control"] = "throttle: %.2f, steer: %.2f, brake: %.2f" % (
            control.throttle,
            control.steer,
            control.brake,
        )
        # TODO Jacopo: i think this is not necessary
        tick_data["meta_infos"] = meta_infos
        
        tick_data["mes"] = "speed: %.2f" % velocity
        tick_data["time"] = "time: %.3f" % timestamp

        # Not used, only for visualization of interfuser
        # surface = self._hic.run_interface(tick_data)
        # tick_data["surface"] = surface
        
        # print(throttle)
        if SAVE_PATH is not None:
            self.save(tick_data)
        return control

    def save(self, tick_data):
        # frame = self.step // self.skip_frames
        # Image.fromarray(tick_data["surface"]).save(
        #     self.save_path / "meta" / ("%04d.jpg" % frame)
        # )
        return

    def destroy(self):
        del self.net
