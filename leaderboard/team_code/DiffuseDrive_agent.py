import os
import json
import datetime
import pathlib
import time
import imp
import cv2
import carla
from collections import deque

import torch
import carla
import numpy as np
from PIL import Image
from easydict import EasyDict

from torchvision import transforms
from leaderboard.autoagents import autonomous_agent
from timm.models import create_model
from team_code.utils import lidar_to_histogram_features, transform_2d_points
from team_code.planner import RoutePlanner
from leaderboard.team_code.DiffuseDrive_controller import DiffuseDriveController

import math
import yaml

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

    if need_scale:
        if input_size_num == 112:
            tfl.append(Resize2FixedSize((170, 128)))
        elif input_size_num == 128:
            tfl.append(Resize2FixedSize((195, 146)))
        elif input_size_num == 224:
            tfl.append(Resize2FixedSize((341, 256)))
        elif input_size_num == 256:
            tfl.append(Resize2FixedSize((288, 288)))
        else:
            raise ValueError("Can't find proper crop size")
    tfl.append(transforms.CenterCrop(img_size))
    tfl.append(transforms.ToTensor())
    tfl.append(transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)))

    return transforms.Compose(tfl)


class DiffuseDriveAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):

        self._hic = DisplayInterface()
        self.lidar_processed = list()
        self.track = autonomous_agent.Track.SENSORS
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False
        
        # TODO Marcus in theory it should not matter much what size we input into the evaluation network as it is resized anyway
        self.rgb_front_transform = create_carla_rgb_transform(224)
        
        self.rgb_left_transform = create_carla_rgb_transform(128)
        self.rgb_right_transform = create_carla_rgb_transform(128)
        self.rgb_center_transform = create_carla_rgb_transform(128, need_scale=False)

        # self.tracker = Tracker()

        self.input_buffer = {
            "rgb": deque(),
            "rgb_left": deque(),
            "rgb_right": deque(),
            "rgb_rear": deque(),
            "lidar": deque(),
            "gps": deque(),
            "thetas": deque(),
        }

        self.config = imp.load_source("MainModel", path_to_conf_file).GlobalConfig()
        self.skip_frames = self.config.skip_frames
        self.controller = DiffuseDriveController(self.config)

        # TODO Jacopo: load the model
        self.net = create_model(self.config.model)
        path_to_model_file = self.config.model_path
        print('load model: %s' % path_to_model_file)
        self.net.load_state_dict(torch.load(path_to_model_file)["state_dict"])
        self.net.cuda()
        self.net.eval()
        
        self.softmax = torch.nn.Softmax(dim=1)
        self.momentum = self.config.momentum
        self.prev_lidar = None
        self.prev_control = None
        self.prev_surround_map = None

        self.save_path = None
        
        
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
        
        # TODO Marcus: here we plan the route and take next waypoint and action 
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
            .cuda()
            .float()
        )
        rgb_left = (
            self.rgb_left_transform(Image.fromarray(tick_data["rgb_left"]))
            .unsqueeze(0)
            .cuda()
            .float()
        )
        rgb_right = (
            self.rgb_right_transform(Image.fromarray(tick_data["rgb_right"]))
            .unsqueeze(0)
            .cuda()
            .float()
        )
        rgb_center = (
            self.rgb_center_transform(Image.fromarray(tick_data["rgb"]))
            .unsqueeze(0)
            .cuda()
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
        
        
        
        # TODO Jacopo: adapt to our model
        with torch.no_grad():
            pred_waypoints = self.net(input_data)

        pred_waypoints = pred_waypoints.detach().cpu().numpy()[0]


        # call the controller to compute the control out of the generated
        # TODO Jacopo: adapt to our outputs
        steer, throttle, brake, meta_infos = self.controller.run_step(
            velocity,
            pred_waypoints,
        )

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

        tick_data["rgb_raw"] = tick_data["rgb"]
        tick_data["rgb_left_raw"] = tick_data["rgb_left"]
        tick_data["rgb_right_raw"] = tick_data["rgb_right"]


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
        surface = self._hic.run_interface(tick_data)
        tick_data["surface"] = surface

        if SAVE_PATH is not None:
            self.save(tick_data)

        return control

    def save(self, tick_data):
        frame = self.step // self.skip_frames
        Image.fromarray(tick_data["surface"]).save(
            self.save_path / "meta" / ("%04d.jpg" % frame)
        )
        return

    def destroy(self):
        if self.ensemble:
            del self.nets
        else:
            del self.net
