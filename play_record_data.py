import carla 
import sys

CARLA_ROOT="/home/marcus/Documents/Semester2/RobotLearning/InterFuser/carla"
sys.path.append(CARLA_ROOT+ "/PythonAPI")
sys.path.append(CARLA_ROOT+ "/PythonAPI/carla")
sys.path.append(CARLA_ROOT+ "/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg")
sys.path.append("/home/marcus/Documents/Semester2/RobotLearning/DiffuseDrive")

client = carla.Client("localhost", 2000)
client.replay_file("/home/marcus/Documents/Semester2/RobotLearning/InterFuser/carla/CarlaUE4/Saved/RouteScenario_16_rep0.log", 0, 100, 200)