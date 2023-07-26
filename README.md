# DiffuseDrive 
#### Robot Learning Course - Spring Semester 2023, ETH Zurich
### Minxuan Qin, Marcus Leong, Jacopo Graldi
---
## Abstract
> Autonomous Driving has long been a deeply studied field in both academia and industry for its terrific societal and economic potential. It is however a difficult task that still has not reached a satisfactory level of trust and capability for a widespread deployment. An end-to-end, interpretable architecture - in contrast to the current state-of-the-art models - is needed for its pervasive adoption. In this project, we implement a proof-of-concept end-to-end Autonomous Driving pipeline based on Diffusion Models. Based on previous attempts at diffusing trajectories for planning pipelines, we employ diffusion models to generate and predict future waypoints and trajectories, conditioning on the past history of the vehicle such as past visual information, high-level commands, and waypoints. Despite lacking understanding and reasonable decision-making in complex traffic situations (e.g. at intersections), our agent is able to learn some basic behavior such as simple turning and lane-following actions. This demonstrates the interesting potential and opportunities given by Diffusion Models for Autonomous Driving, and we pave the way for an interpretable architecture.
---
## How to reproduce our project
* Install the requirements in ```requirement.txt``` with a virtual environment 
* Follow the same installation described in the [Interfuser repository](https://github.com/opendilab/InterFuser) for CARLA simulator and data collection
* To run the training routine, run ```scripts/train.py``` with Python 3.8 or above.
* Configuration of training is in ```config/carla.py```
* To evaluate the model after training, run the script ```leaderboard/scripts/run_evaluation.sh```
* We provide a trained checkpoint of the model (as described in the report) TODO
---
## Further Comments
* Our repository is based on the [Decision Diffuser](https://github.com/anuragajay/decision-diffuser/tree/main/code). 
* Our adaptations and extensions (e.g. classes or functions), all include "Carla" in the name.
* Data handling is specific to our data collection and therefore needs adaptation
* Further details on motivation, architecture, training, and results, can be found in the project report ```DiffuseDrive.pdf```
