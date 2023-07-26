`# DiffuseDrive 
#### Robot Learning Course - Spring Semester 2023, ETH Zurich
### Minxuan Qin, Marcus Leong, Jacopo Graldi
---
## How to reproduce our project
* Install the requirements in ```requirement.txt``` with a virtual environment 
* Follow the same installation described in the [Interfuser repository](https://github.com/opendilab/InterFuser) for CARLA simulator and data collection
* To run the training routine, run ```scripts/train.py``` with Python 3.8 or above.
* Configuration of training is in ```config/carla.py```
* To evaluate the model after training, run the script ```leaderboard/scripts/run_evaluation.sh```
* We provide a trained checkpoint of the model (as described in the report) TODO

## Further Comments
* Our repository is based on the [Decision Diffuser](https://github.com/anuragajay/decision-diffuser/tree/main/code). 
* Our adaptations and extensions (e.g. classes or functions), all include "Carla" in the name.
* Data handling is specific to our data collection and therefore needs adaptation
* Further details on motivation, architecture, training, and results, can be found in the project report ```DiffuseDrive.pdf```
