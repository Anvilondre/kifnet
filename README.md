# KIFNet: Continuous Prediction of Leg Kinematics during Walking using Inertial Sensors, Smart Glasses, and Embedded Computing

This is a repository, containing training source code we used for [our paper](https://drive.google.com/file/d/1VJ9a6I215R_XjWRcR6_4BoY7WIcNk8FG/view) in ICRA 2023.  

In order to run our code you will need:
1. [Egocentric Vision & Kinematics dataset](https://github.com/abs711/The-way-of-the-future). Download it and change appropriate data config files to match your paths.
2. [ml-mobileone](https://github.com/apple/ml-mobileone) package and model weights (we used S0 unfused), that you need to place in `./ml-mobileone/weights/` folder.
3. An existing [Weight & Biases](https://wandb.ai/) project. We used W&B for experiment tracking and configuration, hence it is required to run our pipeline if you are not willing to modify the code.

Contributors:  
* [Oleksii Tsepa](https://github.com/imgremlin)  
* [Roman Burakov](https://github.com/Anvilondre)
