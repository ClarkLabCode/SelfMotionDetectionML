# SelfMotionDetectionML
Code for training a small visual ANN to distinguish self rotation from object motion.

#### Data

The natural scene images used in the ANN training were adapted from an online data base 'Panoramic high dynamic range images in 
diverse environments'. The code for preprocessing of the data can be found in the folder **data_analysis**.

#### Model and training

The neural network models were standard CNNs constrained by spacial invariance, where filtered signals from different locations in space were summed uniformly. The code for model architectures and training can be found in the folder **models**.
