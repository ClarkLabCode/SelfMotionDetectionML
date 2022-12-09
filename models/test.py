#!/usr/bin/env python


import sys
sys.path.append('../helper/')

import os
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import models
import helper_functions as hpfn


def test_run(network_model, transformed_dataset, model_folder, fold):
    # Define data loaders for training and validing data in this fold
    testloader = torch.utils.data.DataLoader(
                      transformed_dataset, 
                      batch_size=100)

    y_test_all = []
    y_pred_all = []
    for i, data in enumerate(testloader):
        # Get inputs
        inputs, targets = data['movie'], data['target']
        outputs = network_model(inputs)
        # print(len(outputs), outputs[0].tolist())
        y_pred_all.extend([output.tolist() for output in outputs])
        y_test_all.extend([target.item() for target in targets])
        
    # Saving the test results
    save_path = model_folder + f'y_test_all_fold_{fold}.pth'
    torch.save(y_test_all, save_path)
    save_path = model_folder + f'y_pred_all_fold_{fold}.pth'
    torch.save(y_pred_all, save_path)













