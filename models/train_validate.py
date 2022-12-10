#!/usr/bin/env python

# Example code from 
# https://www.machinecurve.com/index.php/2021/02/03/how-to-use-k-fold-cross-validation-with-pytorch/
# By Christian Versloot
# but with modifications


import sys
sys.path.append('../helper/')

import os
import numpy as np
import glob
from sklearn.model_selection import KFold
import time
from datetime import datetime
from joblib.externals.loky.backend.context import get_context

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import models
import helper_functions as hpfn


# Function to reset the weights before the training
def reset_weights(m):
    '''
      Try resetting model weights to avoid
      weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


# training and validating function
def train_valid_run(kfold_info, config, network_model, transformed_dataset, loss_function):

    # Configuration options
    k_folds = config.k_folds
    num_epochs = config.num_epochs
    batch_size = config.batch_size
    lr = config.lr
    constraints_weight = config.constraints_weight
    constraints_bias = config.constraints_bias
    constrained_weight_layers = config.constrained_weight_layers
    constrained_bias_layers = config.constrained_bias_layers
    scal = config.velocity_scale
    training_data_path = config.training_data_path
    save_folder = config.save_folder

    # K-fold Cross Validation model evaluation
    fold = kfold_info['fold']
    train_ids = kfold_info['train_ids']
    valid_ids = kfold_info['valid_ids']

    # Save config info to a log file
    if fold == 1:
        log_file = save_folder + 'log.txt'
        with open(log_file, 'w') as f:
            f.write('Model setup:\n')
            f.write('----------------------------------------\n')
            f.write(f'network_model: {network_model}\n')
            f.write(f'constraints_weight: {constraints_weight}\n')
            f.write(f'constraints_bias: {constraints_bias}\n')
            f.write(f'constrained_weight_layers: {constrained_weight_layers}\n')
            f.write(f'constrained_bias_layers: {constrained_bias_layers}\n')
            f.write(f'k_folds: {k_folds}\n')
            f.write(f'num_epochs: {num_epochs}\n')
            f.write(f'batch_size: {batch_size}\n')
            f.write(f'lr: {lr}\n')
            f.write(f'velocity_scale: {scal}\n')
            f.write(f'training data path: {training_data_path}\n')

    start_time_fold = time.time()

    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_ids)

    # Define data loaders for training and validing data in this fold
    if config.parallel:
        trainloader = torch.utils.data.DataLoader(
                          transformed_dataset, 
                          batch_size=batch_size, sampler=train_subsampler, num_workers=1, 
                          multiprocessing_context=get_context('loky'))
        validloader = torch.utils.data.DataLoader(
                          transformed_dataset,
                          batch_size=batch_size, sampler=valid_subsampler, num_workers=1, 
                          multiprocessing_context=get_context('loky'))
    else:
        trainloader = torch.utils.data.DataLoader(
                          transformed_dataset, 
                          batch_size=batch_size, sampler=train_subsampler, num_workers=1, 
                          multiprocessing_context=None)
        validloader = torch.utils.data.DataLoader(
                          transformed_dataset,
                          batch_size=batch_size, sampler=valid_subsampler, num_workers=1, 
                          multiprocessing_context=None)

    # Init the neural network
    network_model.apply(reset_weights)
    # Re-Init the constrained parameters
    for constrained_weight_layer in constrained_weight_layers:
        nn.init.uniform_(network_model._modules[constrained_weight_layer].weight, 0, 1)
    for constrained_bias_layer in constrained_bias_layers:
        nn.init.uniform_(network_model._modules[constrained_bias_layer].bias, -1, 0)

    # Saving the initialized model
    save_path = save_folder + f'model_init_fold_{fold}.pth'
    torch.save(network_model.state_dict(), save_path)

    # Initialize optimizer
    optimizer = torch.optim.Adam(network_model.parameters(), lr=lr, weight_decay=0.01)

    # List to save train losses
    train_loss = []
    valid_loss = []
    
    ######## Training and Validation #######
    print('Training and validation:')

    # Run the training and validating loop for defined number of epochs
    for epoch in range(num_epochs):
        start_time = time.time()
        
        ######## Training #######
        train_loss_epoch = 0
        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader):
            # Get inputs
            inputs, targets = data['movie'], data['target']

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = network_model(inputs)
            
            # Compute loss
            loss = loss_function(outputs, targets[:, 0])

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            # Apply constraints
            for constrained_weight_layer in constrained_weight_layers:
                network_model._modules[constrained_weight_layer].apply(constraints_weight)
            for constrained_bias_layer in constrained_bias_layers:
                network_model._modules[constrained_bias_layer].apply(constraints_bias)

            train_loss_epoch = train_loss_epoch + loss.item()

        train_loss_epoch = train_loss_epoch / (i+1)
        train_loss.append(train_loss_epoch)

        ######## Validation #######
        with torch.no_grad():

            valid_loss_epoch = 0
            # Iterate over the valid data and generate predictions
            for i, data in enumerate(validloader):

                # Get inputs
                inputs, targets = data['movie'], data['target']

                # Generate outputs
                outputs = network_model(inputs)

                # Compute loss
                loss = loss_function(outputs, targets[:, 0])

                valid_loss_epoch = valid_loss_epoch + loss.item()

            valid_loss_epoch = valid_loss_epoch / (i+1)
            valid_loss.append(valid_loss_epoch)

        # Print statistics
        print(f'Fold {fold}: Training loss for epoch {epoch+1} is {train_loss_epoch}')
        print(f'Fold {fold}: Validating loss for epoch {epoch+1} is {valid_loss_epoch}')
        print(f'Fold {fold}: Time for epoch {epoch+1} is {time.time()-start_time}')

    # Saving the model
    save_path = save_folder + f'model_fold_{fold}.pth'
    torch.save(network_model.state_dict(), save_path)
    
    # Saving the loss function, training
    save_path = save_folder + f'train_loss_fold_{fold}.pth'
    torch.save(train_loss, save_path)

    # Saving the loss function, validating
    save_path = save_folder + f'valid_loss_fold_{fold}.pth'
    torch.save(valid_loss, save_path)

    ######## Final Validation #######
    print('Final validation:')

    # Evaluation for this fold
    with torch.no_grad():

        valid_loss = 0
        # Iterate over the valid data and generate predictions
        for i, data in enumerate(validloader):

            # Get inputs
            inputs, targets = data['movie'], data['target']

            # Generate outputs
            outputs = network_model(inputs)

            # Compute loss
            loss = loss_function(outputs, targets[:, 0])

            valid_loss = valid_loss + loss.item()

        valid_loss = valid_loss / (i+1)
                      
        # Print loss
        print(f'Loss for fold {fold}: {valid_loss}.')

        # Saving the loss function, validating
        save_path = save_folder + f'valid_loss_fold_{fold}_final.pth'
        torch.save(valid_loss, save_path)

    # Print training and validating time for the current fold
    print(f'The training and validation of fold {fold} took {time.time()-start_time_fold}.')












