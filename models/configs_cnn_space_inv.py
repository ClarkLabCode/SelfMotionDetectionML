#!/usr/bin/env python

######################
# Configurations for data and models.
######################
from ml_collections import config_dict
import torch.nn as nn

import models


def get_config():
  """
  With spacial invariance and left-right symmetry
  """
  config = config_dict.ConfigDict()

  config.data_folder = '/home/bz242/project/data/'
  config.data_folder_local = '/mnt/d/data/'

  # get lists of hyper parameters
  config.L = 72 # width of the input array
  config.D_cnn_list = [1, 2, 3, 4, 5, 6] # list of depth
  config.C_list = [1, 2, 3, 4, 5, 6, 7, 8] # list of number of independent channels
  config.k_list = [3] # list of kernel size
  config.Repeat = 20 # Repeat * 5 repeats for different initializations

  config.D_cnn = 1 # default number of cnn layers
  config.C = 2 # default number of independent channels
  config.k = 3 # the default longer dimension of the kernel (in multiples of 5 deg)
  config.od = 2 # output dimension
  config.activationf = 'ReLU' # activation functions, which can be 'ReLU', 'LeakyReLU' or 'ELU'.
  config.T = 30 # length in time (in multiples of 10 ms)
  config.k_folds = 5
  config.num_epochs = 300
  config.batch_size = 100
  config.lr = 1e-3
  config.constraints_weight = models.weight_constraint_positive()
  config.constraints_bias = models.bias_constraint_negative()
  config.constrained_weight_layers = []
  config.constrained_bias_layers = []
  config.velocity_scale = 100
  config.training_data_path = config.data_folder + f'panoramic/processed_data_object_scal{config.velocity_scale}/train_test_wide_field/'
  config.testing_data_path = config.data_folder_local + f'panoramic/processed_data_object_scal{config.velocity_scale}/train_test_wide_field/'
  config.save_folder = f'../results/cnn_space_inv_scal{config.velocity_scale}/'
  config.save_folder_local = f'/mnt/d/research/counter_evidence/cnn_space_inv_scal{config.velocity_scale}/'
  config.figure_folder = f'../results/preliminary_figures_cnn_space_inv_scal{config.velocity_scale}/'
  config.loss_function = nn.CrossEntropyLoss()
  config.parallel = True
  
  return config
