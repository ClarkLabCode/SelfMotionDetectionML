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
from ml_collections import config_flags
from absl import app
from absl import flags

import models
import test
import helper_functions as hpfn

_FLAGS = flags.FLAGS

config_flags.DEFINE_config_file('config', None, 'configuration file', lock_config=True)
flags.mark_flags_as_required(['config'])

def main(_):
    ######## Testing data #######
    test_samples = np.load(_FLAGS.config.testing_data_path + f'test_samples.npy')
    test_targets = np.load(_FLAGS.config.testing_data_path + f'test_targets.npy')
    print(f'The shape of the testing samples are {test_samples.shape}.')
    print(f'The shape of the testing targets are {test_targets.shape}.')

    # Get an instance of the dataset class, and feed the loaded data.
    transformed_dataset = hpfn.RigidMotionDataset_ln(targets=test_targets, 
                                                     samples=test_samples,
                                                     transform=hpfn.ToTensor(),
                                                     T=_FLAGS.config.T)
    for R in range(_FLAGS.config.Repeat):
        for k in _FLAGS.config.k_list:
            for D_cnn in _FLAGS.config.D_cnn_list:
                for C in _FLAGS.config.C_list:
                    model_folder = _FLAGS.config.save_folder_local + f'Dcnn{D_cnn}_C{C}_k{k}_' + _FLAGS.config.activationf + f'_R{R+1}/'
                    for fold in range(1, _FLAGS.config.k_folds+1):
                        if os.path.exists(model_folder+f'model_fold_{fold}.pth'):
                            ####### Trained model #######
                            network_model = models.CNNSpaceInv(D_cnn=D_cnn, C=C, k=k, od=_FLAGS.config.od, activationf=_FLAGS.config.activationf, T=_FLAGS.config.T)
                            network_model.load_state_dict(torch.load(model_folder+f'model_fold_{fold}.pth'))

                            ####### Testing #######
                            start_time = time.time()
                            test.test_run(network_model, transformed_dataset, model_folder, fold)
                            print(f'Testing took {time.time()-start_time} for one fold.')

if __name__ == '__main__':
    app.run(main)


















