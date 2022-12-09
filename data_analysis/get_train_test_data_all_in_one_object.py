#!/usr/bin/env python

'''
This script prepares the data for training and testing. 
The whole training and testing dataset can be loaded into RAM at the same time.
'''


import sys
sys.path.append('../helper/')

import os
import numpy as np

n_row = 1
n_col = 72
T = 30
scal = 100 # one standard deviation for the velocity, in degree/s
sample_per_image = 1 # number of 1-by-72 samples from one movie

# data_folder = '/Volumes/Baohua/data_on_hd/' # local
data_folder = '/home/bz242/project/data/' # cluster
folder_processed = data_folder + f'panoramic/processed_data_object_scal{scal}/'
if not os.path.exists(folder_processed + 'train_test_wide_field/'):
    os.makedirs(folder_processed + 'train_test_wide_field/')

# Save config info to a log
log_file = folder_processed + 'train_test_wide_field/log.txt'
with open(log_file, 'w') as f:
    f.write('Data parameters:\n')
    f.write('----------------------------------------\n')
    f.write(f'Number of rows: {n_row}\n')
    f.write(f'Number of columns: {n_col}\n')
    f.write(f'Total length of the sample in time: {T}\n')
    f.write(f'scale of the velocity: {scal}\n')
    f.write(f'samples per image: {sample_per_image}\n')


####### Training data #######
targets_file = folder_processed + 'y_train_all.npy'
targets_file_object = folder_processed + 'y_train_all_object.npy'
root_dir = folder_processed + 'whole_frame_normalized_train/'
targets = np.load(targets_file)
targets_object = np.load(targets_file_object)
N = len(targets)

train_samples = np.zeros((2*N*sample_per_image, T, n_row, n_col))
train_targets = np.zeros((2*N*sample_per_image, 1))
ind = 0
for n in range(N):
    # self motion
    movie_name = os.path.join(root_dir, f'X_train_{n}.npy')
    movie = np.load(movie_name)
    target = targets[n]
    for ii in range(sample_per_image):
        row_start = np.random.randint(0, movie.shape[1]-n_row+1, 1)[0]
        col_start = 0
        sample = movie[:, row_start:row_start+n_row, col_start:col_start+n_col]
        train_samples[ind] = sample
        train_targets[ind] = target
        ind = ind + 1
    # object motion
    movie_name = os.path.join(root_dir, f'X_train_object_{n}.npy')
    movie = np.load(movie_name)
    target = targets_object[n]
    for ii in range(sample_per_image):
        row_start = np.random.randint(0, movie.shape[1]-n_row+1, 1)[0]
        col_start = 0
        sample = movie[:, row_start:row_start+n_row, col_start:col_start+n_col]
        train_samples[ind] = sample
        train_targets[ind] = target
        ind = ind + 1

print(f'Array shape of the targets: {train_targets.shape}')
print(f'Array shape of the samples: {train_samples.shape}')

np.save(folder_processed + 'train_test_wide_field/train_samples', train_samples)
np.save(folder_processed + 'train_test_wide_field/train_targets', train_targets)


####### Testing data #######
targets_file = folder_processed + 'y_test_all.npy'
targets_file_object = folder_processed + 'y_test_all_object.npy'
root_dir = folder_processed + 'whole_frame_normalized_test/'
targets = np.load(targets_file)
targets_object = np.load(targets_file_object)
N = len(targets)

test_samples = np.zeros((2*N*sample_per_image, T, n_row, n_col))
test_targets = np.zeros((2*N*sample_per_image, 1))
ind = 0
for n in range(N):
    # self motion
    movie_name = os.path.join(root_dir, f'X_test_{n}.npy')
    movie = np.load(movie_name)
    target = targets[n]
    for ii in range(sample_per_image):
        row_start = np.random.randint(0, movie.shape[1]-n_row+1, 1)[0]
        col_start = 0
        sample = movie[:, row_start:row_start+n_row, col_start:col_start+n_col]
        test_samples[ind] = sample
        test_targets[ind] = target
        ind = ind + 1
    # object motion
    movie_name = os.path.join(root_dir, f'X_test_object_{n}.npy')
    movie = np.load(movie_name)
    target = targets_object[n]
    for ii in range(sample_per_image):
        row_start = np.random.randint(0, movie.shape[1]-n_row+1, 1)[0]
        col_start = 0
        sample = movie[:, row_start:row_start+n_row, col_start:col_start+n_col]
        test_samples[ind] = sample
        test_targets[ind] = target
        ind = ind + 1

print(f'Array shape of the targets: {test_targets.shape}')
print(f'Array shape of the samples: {test_samples.shape}')

np.save(folder_processed + 'train_test_wide_field/test_samples', test_samples)
np.save(folder_processed + 'train_test_wide_field/test_targets', test_targets)





