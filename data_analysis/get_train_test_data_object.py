#!/usr/bin/env python

'''
This script generates the data for motion detection.
This is not designed to run in parallel, and could take about half an hour to run.
'''


import sys
sys.path.append('../helper/')

import os
import numpy as np
import glob
import random
import time

import helper_functions as hpfn


start_time = time.time()

# data_folder = '/mnt/d/data/' # local
data_folder = '/home/bz242/project/data/' # cluster


#############################
####### Self rotation #######
#############################

folder_nat = data_folder + 'panoramic/data_natural_only_filtered/'
items = glob.glob(folder_nat+'*.npy')
random.shuffle(items)
N_images = len(items)
print('number of images are: ', N_images)

img=np.load(items[0])
K_row = img.shape[0] # in degree
K_col = img.shape[1] # in degree

####### Generate a list of velocities #######
gamma1 = np.log(2.) / 0.2 # half-life is 10000 second, this is to generate a roughly constant speed rotation.
delta_t = 0.01 # simulation step is 0.01 second
acc_length = 100 # length of the acceleration array
vel_length = 29 # length of the velocity array
scal = 100 # one standard deviation for the velocity, in degree/s
acc_mean = 0 # mean of the acceleration, in degree/s^2
acc_std = scal * np.sqrt(2 * gamma1 / (1 - np.exp(-2 * gamma1 * acc_length * delta_t)) / delta_t) # standard deviation of the acceleration, in degree/s^2
N_vel_per_img = 200 # different velocities for one image
N_vel = N_vel_per_img * N_images # total different velocities

folder_processed = data_folder + f'panoramic/processed_data_object_scal{scal}/'

vel_array_list = []
for nv in range(N_vel):
    acc_array = np.random.normal(acc_mean, acc_std, acc_length)
    initial_val = np.random.normal(0, scal)
    vel_array = hpfn.get_filtered_OU_1d(gamma1, delta_t, acc_array, vel_length, initial_val)
    vel_array_list.append(vel_array)
vel_array_list = np.array(vel_array_list)
print(f'The shape of the velocity array is {vel_array_list.shape}.')
savepath = folder_processed + 'vel_array_list.npy'
if not os.path.exists(folder_processed + ''):
    os.makedirs(folder_processed + '')
np.save(savepath, vel_array_list)


####### Subsample the images and generate the movies #######
# Import velocity traces
vel_array_list = np.load(folder_processed + 'vel_array_list.npy')

# Normal order
if not os.path.exists(folder_processed + 'whole_frame/'):
    os.makedirs(folder_processed + 'whole_frame/')
ind = 0
for item in items:
    img = np.load(item)
    for ii in range(N_vel_per_img):
        _, shift_array_pix = hpfn.get_shift_array(vel_array_list[ind], delta_t, img)
        img_processed = []
        for shift in shift_array_pix:
            img_roll = np.roll(img, shift, axis=1)
            img_resized = hpfn.get_resized(img_roll)
            img_processed.append(img_resized)
        img_processed = np.array(img_processed)
        np.save(folder_processed + 'whole_frame/img_processed_{}.npy'.format(ind+1), img_processed)
        ind = ind + 1
        
# Reversed order to enforce symmetry
if not os.path.exists(folder_processed + 'whole_frame_reversed/'):
    os.makedirs(folder_processed + 'whole_frame_reversed/')
ind = 0
for item in items:
    img = np.load(item)
    for ii in range(N_vel_per_img):
        _, shift_array_pix = hpfn.get_shift_array(vel_array_list[ind], delta_t, img)
        img_processed = []
        for shift in shift_array_pix:
            img_roll = np.roll(np.flip(img, axis=1), -shift, axis=1)
            img_resized = hpfn.get_resized(img_roll)
            img_processed.append(img_resized)
        img_processed = np.array(img_processed)
        np.save(folder_processed + 'whole_frame_reversed/img_reversed_{}.npy'.format(ind+1), img_processed)
        ind = ind + 1
        
        
####### Pair the movies with the velocities #######
# Import velocity traces 
vel_array_list = np.load(folder_processed + 'vel_array_list.npy')
print(vel_array_list.shape)
N = len(vel_array_list)
print('Half sample size is ', N)

# Normal samples
samples1 = []
for n in range(N):
    path = folder_processed + f'whole_frame/img_processed_{n+1}.npy'
    samples1.append([path, vel_array_list[n][-1]])

# Reversed samples
samples2 = []
for n in range(N):
    path = folder_processed + f'whole_frame_reversed/img_reversed_{n+1}.npy'
    samples2.append([path, -vel_array_list[n][-1]])
    
    
####### Divid data into training and testing set #######
N_train = int(180 * N_vel_per_img * 2)
samples_training1 = samples1[:int(N_train/2)]
samples_training2 = samples2[:int(N_train/2)]
samples_testing1 = samples1[int(N_train/2):]
samples_testing2 = samples2[int(N_train/2):]

N_train = int(len(samples_training1)+len(samples_training2))
N_test = int(len(samples_testing1)+len(samples_testing2))
print('Training sample size is ', N_train)
print('Testing sample size is ', N_test)
        
        
####### Data normalization, training #######
if not os.path.exists(folder_processed + 'whole_frame_normalized_train/'):
    os.makedirs(folder_processed + 'whole_frame_normalized_train/')
y_train_all = []
for nt in range(int(N_train/2)):
    # normal samples
    X_train = np.load(samples_training1[nt][0])
    X_train = hpfn.get_standardized_row(X_train)
    y_train = 1
    y_train_all.append(y_train)
    np.save(folder_processed + 'whole_frame_normalized_train/X_train_{}.npy'.format(int(nt*2)), X_train)
    # reversed samples
    X_train = np.load(samples_training2[nt][0])
    X_train = hpfn.get_standardized_row(X_train)
    y_train = 1
    y_train_all.append(y_train)
    np.save(folder_processed + 'whole_frame_normalized_train/X_train_{}.npy'.format(int(nt*2+1)), X_train)
y_train_all = np.array(y_train_all)
np.save(folder_processed + 'y_train_all.npy', y_train_all)


####### Data normalization, testing #######
if not os.path.exists(folder_processed + 'whole_frame_normalized_test/'):
    os.makedirs(folder_processed + 'whole_frame_normalized_test/')
y_test_all = []
for nt in range(int(N_test/2)):
    # normal samples
    X_test = np.load(samples_testing1[nt][0])
    X_test = hpfn.get_standardized_row(X_test)
    # y_test = samples_testing1[nt][1]
    y_test = 1
    y_test_all.append(y_test)
    np.save(folder_processed + 'whole_frame_normalized_test/X_test_{}.npy'.format(int(nt*2)), X_test)
    # reversed samples
    X_test = np.load(samples_testing2[nt][0])
    X_test = hpfn.get_standardized_row(X_test)
    # y_test = samples_testing2[nt][1]
    y_test = 1
    y_test_all.append(y_test)
    np.save(folder_processed + 'whole_frame_normalized_test/X_test_{}.npy'.format(int(nt*2+1)), X_test)
y_test_all = np.array(y_test_all)
np.save(folder_processed + 'y_test_all.npy', y_test_all)


#############################
####### Object motion #######
#############################


folder_nat = data_folder + 'panoramic/data_natural_only/'
items = glob.glob(folder_nat+'*.npy')
random.shuffle(items)
N_images = len(items)
print('number of images are: ', N_images)

img=np.load(items[0])
K_row = img.shape[0] # in degree
K_col = img.shape[1] # in degree

pix_per_deg = K_col / 360
FWHM = 5 # in degree
sigma_for_gaussian = np.round(FWHM/(2*np.sqrt(2*np.log(2)))*pix_per_deg, 1) # smoothing gaussian
pad_size = int(4*sigma_for_gaussian) # this comes from the fact that the gaussian is truncated at 4*std

####### Generate a list of velocities #######
gamma1 = np.log(2.) / 0.2 # half-life is 10000 second, this is to generate a roughly constant speed rotation.
delta_t = 0.01 # simulation step is 0.01 second
acc_length = 100 # length of the acceleration array
vel_length = 29 # length of the velocity array
scal = 100 # one standard deviation for the velocity, in degree/s
acc_mean = 0 # mean of the acceleration, in degree/s^2
acc_std = scal * np.sqrt(2 * gamma1 / (1 - np.exp(-2 * gamma1 * acc_length * delta_t)) / delta_t) # standard deviation of the acceleration, in degree/s^2
N_vel_per_img = 200 # different velocities for one image
N_vel = N_vel_per_img * N_images # total different velocities
N_max = 10 # upper limit of the number of moving objects

vel_array_list = []
for nv in range(N_vel):
    acc_array = np.random.normal(acc_mean, acc_std, acc_length)
    initial_val = np.random.normal(0, scal)
    vel_array = hpfn.get_filtered_OU_1d(gamma1, delta_t, acc_array, vel_length, initial_val)
    vel_array_list.append(vel_array)
vel_array_list = np.array(vel_array_list)
print(f'The shape of the velocity array is {vel_array_list.shape}.')
savepath = folder_processed + 'vel_array_list_object.npy'
if not os.path.exists(folder_processed + ''):
    os.makedirs(folder_processed + '')
np.save(savepath, vel_array_list)


####### Subsample the images and generate the movies #######
# Import velocity traces
vel_array_list = np.load(folder_processed + 'vel_array_list_object.npy')

# Normal order and reversed order to enforce symmetry
if not os.path.exists(folder_processed + 'whole_frame/'):
    os.makedirs(folder_processed + 'whole_frame/')
if not os.path.exists(folder_processed + 'whole_frame_reversed/'):
    os.makedirs(folder_processed + 'whole_frame_reversed/')
ind = 0
for item in items:
    img = np.load(item)
    for ii in range(N_vel_per_img):
        N = np.random.randint(1, N_max+1)
        Ls = 5 + 31 * np.random.random(N) # angular spans of the objects, in degree
        X0s = 360 * np.random.random(N) # initial positions of the objects, in degree (leftmost (rightmost) edge is 0 (360).).
        
        row = np.random.randint(0, 20)
        row_raw = int(np.floor(row*12.5)) # 12.5 ~ 251 / 20, where 251 is the first dimension of the image
        row_patch = np.random.randint(0, 20)
        while row_patch == row:
            row_patch = np.random.randint(0, 20)
        row_patch_raw = int(np.floor(row_patch*12.5)) # 12.5 ~ 251 / 20, where 251 is the first dimension of the image

        # Get the image patches as the objects
        image_patches = []
        L1 = img.shape[0]
        L2 = img.shape[1]
        xl = 0
        for n in range(N):
            l = Ls[n]
            xr = int(np.floor(l / 360 * L2))
            yu_patch = row_patch
            yd_patch = row_patch + int(np.floor(5 / 97 * L1))
            shift = np.random.randint(0, L2)
            image_patch = np.roll(img, shift, axis=1)[yu_patch:yd_patch, :xr]
            image_patches.append(image_patch)

        shift_array_deg, _ = hpfn.get_shift_array(vel_array_list[ind], delta_t, img)

        img_processed = []
        img_reversed = []
        for shift in shift_array_deg:
            img_array = img.copy()
            mask_in_x_union = np.zeros(img_array.shape[1])
            for n in range(N):
                x = X0s[n]
                l = Ls[n]
                img_array, mask_in_x = hpfn.add_object_to_image_periodic_boundary(img_array, x+shift, l, row_raw, image_patches[n])
                mask_in_x_union = mask_in_x_union + mask_in_x
            mask_in_x_union[mask_in_x_union!=0] = 1
            img_array = hpfn.get_filtered_spacial(img_array, pad_size, sigma_for_gaussian)
            img_resized = hpfn.get_resized(img_array)
            img_processed.append(img_resized[row:row+1, :])
            img_reversed.append(np.flip(img_resized[row:row+1, :]))
        img_processed = np.array(img_processed)
        img_reversed = np.array(img_reversed)
        np.save(folder_processed + 'whole_frame/mask_in_x_union_{}.npy'.format(ind+1), mask_in_x_union)
        np.save(folder_processed + 'whole_frame/img_processed_object_{}.npy'.format(ind+1), img_processed)
        np.save(folder_processed + 'whole_frame_reversed/img_reversed_object_{}.npy'.format(ind+1), img_reversed)

        ind = ind + 1
        
        
####### Pair the movies with the velocities #######
# Import velocity traces 
vel_array_list = np.load(folder_processed + 'vel_array_list_object.npy')
print(vel_array_list.shape)
N = len(vel_array_list)
print('Half sample size is ', N)

# Normal samples
samples1 = []
for n in range(N):
    path = folder_processed + f'whole_frame/img_processed_object_{n+1}.npy'
    samples1.append([path, vel_array_list[n][-1]])

# Reversed samples
samples2 = []
for n in range(N):
    path = folder_processed + f'whole_frame_reversed/img_reversed_object_{n+1}.npy'
    samples2.append([path, -vel_array_list[n][-1]])
    
    
    
####### Divid data into training and testing set #######
N_train = int(180 * N_vel_per_img * 2)
samples_training1 = samples1[:int(N_train/2)]
samples_training2 = samples2[:int(N_train/2)]
samples_testing1 = samples1[int(N_train/2):]
samples_testing2 = samples2[int(N_train/2):]

N_train = int(len(samples_training1)+len(samples_training2))
N_test = int(len(samples_testing1)+len(samples_testing2))
print('Training sample size is ', N_train)
print('Testing sample size is ', N_test)
        
        
####### Data normalization, training #######
if not os.path.exists(folder_processed + 'whole_frame_normalized_train/'):
    os.makedirs(folder_processed + 'whole_frame_normalized_train/')
y_train_all = []
for nt in range(int(N_train/2)):
    # normal samples
    X_train = np.load(samples_training1[nt][0])
    X_train = hpfn.get_standardized_row(X_train)
    y_train = 0
    y_train_all.append(y_train)
    np.save(folder_processed + 'whole_frame_normalized_train/X_train_object_{}.npy'.format(int(nt*2)), X_train)
    # reversed samples
    X_train = np.load(samples_training2[nt][0])
    X_train = hpfn.get_standardized_row(X_train)
    y_train = 0
    y_train_all.append(y_train)
    np.save(folder_processed + 'whole_frame_normalized_train/X_train_object_{}.npy'.format(int(nt*2+1)), X_train)
y_train_all = np.array(y_train_all)
np.save(folder_processed + 'y_train_all_object.npy', y_train_all)


####### Data normalization, testing #######
if not os.path.exists(folder_processed + 'whole_frame_normalized_test/'):
    os.makedirs(folder_processed + 'whole_frame_normalized_test/')
y_test_all = []
for nt in range(int(N_test/2)):
    # normal samples
    X_test = np.load(samples_testing1[nt][0])
    X_test = hpfn.get_standardized_row(X_test)
    y_test = 0
    y_test_all.append(y_test)
    np.save(folder_processed + 'whole_frame_normalized_test/X_test_object_{}.npy'.format(int(nt*2)), X_test)
    # reversed samples
    X_test = np.load(samples_testing2[nt][0])
    X_test = hpfn.get_standardized_row(X_test)
    y_test = 0
    y_test_all.append(y_test)
    np.save(folder_processed + 'whole_frame_normalized_test/X_test_object_{}.npy'.format(int(nt*2+1)), X_test)
y_test_all = np.array(y_test_all)
np.save(folder_processed + 'y_test_all_object.npy', y_test_all)

print(f'Total time took is {time.time()-start_time}.')
        
        
        
        
        
        
        
        
        
        