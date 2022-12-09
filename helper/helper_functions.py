#!/usr/bin/env python

"""
This script contains helper functions for the data/model of the motion detection.
"""


import numpy as np
import glob
import matplotlib.image as mpimg
import cv2
from scipy.ndimage import gaussian_filter
import random

import torch
from torch.utils.data import Dataset


####################
####### Data #######
####################

# Define the dataset class.
class RigidMotionDataset_ln(Dataset):
    """Rigid Motion dataset."""

    def __init__(self, targets, samples, transform=None, T=30):
        """
        Args:
            targets: numpy array
            samples: numpy array
            transform (callable, optional): Optional transform to be applied
                on a sample.
            T: the length in time (one for 10 ms)
        """
        self.targets = targets
        self.samples = samples
        self.transform = transform
        self.T = T

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        movie = self.samples[idx][-self.T:] # take the last T frames
        target = self.targets[idx]
        sample = {'movie': movie, 'target': target}

        if self.transform:
            sample = self.transform(sample)

        return sample

    
# Transform functions
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        movie, target = sample['movie'], sample['target']

        return {'movie': torch.from_numpy(movie).float(),
                'target': torch.from_numpy(target).long()}


################################
####### helper functions #######
################################

def get_standardized_row(input_array):
    """
    Standardize an array.
    """
    mu = np.expand_dims(np.mean(input_array, axis=-1), axis=-1)
    std = np.expand_dims(np.std(input_array, axis=-1), axis=-1)

    assert std.all()
    output_array = (input_array - mu) / std
    
    return output_array


def get_normalized_row(input_array):
    """
    Normalize an array to be within [-1, 1]
    """
    min_v = np.expand_dims(np.min(input_array, axis=-1), axis=-1)
    max_v = np.expand_dims(np.max(input_array, axis=-1), axis=-1)

    assert (max_v - min_v).all()
    output_array = (input_array - min_v) / (max_v - min_v)
    output_array = (output_array - 0.5) * 2
    
    return output_array 


def get_standardized(input_array):
    """
    Standardize an array.
    """
    mu = np.mean(input_array)
    std = np.std(input_array)

    assert std > 0
    output_array = (input_array - mu) / std
    
    return output_array


def get_normalized(input_array, lower=-1, upper=1):
    """
    Normalize an array to be within [-1, 1]
    """
    min_v = np.min(input_array)
    max_v = np.max(input_array)

    assert max_v > min_v
    output_array = (input_array - min_v) / (max_v - min_v)
    output_array = output_array * (upper - lower) + lower
    
    return output_array   


def get_filtered_spacial(input_frame, pad_size, sigma_for_gaussian):
    """
    Filter the input_frame with a Gaussian filter spacially.
    """
    K_row = input_frame.shape[0]
    K_col = input_frame.shape[1]
    padded_frame = np.zeros((K_row+2*pad_size, K_col+2*pad_size))
    padded_frame[pad_size:-pad_size, pad_size:-pad_size] = input_frame
    padded_frame[pad_size:-pad_size, :pad_size] = input_frame[:, -pad_size:]
    padded_frame[pad_size:-pad_size, -pad_size:] = input_frame[:, :pad_size]
    filtered_frame = gaussian_filter(padded_frame, sigma_for_gaussian)
    output_frame = filtered_frame[pad_size:-pad_size, pad_size:-pad_size]
    
    return output_frame


def get_filtered_spacial_row(input_frame, sigma_for_gaussian):
    """
    Filter the input_frame with a Gaussian filter spacially.
    """
    K_row = input_frame.shape[0]
    K_col = input_frame.shape[1]
    output_frame = np.zeros((K_row, K_col))
    for kk in range(K_row):
        output_frame[kk, :] = gaussian_filter(input_frame[kk, :], sigma_for_gaussian, mode='wrap')
    
    return output_frame


def get_resized(input_array, n_row=20, n_col=72):
    """
    Resize the input_array.
    """
    step_row = input_array.shape[0] / n_row
    step_col = input_array.shape[1] / n_col

    row_sample = (np.arange(n_row) * step_row + step_row / 2).astype(int)
    col_sample = (np.arange(n_col) * step_col + step_col / 2).astype(int)
    resized_array = input_array[row_sample, :]
    resized_array = resized_array[:, col_sample]

    return resized_array


def get_resized_cv2(input_array, new_size=(72, 20)):
    """
    Resize the input_array using cv2.
    """
    resized_array = cv2.resize(input_array, dsize=new_size, interpolation=cv2.INTER_AREA)

    return resized_array


def get_filtered_OU_1d(gamma1, delta_t, input_array, vel_length, initial_val=0):
    """
    This function filters a 1d input array in an Ornstein-Uhlenbeck fashion.
    This is to generate correlated a velocity trace.
    ________
    Args:
    gamma1 - time scale
    delta_t - time resolution
    input_array - input array of accelerations
    vel_length - choose the last vel_length elements to avoid correlations with 
                 the initial value, which is the length of the velocity trace.
    initial_val - initial value
    """
    output_array = [initial_val]
    e_factor = np.exp(-gamma1*delta_t)
    for ind, ele in enumerate(input_array):
        ele_out = e_factor * output_array[ind] + (1 - e_factor) * ele / gamma1
        output_array.append(ele_out)
    output_array = np.array(output_array[-vel_length:])
    
    return output_array


def get_shift_array(vel_array, delta_t, img):
    """
    Get the arrays that store the shift sizes at each time point.
    The shift sizes can be in degrees or in pixels.
    """
    pix_per_deg = img.shape[1] / 360
    shift_array_deg = [0]
    shift_array_pix = [0]
    for ind, vel in enumerate(vel_array):
        shift = shift_array_deg[ind] + vel * delta_t
        shift_array_deg.append(shift)
        shift_array_pix.append(np.int(np.round(shift*pix_per_deg)))
        
    return shift_array_deg, shift_array_pix


def add_object_to_image_periodic_boundary(img_array, x, l, row, image_patch, deg_x=360, deg_y=97):
    """
    Add an object to an image with periodic boundary.
    _________
    Args:
    img_array - the image
    x - position of the left end of the object, in degree (leftmost (rightmost) edge is 0 (360).).
    l - length of the object, in degree
    row - which row in the image the object is at.
    image_patch - image patch as the object.
    deg_x - span in degree in the x dimension (horizontal)
    deg_y - span in degree in the y dimension (vertical)

    Returns:
    img_array - the image with objects on it
    mask_in_x - the image areas occupied by objects
    """
    L1 = img_array.shape[0]
    L2 = img_array.shape[1]
    xl = np.int(np.floor((x % deg_x) / deg_x * L2))
    xr = xl + np.int(np.floor(l / deg_x * L2))
    xr = xr % L2
    yu = row
    yd = row + np.int(np.floor(5 / deg_y * L1))
    mask_in_x = np.zeros(L2)
    if xl < xr:
        img_array[yu:yd, xl:xr] = image_patch[:, :]
        mask_in_x[xl:xr] = 1
    else:
        img_array[yu:yd, xl:] = image_patch[:, :(L2-xl)]
        img_array[yu:yd, :xr+1] = image_patch[:, -xr-1:]
        mask_in_x[xl:] = 1
        mask_in_x[:xr+1] = 1
        
    return img_array, mask_in_x


def get_1d_foreground(pattern, val=0, pix_per_deg=1, FWHM=5, N=36):
    """
    Args:
    pattern - foreground pattern, either 'uniform' or 'structured' or 'none' (no foreground)
    val - the value of the element in the uniform pattern
    pix_per_deg - number of pixels per degree
    FWHM - in degree
    N - number of moving objects
    
    Returns:
    foreground - the foreground mask
    """
    if pattern == 'uniform':
        foreground = val * np.ones((1, 72))
    elif pattern == 'structured':
        sigma_for_gaussian = np.round(FWHM/(2*np.sqrt(2*np.log(2)))*pix_per_deg, 1) # smoothing gaussian
        pad_size = int(4*sigma_for_gaussian) # this comes from the fact that the gaussian is truncated at 4*std
        Ls = 5 + 0 * np.random.random(N) # angular spans of the objects, in degree
        X0s = 360 * np.random.random(N) # initial positions of the objects, in degree (leftmost (rightmost) edge is 0 (360).).

        foreground = np.zeros((1, 360))
        image_patch = np.ones((1, 5)) # Get the image patches as the objects
        L2 = foreground.shape[1]
        row = 0
        for n in range(N):
            x = X0s[n]
            l = Ls[n]
            foreground, _, = add_object_to_image_periodic_boundary(foreground, x, l, row, image_patch, deg_x=360, deg_y=5)
        foreground = get_filtered_spacial(foreground, pad_size, sigma_for_gaussian)
        foreground = get_resized(foreground, n_row=1)
        foreground = get_standardized(foreground)
    elif pattern == 'none':
        foreground = None
        
    return foreground


def get_open_windows(K, W):
    """
    Args:
    K - number of windows
    W - width of each window
    
    Returns:
    open_windows - open windows
    """
    # open windows
    open_windows = np.zeros(72)
    for k in range(K):
        rnd_int = np.random.randint(0, 69)
        open_windows[rnd_int:rnd_int+W] = 1
        
    return open_windows
        
    
def get_movie_with_1d_foreground(img_with_objects_smoothed, delta_t, T, shift, open_windows, pattern, val=0, flicker=False, moving_window=False):

    img_with_objects_smoothed_copy = img_with_objects_smoothed.copy()

    L1, L2 = img_with_objects_smoothed.shape


    img_with_objects_smoothed_motion = np.zeros(((T, L1, L2)))
    img_with_objects_resized_motion = np.zeros((T, 1, 72))
    img_with_objects_resized_motion_with_foreground = np.zeros((T, 1, 72))

    foreground = get_1d_foreground(pattern, val)

    shift_t = 0
    for t in range(T):
        shift_t = shift_t + shift
        shift_in_px_per_step = int(shift_t / 360 * delta_t * L2)
        shift_in_px_per_step_window = shift_in_px_per_step
        img_with_objects_smoothed_motion[t, :, :] = np.roll(img_with_objects_smoothed, shift_in_px_per_step)
        img_with_objects_resized_motion[t, :, :] = get_resized(img_with_objects_smoothed_motion[t, :, :], n_row=1)
        img_with_objects_resized_motion[t, :, :] = get_standardized(img_with_objects_resized_motion[t, :, :])
        if pattern != 'none':
            if moving_window:
                open_windows_now = np.roll(open_windows, int(np.round(shift_in_px_per_step_window/5, 0)))
                shift_in_px_per_step_window = shift_in_px_per_step_window + shift_in_px_per_step
            else:
                open_windows_now = open_windows.copy()
            img_with_objects_resized_motion_with_foreground[t, :, :] = foreground.copy()
            img_with_objects_resized_motion_with_foreground[t, 0, :][open_windows_now==1] \
            = img_with_objects_resized_motion[t, 0, :][open_windows_now==1]

            if flicker and (t+1) % 6 == 0:
                foreground = get_1d_foreground(pattern, val)
        
    return foreground, \
           img_with_objects_resized_motion_with_foreground, \
           img_with_objects_resized_motion, \
           img_with_objects_smoothed_motion


def get_rank_order_corr(order1, order2):
    d_square = np.sum((order1 - order2) ** 2)
    n = len(order1)
    rho_corr = 1 - 6 * d_square / (n * (n ** 2 - 1))
    
    return rho_corr


####### For plotting and others #######
def get_hist(arr, pseudo_count=True):
    arr = arr.flatten()
    bins = np.int(np.sqrt(arr.shape[0]))
    bins = np.minimum(bins, 1000)
    hist, bin_edges = np.histogram(arr, bins, density=True)
    bin_centers = 0.5*(bin_edges[:-1]+bin_edges[1:])
    
    return hist, bin_centers


def get_sigmoid(a, b, x):
    y = 1. / (1. + np.exp(-a*x-b))
    
    return y


def convert_logit_to_probability(logits):
    logit = logits[:, 1] - logits[:, 0]
    prob = get_sigmoid(1, 0, logit)

    return prob, logit


def get_mean_upper_lower_bound(input_all, ax=0):
    mean_curve = input_all.mean(axis=ax)
    std_curve = input_all.std(axis=ax)
    upper_curve = mean_curve + std_curve
    lower_curve = mean_curve - std_curve
    
    return mean_curve, upper_curve, lower_curve


def get_probability(network, img_motion, offset=30):
    T = img_motion.shape[1]
    pred_prob = np.zeros((img_motion.shape[0], T-offset))
    pred_logit = np.zeros((img_motion.shape[0], T-offset))
    for t in range(T-offset):
        inputs = img_motion[:, t:t+offset]
        inputs = torch.from_numpy(inputs).float()
        outputs = network(inputs)
        outputs = outputs.detach().numpy()
        prob, logit = convert_logit_to_probability(outputs)
        pred_prob[:, t] = prob
        pred_logit[:, t] = logit
        
    return pred_prob, pred_logit



















