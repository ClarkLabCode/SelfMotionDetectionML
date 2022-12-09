#!/usr/bin/env python

'''
This script selects natural scene images. 
The matlab file mat_file_resave.m should be run before this script.
'''


import sys
sys.path.append('../helper/')

import os
import numpy as np
import glob
import scipy.io as sio

import helper_functions as hpfn


# folder_main = '/mnt/d/data/' # local
folder_main = '/home/bz242/project/data/' # cluster

# Scenes that are 'natural' by Omer Mano
# goodScenes = [29,31,32,35:55,62,63,72:88,91,93,\
#               101,103,104,106,108,110:118,121,124:128,131,132,138,141:154,157,160,168,173,175,177:198,\
#               202:218,220:230,234,235,241:251,261:268,275:297,\
#               306,309:321,327:335,347:352,354,362:367,375:377,379:387,389,391,392,394,404,407:411,415,418]

# Restructure the goodScenes for python
goodScenes = [29, 31, 32, list(range(35, 55+1)), 62, 63, list(range(72, 88+1)), 91, 93, \
              101, 103, 104, 106, 108, list(range(110, 118+1)), 121, list(range(124, 128+1)), \
              131, 132, 138, list(range(141, 154+1)), 157, 160, 168, 173, 175, list(range(177, 198+1)),\
              list(range(202, 218+1)), list(range(220, 230+1)), 234, 235, list(range(241, 251+1)), \
              list(range(261, 268+1)), list(range(275, 297+1)), \
              306, list(range(309, 321+1)), list(range(327, 335+1)), list(range(347, 352+1)), 354, \
              list(range(362, 367+1)), list(range(375, 377+1)), list(range(379, 387+1)), 389, 391, 392, \
              394, 404, list(range(407, 411+1)), 415, 418]
goodScenes_new = []
for item in goodScenes:
    if isinstance(item, int):
        goodScenes_new.append(item)
    else:
        goodScenes_new.extend(item)
np.save(folder_main + 'panoramic/natural_scenes_numbers', goodScenes_new)
print(f'There are {len(goodScenes_new)} natural scene images.')

# Select natural scene images and save them in a different folder, also save a spacially filtered version if another folder
folder_nat = folder_main + 'panoramic/data_natural_only/'
if not os.path.exists(folder_nat):
    os.makedirs(folder_nat)
folder_nat_filtered = folder_main + 'panoramic/data_natural_only_filtered/'
if not os.path.exists(folder_nat_filtered):
    os.makedirs(folder_nat_filtered)


folder = folder_main + 'panoramic/data_001-421_v7/'
items = glob.glob(folder+'*.mat')
for item in items:
    if int(item[-22:-18]) in goodScenes_new:
        mat_contents = sio.loadmat(item)
        img = mat_contents['projection']
        np.save(folder_nat + f'natural_{int(item[-22:-18])}', img)

        K_row = img.shape[0] 
        K_col = img.shape[1] 
        pix_per_deg = K_col / 360
        FWHM = 5 # in degree
        sigma_for_gaussian = np.round(FWHM/(2*np.sqrt(2*np.log(2)))*pix_per_deg, 1) # smoothing gaussian
        pad_size = np.int(4*sigma_for_gaussian) # this comes from the fact that the gaussian is truncated at 4*std
        img_filtered = hpfn.get_filtered_spacial(img, pad_size, sigma_for_gaussian)
        np.save(folder_nat_filtered + f'natural_filtered_{int(item[-22:-18])}', img_filtered)








        