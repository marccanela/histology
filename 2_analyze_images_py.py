
import pickle as pkl
from master_script import analyze_images

# Enter the directory where you store your .ND2 images, your roi_dict.pkl, and the best_hyperparameters.pkl
directory = '//folder/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2024-01a02 - cFos/microscope/pagm/results/'

# Import your ROIs and hyperparameters
with open(directory + 'rois_pagm.pkl', 'rb') as file:
    dict_rois = pkl.load(file)

# hyperparameters_cfos = {
#     'distance_2': 7.313, 
#     'distance_3': 4.01,
#     'distance_4': 4.085
#     }

directory = '//folder/becell/Macro tests/'
with open(directory + 'best_hyperparameters.pkl', 'rb') as file:
    best_hyperparameters = pkl.load(file)
    
# Create a dictionary indicating which hyperparameters correspond to each layer
layer_dict = {
    'layer_1': best_hyperparameters
}

# Run the function to analyze your images
analyze_images(dict_rois, directory, layer_dict, ratio=1.55)
