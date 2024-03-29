
import pickle as pkl
from master_script import analyze_images

# Enter the directory where you store your .ND2 images, your roi_dict.pkl, and the best_hyperparameters.pkl
directory = '//folder/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2024-01a02 - cFos/microscope/hippocampus_mae/'

# Import your ROIs and hyperparameters
with open(directory + 'rois_dg.pkl', 'rb') as file:
    dict_rois = pkl.load(file)
with open(directory + 'hyperparameters_cfos.pkl', 'rb') as file:
    hyperparameters_cfos = pkl.load(file)
    
# Create a dictionary indicating which hyperparameters correspond to each layer
layer_dict = {
    'layer_1': hyperparameters_cfos
}

# Run the function to analyze your images
analyze_images(dict_rois, directory, layer_dict, ratio=1.55)