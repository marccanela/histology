import pickle as pkl

from master_script import analyze_images

# Enter the directory where you store your .ND2 images, your roi_dict.pkl, and the best_hyperparameters.pkl
directory = "//folder/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2024-02a04_TRAP2females/microscopi/bla/"

# Import your ROIs and hyperparameters
with open(directory + "dict_rois_cg1.pkl", "rb") as file:
    dict_rois = pkl.load(file)

# hyperparameters_cfos = {
#     'distance_2': 7.313,
#     'distance_3': 4.01,
#     'distance_4': 4.085
#     }

directory_hyper = "//folder/becell/Macro tests/"
directory_hyper = "//folder/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2024-02a03 - TRAP2 females batch 1/Microscopi/"
with open(directory_hyper + "best_hyperparameters_mae.pkl", "rb") as file:
    best_hyperparameters_cfos = pkl.load(file)

# Create a dictionary indicating which hyperparameters correspond to each layer
layer_dict = {
    "layer_1": best_hyperparameters_cfos,
    # 'layer_2': best_hyperparameters_tdt
}

# Run the function to analyze your images
my_binaries = analyze_images(dict_rois, directory, layer_dict, 1.55)
