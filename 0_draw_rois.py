"""
DRAW ROIs
@author: mcanela
"""

import pickle as pkl

# Define your ROIs and save them in your computer
directory = '//folder/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2024-02a04_TRAP2females/microscopi/bla/'
from master_script import create_dict_rois
dict_rois = create_dict_rois(directory)
with open(directory + 'dict_rois.pkl', 'wb') as file:
    pkl.dump(dict_rois, file)

# Alternatively, you can open already saved ROIs and use them
with open(directory + 'dict_rois.pkl', 'rb') as file:
    dict_rois = pkl.load(file)


