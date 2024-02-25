'''
Created by @mcanela on Tuesday, 13/02/2024
'''

# Specify data and load previously trained logistic regressors
directory = '/run/user/1000/gvfs/smb-share:server=172.20.4.47,share=becell/Macro tests/List of images/More ROIs/'
ratio = 1.55 # (1.55 px/µm at 10X)

# import pickle as pk
# with open('//folder/becell/Macro tests/Marc countings script/tdtomato/log_reg.pkl', 'rb') as file:
#     tdt = pk.load(file)
with open(directory + 'train_dict.pkl', 'rb') as file:
    dict_of_binary = pk.load(file)

# dict_of_layers = {
#     # 'layer_1': tdt,
#     'layer_2': cfos,
#     }

list_of_layers = ['layer_1']

# Run the function to calculate cells in each layer
from master_script import analyze_layers
my_binaries = analyze_layers(directory, list_of_layers, ratio)










# =========

binary_channel1 = [elem for elem in my_binaries.values()][0]
binary_channel2  = [elem for elem in my_binaries.values()][1]


















