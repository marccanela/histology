'''
Created by @mcanela on Tuesday, 13/02/2024
'''

# Specify data
directory = '//folder/becell/Macro tests/List of images/ROIs to analyze/'
ratio = 1.55 # px/µm

# Load previously trained logistic regressor
import pickle as pk
with open(directory + 'log_reg.pkl', 'rb') as file:
    log_reg = pk.load(file)

# Select your ROIs
from master_script import create_dict_of_binary
dict_of_binary = create_dict_of_binary(directory)

# Apply the logistic regressor
from master_script import compiler
compiler(directory, dict_of_binary, ratio, log_reg)