"""
Created on Tue Jan 23 14:11:20 2024
@author: mcanela
"""
import pandas as pd
import numpy as np
import os
import deepof.data
import seaborn as sns
import matplotlib.pyplot as plt
import math
import pingouin as pg
import pickle

directory = '//folder/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2023-10 - TRAP2/Microscope TRAP2/Females/counts_python/'
df = pd.DataFrame()
for file in os.listdir(directory):
    if file.endswith('.csv'):
        file_path = os.path.join(directory, file)
        df2 = pd.read_csv(file_path)
        df = pd.concat([df, df2], ignore_index=True)

df['animal_id'] = df.file_name.str.split('_').str[1].astype(int)
df['brain_area'] = df.file_name.str.split('_').str[2]

# Import DeepOF data
directory_deepof = "//folder/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2023-10 - TRAP2/Female/DeepOF/"
with open(directory_deepof + 'supervised_annotation.pkl', 'rb') as file:
    supervised_annotation = pickle.load(file)

deepof_dict = {}
for tag, deepof_df in supervised_annotation.items():
    # Create 6 bins using cut
    deepof_df = deepof_df.reset_index(drop=True)
    deepof_df['bins'] = pd.cut(deepof_df.index, bins=6, labels=False)
    # Select the rows in the 3rd bin (bin index starts from 0)
    off_data = deepof_df[deepof_df['bins'] == 2]
    off_data = off_data['huddle'].mean() * 100
    # Select the rows in the 4th bin (bin index starts from 0)
    on_data = deepof_df[deepof_df['bins'] == 3]
    on_data = on_data['huddle'].mean() * 100
    deepof_dict[tag] = [off_data, on_data]

blue = '#194680'
red = '#801946'
grey = '#636466'

def remove_outliers(df, column_name):
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (df[column_name] < lower_bound) | (df[column_name] > upper_bound)
    df_cleaned = df[~outliers]
    outlier_name = df[outliers]['file_name']
    return df_cleaned, outlier_name

# Write the equivalence between cFos tags and DeepOF tags
tags_dict = {
    '20231027_Marc_ERC TRAP2 S2_Females_box a_230030238_02_01_1': 1,
    '20231027_Marc_ERC TRAP2 S2_Females_box a_230030239_02_01_1': 2,
    '20231027_Marc_ERC TRAP2 S2_Females_box a_230030240_02_01_1': 3,
    '20231027_Marc_ERC TRAP2 S2_Females_box b_230031459_02_01_1': 4,
    '20231027_Marc_ERC TRAP2 S2_Females_box b_230031460_02_01_1': 5,
    '20231027_Marc_ERC TRAP2 S2_Females_box c_230030602_02_01_1': 6,
    '20231027_Marc_ERC TRAP2 S2_Females_box c_230030603_02_01_1': 7,
    '20231027_Marc_ERC TRAP2 S2_Females_box c_230030604_02_01_1': 8,
    '20231027_Marc_ERC TRAP2 S2_Females_box c_230030605_02_01_1': 9
    }

# Filter tag_dict
noshock = dict(list(tags_dict.items())[:3])
shock = dict(list(tags_dict.items())[3:])

# Create the dataframe to plot
def create_dict_to_plot(filtered_tag_dict, brain_area):
    dict_to_plot = {}
    for tag_deepof, tag_cfos in filtered_tag_dict.items():
        df_cfos = df[df.animal_id == tag_cfos]
        # Select the brain region to plot
        df_cfos = df_cfos[df_cfos.brain_area == brain_area]
        df_cfos, outliers = remove_outliers(df_cfos, 'cells_per_squared_mm')    
        final_count = df_cfos.cells_per_squared_mm.mean()
        dict_to_plot[final_count] = deepof_dict[tag_deepof]
    return dict_to_plot

dict_to_plot = create_dict_to_plot(shock, 'dg')

# Extracting keys and corresponding first list items
counts = list(dict_to_plot.keys())
off = [lst[0] for lst in dict_to_plot.values()]
on = [lst[1] for lst in dict_to_plot.values()]

# Creating a scatter plot
plt.scatter(counts, off)

# Adding labels and title
plt.xlabel('cFos+/mm^2 DG shock')
plt.ylabel('Huddle OFF')
plt.title('')

# Display the plot
plt.show()



















