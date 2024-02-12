"""
Created on Wed Dec 13 15:24:22 2023
@author: MCG
"""
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import math
import pingouin as pg

directory = '//folder/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2023-10 - TRAP2/Microscope TRAP2/Females/counts_python/'
df = pd.DataFrame()
for file in os.listdir(directory):
    if file.endswith('.csv'):
        file_path = os.path.join(directory, file)
        df2 = pd.read_csv(file_path)
        df = pd.concat([df, df2], ignore_index=True)

df['animal_id'] = df.file_name.str.split('_').str[1].astype(int)
df['brain_area'] = df.file_name.str.split('_').str[2]

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

def convert_pvalue_to_asterisks(pvalue):
    ns = "ns (p=" + str(pvalue)[1:4] + ")"
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return ns

def boxplot(df, brain_area='rsc', color_contrast=blue, ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(3.5,4))
    
    ax.set_title(brain_area, loc='left', color='#636466')
    sns.set_theme(style="whitegrid")
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    
    df = df[df.brain_area == brain_area]
    
    # No-shock data
    data1_position = 0
    data1 = df[df.animal_id < 4]
    data1, outliers1 = remove_outliers(data1, 'cells_per_squared_mm')
    print('Outliers found: ' + outliers1)
    
    # data1 = data1['cells_per_squared_mm'].tolist()
    # data1 = [x for x in data1 if not math.isnan(x)]
    data1 = data1.groupby('animal_id')['cells_per_squared_mm'].mean().tolist()
    
    data1_mean = np.mean(data1)
    data1_error = np.std(data1, ddof=1)
    
    # Shock data
    data2_position = 1
    data2 = df[df.animal_id > 3]
    data2, outliers2 = remove_outliers(data2, 'cells_per_squared_mm')
    print('Outliers found: ' + outliers2)

    # data2 = data2['cells_per_squared_mm'].tolist()
    # data2 = [x for x in data2 if not math.isnan(x)]
    data2 = data2.groupby('animal_id')['cells_per_squared_mm'].mean().tolist()
    
    data2_mean = np.mean(data2)
    data2_error = np.std(data2, ddof=1)

    ax.hlines(data1_mean, xmin=data1_position-0.25, xmax=data1_position+0.25, color='#636466', linewidth=1.5)
    ax.hlines(data2_mean, xmin=data2_position-0.25, xmax=data2_position+0.25, color='#636466', linewidth=1.5)
    
    ax.errorbar(data1_position, data1_mean, yerr=data1_error, lolims=False, capsize = 3, ls='None', color='#636466', zorder=-1)
    ax.errorbar(data2_position, data2_mean, yerr=data2_error, lolims=False, capsize = 3, ls='None', color='#636466', zorder=-1)

    ax.set_xticks([data1_position, data2_position])
    ax.set_xticklabels(['No-shock', 'Shock'])
    
    jitter = 0.15 # Dots dispersion
    
    dispersion_values_data1 = np.random.normal(loc=data1_position, scale=jitter, size=len(data1)).tolist()
    ax.plot(dispersion_values_data1, data1,
            'o',                            
            markerfacecolor=color_contrast,    
            markeredgecolor=color_contrast,
            markeredgewidth=1,
            markersize=5, 
            label='Data1')      
    
    dispersion_values_data2 = np.random.normal(loc=data2_position, scale=jitter, size=len(data2)).tolist()
    ax.plot(dispersion_values_data2, data2,
            'o',                          
            markerfacecolor=color_contrast,    
            markeredgecolor=color_contrast,
            markeredgewidth=1,
            markersize=5, 
            label='Data2')               
    
    if len(data1) == len(data2):
        for x in range(len(data1)):
            ax.plot([dispersion_values_data1[x], dispersion_values_data2[x]], [data1[x], data2[x]], color = '#636466', linestyle='--', linewidth=0.5)
    
    ax.set_ylim(0,350)
    
    ax.set_xlabel('')
    ax.set_ylabel('cFos+/mm^2', loc='top')    
    
    # Grey color
    ax.xaxis.label.set_color('#636466')
    ax.yaxis.label.set_color('#636466')
    ax.tick_params(axis='x', colors='#636466')
    ax.tick_params(axis='y', colors='#636466')
    
    # pvalue = pg.ttest(data1, data2, paired=True)['p-val'][0]

    # y, h, col = max(max(data1), max(data2)) + 5, 2, 'grey'
    
    # ax.plot([data1_position, data1_position, data2_position, data2_position], [y, y+h, y+h, y], lw=1.5, c=col)
    
    # if pvalue > 0.05:
    #     ax.text((data1_position+data2_position)*.5, y+2*h, convert_pvalue_to_asterisks(pvalue), ha='center', va='bottom', color=col, size=11)
    # elif pvalue <= 0.05:    
    #     ax.text((data1_position+data2_position)*.5, y, convert_pvalue_to_asterisks(pvalue), ha='center', va='bottom', color=col, size=18)
    
    plt.tight_layout()
    return ax

boxplot(df)

