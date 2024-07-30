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
import pickle as pkl
import statsmodels.api as sm
from statsmodels.formula.api import ols

dir_countings = '//folder/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2024-02a04_TRAP2females/microscopi_batch3/'
dir_supervised = '//folder/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2024-02a04_TRAP2females/behavior/combined/'
dir_correspondence = '//folder/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2024-02a04_TRAP2females/'

correspondence_tag = 'correspondence_tone.csv'
brain_area = 'vlpag'
values_column = 'engrams_mm'
# values_column = 'cells_per_squared_mm'
# values_column = 'ratio'


# Import countings
df = pd.DataFrame()
for file in os.listdir(dir_countings):
    if file.endswith('.csv'):
        file_path = os.path.join(dir_countings, file)
        df2 = pd.read_csv(file_path)
        df = pd.concat([df, df2], ignore_index=True)

# Import supervised
with open(dir_supervised + 'supervised_annotation_abc.pkl', 'rb') as file:
    supervised_annotation = pkl.load(file)

# Import conditions
for file in os.listdir(dir_correspondence):
    if file.endswith(correspondence_tag):
        file_path = os.path.join(dir_correspondence, file)
        correspondence = pd.read_csv(file_path)

df['animal'] = df.file_name.str.split('_').str[1].astype(int)
df['brain_area'] = df.file_name.str.split('_').str[2]
df['group'] = df.file_name.str.split('_').str[0]

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

def boxplot(df, brain_area, behavior='huddle', ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(3.5,4))
    
    ax.set_title(brain_area, loc='left', color='#636466')
    sns.set_theme(style="whitegrid")
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    jitter = 0.15 # Dots dispersion
    
    # Grey color
    ax.xaxis.label.set_color('#636466')
    ax.yaxis.label.set_color('#636466')
    ax.tick_params(axis='x', colors='#636466')
    ax.tick_params(axis='y', colors='#636466')
    
    df2 = df[df.brain_area == brain_area]
    
    groups = ['paired', 'unpaired', 'noshock']
    positions = []
    
    # To calculate behavior
    num_of_bins = {}
    for key, value in supervised_annotation.items():
        factor = int(360/60)
        
        value.reset_index(inplace=True)
        value.drop('index', axis=1, inplace=True)
        value.reset_index(inplace=True)
                
        bin_length = int(len(value) // factor)
        cutoffs = [i * bin_length for i in range(1, factor)]
        
        # Determine the minimum and maximum of the 'index' column
        min_value = value['index'].min()
        max_value = value['index'].max() + 1
    
        # Add the leftmost and rightmost edges for the first and last bins
        cutoffs = [min_value] + cutoffs + [max_value]
        
        value['bin'] = pd.cut(value['index'], bins=cutoffs, labels=False, right=False, include_lowest=True)
        
        num_of_bins[key] = value
       
    behavior_df = pd.concat(num_of_bins.values(), keys=num_of_bins.keys()).reset_index()
    
    mean_values = behavior_df.groupby(['bin', 'level_0'])[behavior].mean()
    mean_values = mean_values.reset_index()
    mean_values['bin'] = mean_values['bin'] + 1
    mean_values_3 = mean_values[mean_values.bin == 3]
    mean_values_3.rename(columns={behavior: 'behavior_3'}, inplace=True)
    
    mean_values_4 = mean_values[mean_values.bin == 4]
    mean_values_4.rename(columns={behavior: 'behavior_4'}, inplace=True)

    mean_values = pd.merge(mean_values_3, mean_values_4, on='level_0')
    mean_values['di'] = (mean_values.behavior_4 - mean_values.behavior_3) / (mean_values.behavior_4 + mean_values.behavior_3)
    
    # Calculate cFos
    dataframes_for_stats = []
    for group in groups:
        data_position = groups.index(group)
        positions.append(data_position)
        data = df2[df2['group'] == group]
        # data, outliers = remove_outliers(data, values_column)
        # print('Outliers found: ' + outliers)
        
        # Combine behavior and cFos
        data = pd.DataFrame(data.groupby('animal')[values_column].mean())
        
        # data = pd.DataFrame(data.groupby('animal')[['engrams_mm', 'cells_per_squared_mm']].mean())
        # data[values_column] = data['engrams_mm'] / data['cells_per_squared_mm']
        
        data.reset_index(inplace=True)
        data = pd.merge(data, correspondence, on='animal', how='left')
        combined_df = pd.merge(data, mean_values, left_on='video', right_on='level_0', how='inner')
        
        # Data for stats
        data_stats = combined_df[['animal', values_column]]
        data_stats['group'] = group
        dataframes_for_stats.append(data_stats)
        
        # Set colors
        bins = [-1, -0.25, 0, 0.25, 1]
        colors = ['#cb181d', '#fc9272', '#9ecae1', '#2171b5']
        combined_df['colors'] = pd.cut(combined_df.di, bins, labels=colors, include_lowest=True)
        
        data_mean = np.mean(combined_df[values_column])
        data_error = np.std(combined_df[values_column], ddof=1)
        
        ax.hlines(data_mean, xmin=data_position-0.25, xmax=data_position+0.25, color='#636466', linewidth=1.5)
        ax.errorbar(data_position, data_mean, yerr=data_error, lolims=False, capsize = 3, ls='None', color='#636466', zorder=-1)
        
        dispersion_values_data = np.random.normal(loc=data_position, scale=jitter, size=len(combined_df)).tolist()
        plot_dict = dict(zip(combined_df.animal, zip(combined_df[values_column], dispersion_values_data, combined_df.colors)))
        
        for value in plot_dict.values():
            ax.plot(value[1], value[0],
                    'o',                            
                    markerfacecolor=value[2],    
                    markeredgecolor=grey,
                    markeredgewidth=1,
                    markersize=5, 
                    label=group
                    )

    ax.set_xticks(positions)
    ax.set_xticklabels(groups)
    ax.set_xlabel('')
    ax.set_ylabel('+cells/mm^2', loc='top')

    # if len(data1) == len(data2):
    #     for x in range(len(data1)):
    #         ax.plot([dispersion_values_data1[x], dispersion_values_data2[x]], [data1[x], data2[x]], color = '#636466', linestyle='--', linewidth=0.5)
    
    # max_value = max(countings)
    # min_value = min(countings)
    # sd = np.std(combined_df.cells_per_squared_mm)
    # ax.set_ylim(min_value - 2*sd, max_value + 2*sd)
    
    stats_df = pd.concat(dataframes_for_stats, ignore_index=True)
        
    plt.tight_layout()
    return ax, stats_df

ax, stats_df = boxplot(df, brain_area)
# stats_df['batch'] = 'b'
# stats_df.loc[stats_df['animal'] > 100, 'batch'] = 'a'

# Test normality
# pg.normality(stats_df, dv=values_column, group='group')
# pg.normality(stats_df, dv=values_column, group='batch')

# Test equal variances
# pg.homoscedasticity(stats_df, dv=values_column, group='group')
# pg.homoscedasticity(stats_df, dv=values_column, group='batch')

# ttest
# pg.ttest(stats_df[stats_df.group == 'paired'][values_column],
#          stats_df[stats_df.group == 'unpaired'][values_column])['p-val']

# RCBD
# model = ols(values_column + ' ~ C(group) + C(batch)', data=stats_df).fit()
# sm.stats.anova_lm(model, typ=2)

# Perform ANOVA
# pg.anova(stats_df, dv=values_column, between=['group', 'batch'])
# pg.anova(stats_df, dv=values_column, between='group')

# Perform post-hoc
# pg.pairwise_tukey(stats_df, dv=values_column, between='group')

# Non-parametric
# pg.kruskal(stats_df, dv=values_column, between='group')

































