"""
Created on Mon May 13 14:03:34 2024
@author: mcanela
"""
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import networkx as nx

blue = '#194680'
red = '#801946'
grey = '#636466'
groups = ['paired', 'unpaired', 'noshock']

directory_csv = '//folder/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2024-02a04_TRAP2females/microscopi/'
values_column = 'engrams_mm'
# values_column = 'cells_per_squared_mm'

# Import countings
df = pd.DataFrame()
for file in os.listdir(directory_csv):
    if file.endswith('.csv'):
        file_path = os.path.join(directory_csv, file)
        df2 = pd.read_csv(file_path)
        df2['brain_area'] = file.split('_')[-1].split('.')[0]
        df = pd.concat([df, df2], ignore_index=True)
        
df['animal'] = df.file_name.str.split('_').str[1].astype(int)
df['group'] = df.file_name.str.split('_').str[0]

# Calculate the means dataframe
means_df = pd.DataFrame()
for brain_area in list(set(df.brain_area)):
    df2 = df[df.brain_area == brain_area]
    for group in groups:
        data = df2[df2.group == group]
        means = pd.DataFrame(data.groupby('animal')[values_column].mean())
        means['animal'] = means.index
        means['group'] = group
        means['brain_area'] = brain_area
        means_df = pd.concat([means_df, means], ignore_index=True)


# Define a function to compute correlation
def compute_correlation(means_df, my_group):
    # Get all unique pairs of brain areas within the group
    unique_areas = means_df['brain_area'].unique()
    pairs = [(area1, area2) for i, area1 in enumerate(unique_areas) 
             for j, area2 in enumerate(unique_areas) if i < j]
    
    # Initialize correlation matrix
    num_areas = len(unique_areas)
    correlation_matrix = np.zeros((num_areas, num_areas))
    
    # Compute correlation for each pair of brain areas
    for i, area1 in enumerate(unique_areas):
        for j, area2 in enumerate(unique_areas):
            if i < j:
                data_area1 = means_df[(means_df.brain_area == area1) & (means_df.group == my_group)]
                data_area2 = means_df[(means_df.brain_area == area2) & (means_df.group == my_group)]

                # Find animals present in both brain areas
                common_animals = set(data_area1['animal']).intersection(set(data_area2['animal']))

                # Filter data for common animals
                data_area1_common = data_area1[data_area1['animal'].isin(common_animals)][values_column]
                data_area2_common = data_area2[data_area2['animal'].isin(common_animals)][values_column]
                
                # Compute correlation if there are common animals
                if len(common_animals) > 0:
                    correlation, p_value = pearsonr(data_area1_common, data_area2_common)
                    correlation_matrix[i, j] = correlation
                    correlation_matrix[j, i] = correlation  # Because correlation_matrix is symmetric
                    
                    # # Check if correlation is significant
                    # if p_value <= 0.05:
                    #     correlation_matrix[i, j] = correlation
                    #     correlation_matrix[j, i] = correlation  # Because correlation_matrix is symmetric
                    # else:
                    #     correlation_matrix[i, j] = 0
                    #     correlation_matrix[j, i] = 0 # Because correlation_matrix is symmetric
    
    return correlation_matrix, unique_areas


def plot_transition_matrix(learning='paired', ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    transition_matrix, unique_states = compute_correlation(means_df, learning)
    
    # Create a DataFrame for better labeling of the heatmap
    sns.heatmap(transition_matrix, cmap="seismic_r", xticklabels=unique_states, yticklabels=unique_states,
                center=0) # Center the cmap at zero

    # Plot the heatmap
    sns.set_theme(style="whitegrid")

    # Set labels and title
    plt.title("Correlation Matrix between Brain Areas", loc = 'left', color='#636466')
    plt.xlabel("", loc='left')
    plt.ylabel("", loc='top')
    
    # Grey color
    ax.xaxis.label.set_color('#636466')
    ax.yaxis.label.set_color('#636466')
    ax.tick_params(axis='x', colors='#636466')
    ax.tick_params(axis='y', colors='#636466')

    return ax


def plot_circular_transition_graph(learning='noshock', ax=None):
    
    transition_matrix, unique_states = compute_correlation(means_df, learning)
    
    G = nx.DiGraph()
    G.add_nodes_from(range(len(unique_states)))
    
    for i in range(len(unique_states)):
        for j in range(len(unique_states)):
            if i != j:  # Avoid self-transitions
                G.add_edge(i, j, weight=transition_matrix[i][j])
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_title("Correlation between brain areas", color='#636466')

    # Get edge weights
    edge_weights = nx.get_edge_attributes(G, 'weight')
    
    # pos = nx.spring_layout(G)
    pos = nx.circular_layout(G)
    
    # Center colormap at zero
    vmin = min(edge_weights.values())
    vmax = max(edge_weights.values())
    abs_max = max(abs(vmin), abs(vmax))
    norm = plt.Normalize(vmin=-abs_max, vmax=abs_max)
    
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='snow', edgecolors='grey',
            edge_color=[edge_weights[edge] for edge in G.edges()],
            width=[abs(edge_weights[edge]) * 10 for edge in G.edges()],
            edge_cmap=plt.cm.seismic_r, edge_vmin=-abs_max, edge_vmax=abs_max, 
            arrows=False,
            labels={i: state.upper() for i, state in enumerate(unique_states)}, 
            font_size=8, font_color='#636466', font_weight='bold')
    
    return ax


def plot_spring_transition_graph(learning='noshock', ax=None):
    
    transition_matrix, unique_states = compute_correlation(means_df, learning)
    
    G = nx.DiGraph()
    G.add_nodes_from(range(len(unique_states)))
    
    # Threshold the edges
    for i in range(len(unique_states)):
        for j in range(len(unique_states)):
            if i != j:  # Avoid self-transitions
                if transition_matrix[i][j] >= 0.6:
                    G.add_edge(i, j, weight=transition_matrix[i][j])
                # if transition_matrix[i][j] < 0.6:
                #     G.add_edge(i, j, weight=0) 

    if ax is None:
        fig, ax = plt.subplots(figsize=(8,8))

    # ax.set_title("Correlation between brain areas", color='#636466')
    
    pos = nx.spring_layout(G, k=0.8, iterations=50)  # Adjusting the spring layout
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=2000, node_color='snow', edgecolors='grey')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, width=2, edge_color='grey', arrows=False)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, ax=ax, labels={i: state.upper() for i, state in enumerate(unique_states)}, 
                            font_size=10, font_color='#636466', font_weight='bold')

    # Set axis limits with padding
    padding = 0.1
    x_values = [coord[0] for coord in pos.values()]
    y_values = [coord[1] for coord in pos.values()]
    ax.set_xlim(min(x_values) - padding, max(x_values) + padding)
    ax.set_ylim(min(y_values) - padding, max(y_values) + padding)

    plt.axis('off')  # Turn off axis
    
    return ax


def structural_measures(means_df, groups):
    
    all_groups_list = {}
    for group in groups:
        transition_matrix, unique_states = compute_correlation(means_df, group)
        
        G = nx.DiGraph()
        G.add_nodes_from(range(len(unique_states)))
        for i in range(len(unique_states)):
            for j in range(len(unique_states)):
                if i != j and transition_matrix[i][j] > 0:  # Avoid self-transitions
                    G.add_edge(i, j, weight=transition_matrix[i][j])
        
        results_dict = {}
    
        # Calculate and return node and edge counts
        node_count = G.number_of_nodes()
        results_dict["Number of Areas"] = node_count
        edge_count = G.number_of_edges()
        results_dict["Number of Connections"] = edge_count
       
        # # Calculate the shortest path between all pairs of nodes
        unique_path_lengths = []
        
        for source in G.nodes():
            for target in G.nodes():
                if source != target:
                    try:
                        length = nx.shortest_path_length(G, source=source, target=target)
                        unique_path_lengths.append(length)
                    except nx.NetworkXNoPath:
                        pass
        
        results_dict["Average Path Length"] = unique_path_lengths

        # Calculate the degree distribution
        degree_sequence = [G.out_degree(node) for node in G.nodes()]
        results_dict["Degree Distribution"] = degree_sequence
    
        # Calculate the clustering coefficient for individual nodes
        node_clustering = nx.clustering(G)
        node_clustering_list = [value for value in node_clustering.values()]
        results_dict["Clustering Coefficient"] = node_clustering_list
        # Calculate the clustering coefficient for the entire graph (it's the mean of the above list)
        # average_clustering = nx.average_clustering(G, count_zeros=True)  # count_zeros=True includes nodes with zero clustering coefficient
    
        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(G)
        degree_centrality_list = [value for value in degree_centrality.values()]
        results_dict["Degree Centrality"] = degree_centrality_list
        
        betweenness_centrality = nx.betweenness_centrality(G)
        betweenness_centrality_list = [value for value in betweenness_centrality.values()]
        results_dict["Betweenness Centrality"] = betweenness_centrality_list
        
        eigenvector_centrality = nx.eigenvector_centrality(G)
        eigenvector_centrality_list = [value for value in eigenvector_centrality.values()]
        results_dict["Eigenvector Centrality"] = eigenvector_centrality_list
        
        all_groups_list[group] = results_dict
        
    return all_groups_list


def boxplot(df, groups, measure="Eigenvector Centrality", ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(3.5,4))
    
    ax.set_title('', loc='left', color='#636466')
    sns.set_theme(style="whitegrid")
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    jitter = 0.15 # Dots dispersion
    
    # Grey color
    ax.xaxis.label.set_color('#636466')
    ax.yaxis.label.set_color('#636466')
    ax.tick_params(axis='x', colors='#636466')
    ax.tick_params(axis='y', colors='#636466')
    
    all_groups_list = structural_measures(means_df, groups)
    
    groups = ['paired', 'unpaired', 'noshock']
    positions = []
    
    # Calculate cFos
    for group in groups:
        data_position = groups.index(group)
        positions.append(data_position)
        data = all_groups_list[group][measure]
        
        data_mean = np.mean(data)
        data_error = np.std(data, ddof=1)
        
        ax.hlines(data_mean, xmin=data_position-0.25, xmax=data_position+0.25, color='#636466', linewidth=1.5)
        ax.errorbar(data_position, data_mean, yerr=data_error, lolims=False, capsize = 3, ls='None', color='#636466', zorder=-1)
        
        dispersion_values_data = np.random.normal(loc=data_position, scale=jitter, size=len(data)).tolist()
        
        for x, y in zip(dispersion_values_data, data):
            ax.plot(x, y,
                    'o',                            
                    markerfacecolor=blue,    
                    markeredgecolor=grey,
                    markeredgewidth=1,
                    markersize=5, 
                    label=group
                    )

    ax.set_xticks(positions)
    ax.set_xticklabels(groups)
    ax.set_xlabel('')
    ax.set_ylabel(measure, loc='top')

    # if len(data1) == len(data2):
    #     for x in range(len(data1)):
    #         ax.plot([dispersion_values_data1[x], dispersion_values_data2[x]], [data1[x], data2[x]], color = '#636466', linestyle='--', linewidth=0.5)
        
    plt.tight_layout()
    return ax









































