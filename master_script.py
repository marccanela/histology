"""
Created on Tue Jan  9 09:50:17 2024
@author: mcanela
"""

import nd2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
import cv2
from kneed import KneeLocator
from scipy.signal import find_peaks
from skimage.measure import regionprops
from scipy.ndimage import label
from scipy import ndimage as ndi
import os
from matplotlib.patches import Polygon
import random
import math
from tqdm import tqdm
import time
from skimage import morphology
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import copy

def split_layers(image):
    layers = {}
    for n in range(image.shape[0]):
        my_layer = image[n]
        layers['layer_' + str(n)] = my_layer
    return layers

def draw_ROI(layer_data, tag):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(layer_data, cmap='gray')
    ax.set_title(tag)
    print('Draw a ROI on this layer and click on the initial dot to finish.')
    print('To use the next layer to draw the ROI, just close this layer.')
    print('Close all layers to use the whole image.')

    polygon_coords = []  # List to store selected points
    def onselect(verts):
        polygon_coords.append(verts)
        print('The ROI has been correctly saved.')

    # Set the facecolor parameter to 'r' for red pointer
    polygon_selector = PolygonSelector(ax, onselect, props=dict(color='r', linestyle='-', linewidth=2, alpha=0.5))
    
    plt.show(block=True)
    
    if len(polygon_coords) == 0:
        return None
    else:
        return polygon_coords[-1]

def calculate_elbow(hist, bins):
    peaks, _ = find_peaks(hist)
    closest_peak_index = np.argmax(hist[peaks])
        
    # Create a subset of histogram and bins values between the identified peak and the end
    subset_hist = hist[peaks[closest_peak_index]:]
    subset_bins = bins[peaks[closest_peak_index]:-1]
        
    # Find the elbow using KneeLocator on the subset
    knee = KneeLocator(subset_bins, subset_hist, curve='convex', direction='decreasing')
    elbow_value = knee.elbow
    return elbow_value

def background_threshold(blurred_normalized):
    hist, bins = np.histogram(blurred_normalized[blurred_normalized != 0], bins=64, range=(1, 256)) 
    elbow_value = calculate_elbow(hist, bins)
        
    return elbow_value

def normalize_array(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_arr = (arr - min_val) / (max_val - min_val) * 255
    return normalized_arr.astype(int)

def image_to_binary(roi_data, layers, layer):
    
    # Apply the defined ROI
    cfos = layers[layer]  # Select layer_0, layer_1, layer_2, etc.
    mask = np.zeros_like(cfos)
    cv2.fillPoly(mask, roi_data, 255)
    layer_roi = np.where(mask == 255, cfos, 0)#.astype(int)
    # plt.imshow(layer_roi, cmap='grey');
    
    # Apply Gaussian blur to reduce noise
    layer_roi = cv2.GaussianBlur(layer_roi, (5, 5), 0)
    # plt.imshow(blurred, cmap='grey');
    
    # Normalize the cfos layer
    blurred_normalized = normalize_array(layer_roi)
    # plt.imshow(blurred_normalized, cmap='grey');
  
    # Apply a threshold for the background    
    elbow_value = background_threshold(blurred_normalized)
    binary_image = blurred_normalized < elbow_value
    # plt.imshow(binary_image, cmap='grey');
    
    # Plot identified cells
    # identified = cfos[:]
    # identified[binary_image] = 0
    # plt.imshow(identified, cmap='grey');
    
    return [binary_image, roi_data, elbow_value, cfos, blurred_normalized]

def get_square_coordinates(min_row, min_col, max_row, max_col):
    rows = np.arange(min_row, max_row + 1)
    cols = np.arange(min_col, max_col + 1)
    
    coordinates = np.array(np.meshgrid(rows, cols)).T.reshape(-1, 2)
    
    return coordinates


def logistic_regression(dict_of_binary, ratio):
    # Calculate the minimum area in pixels using the conversion factor
    # min_diameter_threshold_micron = 10  # Minimum diameter of a neuron nucleus in µm
    min_diameter_threshold_micron = 3  # Minimum diameter of a neuron nucleus in µm
    min_area_threshold_micron = math.pi * (min_diameter_threshold_micron/2)**2
    min_area_threshold_pixels = min_area_threshold_micron * (ratio ** 2)
    
    all_my_regions = []
    for key, value in dict_of_binary.items():
        binary_image = value[0]
        blurred_normalized = value[4]
        # Label connected clusters
        labeled_array, num_clusters = label(~binary_image)  # Invert the array because we want to label False values
        # Find properties of each labeled region
        regions = regionprops(labeled_array)
        for region in regions:
            if region.area >= min_area_threshold_pixels:
                all_my_regions.append([region, binary_image, blurred_normalized]) 
        
    # Select 100 random items from the list along with their indices
    selected_regions = random.sample(all_my_regions, 100)    
        
    area_values = []
    perimeter_values = []
    eccentricity_values = []
    intensity_max = []
    intensity_mean = []
    intensity_max_background = []
    intensity_mean_background = []

    
    my_cells_uncroped = []   
    
    frame_pixel = int(15 * ratio)
    for selected_region in tqdm(selected_regions, desc="Processing inputs", unit="input"):
        region = selected_region[0]
        binary_image = selected_region[1]
        blurred_normalized = selected_region[2]
        
        area_values.append(region.area)
        perimeter_values.append(region.perimeter)
        eccentricity_values.append(region.eccentricity)
        coords = region.coords  # Coordinates of the current region

        colors = []
        for coord in coords:
            x, y = coord
            colors.append(blurred_normalized[x][y])
        intensity_max.append(max(colors))
        intensity_mean.append(np.mean(colors))
        
        # Find background color
        square_coordinates = get_square_coordinates(region.bbox[0], region.bbox[1], region.bbox[2], region.bbox[3])
        coordinates = np.array([coord for coord in square_coordinates if not np.any(np.all(coord == coords, axis=1))])    
        colors = []
        for coord in coordinates:
            x, y = coord
            colors.append(blurred_normalized[x][y])
        intensity_max_background.append(max(colors))
        intensity_mean_background.append(np.mean(colors))
        
        new_cfos = np.zeros(binary_image.shape).astype(int)
        new_blurred = blurred_normalized // 3
                
        for coord in coords:
            x, y = coord
            new_cfos[x][y] = 1
            new_blurred[x][y] = 255
            
        # Find rows and columns that are all False
        rows_to_keep = np.any(new_cfos, axis=1)
        cols_to_keep = np.any(new_cfos, axis=0)
        # Crop the array based on the identified rows and columns
        cropped_arr = blurred_normalized[rows_to_keep][:, cols_to_keep]
        
        new_boolean_arrays = []
        for boolean_array in [rows_to_keep, cols_to_keep]:
            # Find the index where consecutive True values start
            start_index = np.argmax(boolean_array)
            # Find the index where consecutive True values end
            end_index = len(boolean_array) - np.argmax(boolean_array[::-1]) - 1
            # Add 'frame_pixel' True values at the beginning and end
            modified_arr = np.concatenate([boolean_array[:start_index-frame_pixel],
                                           np.full(frame_pixel, True),
                                           boolean_array[start_index:end_index],
                                           np.full(frame_pixel, True),
                                           boolean_array[end_index+frame_pixel:]])
            new_boolean_arrays.append(modified_arr)
        
        # Crop the blurred_normalized and new_blurred
        cropped_blurred_normalized = blurred_normalized[new_boolean_arrays[0]][:, new_boolean_arrays[1]]
        cropped_new_blurred = new_blurred[new_boolean_arrays[0]][:, new_boolean_arrays[1]]
        
        # color_values.append(np.mean(colors_to_calculate_mean))
        my_cells_uncroped.append([cropped_blurred_normalized, cropped_new_blurred, cropped_arr])
        time.sleep(0.1)
    
    X = np.column_stack((area_values,
                           perimeter_values,
                           eccentricity_values,
                           intensity_max,
                          intensity_mean,
                           intensity_max_background,
                          intensity_mean_background,
                         ))
    
    # Display and label images
    y = []
    n = 0
    for image in my_cells_uncroped:
        n += 1
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
        max_edge = max([image[2].shape[0], image[2].shape[1]])
        ax1.imshow(image[0], cmap='viridis', extent=[0, max_edge, 0, max_edge])
        ax2.imshow(image[1], cmap='gray', extent=[0, max_edge, 0, max_edge])
        ax3.imshow(image[2], cmap='gray')
        plt.title('Image ' + str(n) + '/100')
        plt.draw()  # Redraw the figure
                
        plt.pause(0.1)
        # Take user input for each image
        user_input = input("Enter 0 (no-cell) or 1 (cell): ")
        # Validate user input (optional)
        while user_input not in ['0', '1', '2', '3', '4', '5', '6', '7', '8']:
            print("Invalid input. Please enter 0 or 1.")
            user_input = input("Enter 0 (no-cell) or 1 (cell): ")
    
        # Convert user input to int and append to the list
        y.append(int(user_input))
        plt.close()
    
    # Data exploration
    attributes = [
                    'area_values',
                    'perimeter_values',
                    # 'ratio_pa',
                  'eccentricity_values',
                   # 'ratio_back_max',
                   'intensity_max',
                   'intensity_mean',
                   'intensity_max_background',
                   'intensity_mean_background',
                    'y',
                  ]

    df = pd.DataFrame(np.column_stack((X, y)), columns=attributes)
    df['ratio_pa'] =  df.perimeter_values / df.area_values
    df['ratio_back_max'] =  df.intensity_mean_background / df.intensity_max
    
    df = df.drop('perimeter_values', axis=1)
    df = df.drop('area_values', axis=1)
    df = df.drop('intensity_max', axis=1)
    df = df.drop('intensity_mean', axis=1)
    df = df.drop('intensity_max_background', axis=1)
    df = df.drop('intensity_mean_background', axis=1)
    
    # df.hist(bins=50, figsize=(20,15))

    # corr_matrix = df.corr()
    # corr_matrix["y"].sort_values(ascending=False)
    # from pandas.plotting import scatter_matrix
    # scatter_matrix(df[attributes], figsize=(12, 8))

    
    X = df[['ratio_pa', 'ratio_back_max', 'eccentricity_values']].values
    
    
    # Training a logistic regressor
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Applying a PCA to reduce multicollineality, followed by training the logistic regression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    pipeline = Pipeline([
        ('standarize', StandardScaler()),
        ('pca', PCA(n_components=2)),
        ('classifier', LogisticRegression()),
        # ('classifier', RandomForestClassifier(n_estimators=100, random_state=42)),
    ])
    
    
    # from sklearn.feature_selection import RFE
    # from itertools import combinations
    # model = LogisticRegression()
    # rfe = RFE(model, n_features_to_select=1)
    # rfe.fit(X_train, y_train)
    # feature_ranking = list(zip(range(1, len(rfe.ranking_) + 1), rfe.ranking_))
    # feature_ranking.sort(key=lambda x: x[1])
    # best_accuracy = 0.0
    # best_feature_set = None
    
    # for k in range(1, len(feature_ranking) + 1):
    #     selected_features = [feature[0] - 1 for feature in feature_ranking[:k]]  # Adjust indices for zero-based indexing
    #     X_train_selected = X_train[:, selected_features]
    #     X_test_selected = X_test[:, selected_features]
    
    #     model.fit(X_train_selected, y_train)
    #     accuracy = model.score(X_test_selected, y_test)
    
    #     if accuracy > best_accuracy:
    #         best_accuracy = accuracy
    #         best_feature_set = selected_features
    # return best_accuracy, best_feature_set
    
    pipeline.fit(X_train, y_train)
        
    return pipeline, X_train, X_test, y_train, y_test
    
def plot_boundary(log_reg, X, y):
    regressor = log_reg.named_steps['classifier']
    plt.figure(figsize=(8, 6))
    
    X_standarized = log_reg.named_steps['standarize'].transform(X)
    X_pca = log_reg.named_steps['pca'].transform(X_standarized)
    
    h = .02  # Step size in the mesh
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = regressor.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
        
    plt.contourf(xx, yy, Z, cmap='Paired_r', alpha=0.8)
    
    # Scatter plot of data points
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolors='k', cmap='Paired_r', marker='o')
    plt.title('Logistic Regression with PCA Decision Boundary')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    
    # Create a custom legend for the scatter plot
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.Paired_r.colors[0], markersize=10),
               plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.Paired_r.colors[-1], markersize=10)]
    plt.legend(handles, ['Non-cell', 'Cell'])
    
    plt.show()


def micrometer_to_pixels(distance, ratio):
    converted = distance * ratio
    return converted

def diameter_to_area(diameter):
    area = math.pi * (diameter/2)**2
    return area

def apply_watershed_cfos(binary_image, blurred_normalized, ratio,
                         hyperparameters = None,
                         min_nuclear_diameter = 3, # um
                         min_hole_diameter = 5, # um
                         # distance_2 = 7.313, # (between 7-10)
                         # eccentricity_1 = 0.539, # (between 0.5-0.6)
                         # distance_3 = 4.01, # (between 3-5)
                         # color_1 = 0.894, # (between 0.8-1)
                         # distance_4 = 4.085, # max range! (between 3-20)
                         ):
    
    if hyperparameters is not None:
        distance_2 = hyperparameters['distance_2']
        distance_3 = hyperparameters['distance_3']
        distance_4 = hyperparameters['distance_4']
    else:
        print('No hyperparameters are found.')
    
# =============================================================================
# Dectect artifacts to separate
# =============================================================================
    
    remove_holes = morphology.remove_small_holes(binary_image, int(diameter_to_area(micrometer_to_pixels(min_hole_diameter, ratio))))
    # remove_holes = binary_image
    
    labeled_array, num_clusters = label(remove_holes)
    regions = regionprops(labeled_array)
    
    artifacts = []
    non_artifacts = []
    for region in regions:
        if region.area > diameter_to_area(micrometer_to_pixels(min_nuclear_diameter, ratio)):
            # if region.area < diameter_to_area(micrometer_to_pixels(distance_2, ratio)) or region.eccentricity < eccentricity_1:
                # non_artifacts.append(region)
            # else:
                artifacts.append(region)
    
    cell_landscape = np.zeros(binary_image.shape, dtype=bool)
    for region in artifacts:
        coords = region.coords  # Coordinates of the current region
        cell_landscape[coords[:, 0], coords[:, 1]] = True
    
    distance_ndi = ndi.distance_transform_edt(remove_holes)
    mask = np.zeros(distance_ndi.shape, dtype=bool)
    # coords = peak_local_max(distance_ndi, footprint=np.ones((3, 3)), labels=cell_landscape, min_distance=int(micrometer_to_pixels(distance_2, ratio)*0.5))
    coords = peak_local_max(distance_ndi, footprint=np.ones((3, 3)), labels=cell_landscape, min_distance=int(micrometer_to_pixels(distance_2, ratio)))
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance_ndi, markers, mask=cell_landscape)
    # plt.imshow(labels);
    separated_regions = regionprops(labels)
    for region in separated_regions:
        if region.area > diameter_to_area(micrometer_to_pixels(min_nuclear_diameter, ratio)):
            if region.area > diameter_to_area(micrometer_to_pixels(distance_3, ratio)):
                non_artifacts.append(region)
                    
# =============================================================================
# Identify actual cells        
# =============================================================================
    
    actual_cells = []         
    for region in non_artifacts:
        coords = region.coords
        
        # Find colors
        # colors = []
        # for coord in coords:
        #     x, y = coord
        #     colors.append(blurred_normalized[x][y])
            
        # Find background color
        # square_coordinates = get_square_coordinates(region.bbox[0], region.bbox[1], region.bbox[2], region.bbox[3])
        # coordinates = np.array([coord for coord in square_coordinates if not np.any(np.all(coord == coords, axis=1))])    
        # background_colors = []
        # for coord in coordinates:
        #     x, y = coord
        #     background_colors.append(blurred_normalized[x][y])

        # Calculate color ratio
        # color_ratio = np.mean(background_colors) / max(colors)
        
        if region.area > diameter_to_area(micrometer_to_pixels(min_nuclear_diameter, ratio)):
            # if color_ratio < color_1:
                if region.area > diameter_to_area(micrometer_to_pixels(distance_4, ratio)):
                    actual_cells.append(region)
        
    # cell_landscape = np.zeros(binary_image.shape, dtype=bool)
    # for region in actual_cells:
    #     coords = region.coords  # Coordinates of the current region
    #     cell_landscape[coords[:, 0], coords[:, 1]] = True
    # plt.imshow(cell_landscape);
            
    # median_area = max([region.area for region in actual_cells])
    # median_diameter = np.median([region.equivalent_diameter_area for region in actual_cells])
    
    return actual_cells
    

def apply_watershed(binary_image, blurred_normalized, log_reg, ratio, image_type):
    
    if image_type == 'tdt':
        min_diameter_threshold_micron = 10  # Minimum diameter of a neuron in µm
    elif image_type == 'cfos':
        min_diameter_threshold_micron = 3  # Minimum diameter of a neuron nucleus in µm
    min_area_threshold_micron = math.pi * (min_diameter_threshold_micron/2)**2
    min_diameter_threshold_pixels = min_diameter_threshold_micron * ratio
    min_area_threshold_pixels = min_area_threshold_micron * (ratio ** 2)
    
    # Apply logistic regression
    actual_cells = []
    labeled_array, num_clusters = label(~binary_image)  # Invert the array because we want to label False values
    regions = regionprops(labeled_array)
    for region in regions:
        if region.area >= min_area_threshold_pixels:
            
            colors = []
            coords = region.coords  # Coordinates of the current region
            for coord in coords:
                x, y = coord
                colors.append(blurred_normalized[x][y])
            
            # Find background color
            square_coordinates = get_square_coordinates(region.bbox[0], region.bbox[1], region.bbox[2], region.bbox[3])
            coordinates = np.array([coord for coord in square_coordinates if not np.any(np.all(coord == coords, axis=1))])    
            background_colors = []
            for coord in coordinates:
                x, y = coord
                background_colors.append(blurred_normalized[x][y])
            
            if log_reg.predict([[
                               # region.area,
                                 # region.perimeter,
                                 region.perimeter / region.area,
                                 np.mean(background_colors) / max(colors),
                                  region.eccentricity,
                                 # max(colors),
                                  # np.mean(colors),
                                 # max(background_colors),
                                 # np.mean(background_colors),
                                 ]]) == 1:
                actual_cells.append(region)
        
    # median_area = np.median([region.area for region in actual_cells])
    # median_diameter = np.median([region.equivalent_diameter_area for region in actual_cells])
    
    output_coords = []
    for region in tqdm(actual_cells, desc="Processing inputs", unit="input"):
        cell_landscape = np.zeros(binary_image.shape, dtype=bool)
        coords = region.coords
        cell_landscape[coords[:, 0], coords[:, 1]] = True
        
        remove_holes = morphology.remove_small_holes(cell_landscape, int(region.equivalent_diameter_area))
        labeled_array, num_clusters = label(remove_holes)
        new_region = regionprops(labeled_array)
        
        if len(new_region) != 1:
            print('More than one region found after removing holes.')
        else:
            new_region = new_region[0]
            if new_region.eccentricity < 0.8:
                if new_region.area >= min_area_threshold_pixels:
                    
                    colors = []
                    coords = new_region.coords  # Coordinates of the current region
                    for coord in coords:
                        x, y = coord
                        colors.append(blurred_normalized[x][y])
                    # Find background color
                    square_coordinates = get_square_coordinates(new_region.bbox[0], new_region.bbox[1], new_region.bbox[2], new_region.bbox[3])
                    coordinates = np.array([coord for coord in square_coordinates if not np.any(np.all(coord == coords, axis=1))])    
                    background_colors = []
                    for coord in coordinates:
                        x, y = coord
                        background_colors.append(blurred_normalized[x][y])
                    
                    if log_reg.predict([[
                                       # new_region.area,
                                         # new_region.perimeter,
                                         new_region.perimeter / new_region.area,
                                         np.mean(background_colors) / max(colors),
                                          new_region.eccentricity,
                                         # max(colors),
                                          # np.mean(colors),
                                         # max(background_colors),
                                         # np.mean(background_colors),
                                         ]]) == 1:
                        output_coords.append(new_region)
            
            if new_region.eccentricity >= 0.8:
                distance = ndi.distance_transform_edt(remove_holes)
                coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=remove_holes, min_distance=int(min_diameter_threshold_pixels))
                mask = np.zeros(distance.shape, dtype=bool)
                mask[tuple(coords.T)] = True
                markers, _ = ndi.label(mask)
                labels = watershed(-distance, markers, mask=remove_holes)
                # plt.imshow(labels);
                
                separated_regions = regionprops(labels)
                for separated_region in separated_regions:
                    if separated_region.area >= min_area_threshold_pixels:
                        colors = []
                        coords = separated_region.coords  # Coordinates of the current region
                        for coord in coords:
                            x, y = coord
                            colors.append(blurred_normalized[x][y])
                        # Find background color
                        square_coordinates = get_square_coordinates(separated_region.bbox[0], separated_region.bbox[1], separated_region.bbox[2], separated_region.bbox[3])
                        coordinates = np.array([coord for coord in square_coordinates if not np.any(np.all(coord == coords, axis=1))])    
                        background_colors = []
                        for coord in coordinates:
                            x, y = coord
                            background_colors.append(blurred_normalized[x][y])
                        if log_reg.predict([[
                                           # separated_region.area,
                                             # separated_region.perimeter,
                                             separated_region.perimeter / separated_region.area,
                                             np.mean(background_colors) / max(colors),
                                              separated_region.eccentricity,
                                             # max(colors),
                                              # np.mean(colors),
                                             # max(background_colors),
                                             # np.mean(background_colors),
                                             ]]) == 1:
                            output_coords.append(separated_region)
        time.sleep(0.1)
    
    # cell_landscape = np.zeros(binary_image.shape, dtype=bool)
    # for region in actual_cells:
    #     coords = region.coords  # Coordinates of the current region
    #     cell_landscape[coords[:, 0], coords[:, 1]] = True
    # # plt.imshow(cell_landscape);
    
    # # Apply extra filters
    # remove_holes = morphology.remove_small_holes(cell_landscape, median_area // 2)
    # # plt.imshow(remove_holes);
    # # remove_objects = morphology.remove_small_objects(remove_holes, int(median_area*0.95))
    # # plt.imshow(remove_objects);
    # remove_objects = remove_holes
    
    # # Apply watershed
    # distance = ndi.distance_transform_edt(remove_objects)  
    # coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=remove_objects, min_distance=int(min_diameter_threshold_pixels))
    # mask = np.zeros(distance.shape, dtype=bool)
    # mask[tuple(coords.T)] = True
    # markers, _ = ndi.label(mask)
    # labels = watershed(-distance, markers, mask=remove_objects)
    # # plt.imshow(labels);
    
    
    
    # output_coords = []
    # regions = regionprops(labels)
    # for region in regions:
    #     if region.area >= min_area_threshold_pixels:
    #         if log_reg.predict([[region.area,
    #                               region.perimeter,
    #                               # region.eccentricity,
    #                               ]]) == 1:
    #             output_coords.append(region)
    
    #========
    
    # # Label connected clusters
    # labeled_array, num_clusters = label(~binary_image)  # Invert the array because we want to label False values

    # # Find properties of each labeled region
    # regions = regionprops(labeled_array)
    
    # # Parameters for filtering
    # threshold_circularity = 0.75  # Circularity threshold

    # # Calculate the minimum area in pixels using the conversion factor
    # min_area_threshold_micron = 10  # Minimum area in µm^2
    # min_area_threshold_pixels = min_area_threshold_micron * (ratio ** 2)
        
    # # Filter elliptical/circular clusters based on circularity
    # circular_clusters = []
    # artifact_clusters = []
    
    # for region in tqdm(regions, desc="Processing inputs", unit="input"):
    #     colors_to_calculate_mean = []
    #     coords = region.coords  # Coordinates of the current region
    #     for coord in coords:
    #         x, y = coord
    #         colors_to_calculate_mean.append(blurred_normalized[x][y])
            
    #     if region.area >= min_area_threshold_pixels:
    #         # Calculate circularity: 4 * pi * area / (perimeter^2)
    #         if region.perimeter != 0:
    #             circularity = 4 * pi * region.area / (region.perimeter ** 2)
    #         else:
    #             circularity = 0
                    
    #         if log_reg.predict([[region.area,
    #                              region.perimeter,
    #                              np.mean(colors_to_calculate_mean),
    #                              # circularity
    #                              ]]) == 1:
                
    #             # Check circularity and minimum area
    #             if circularity >= threshold_circularity:
    #                     circular_clusters.append(region)
    #             elif circularity < threshold_circularity:
    #                 artifact_clusters.append(region)
    #     time.sleep(0.1)
    
    
    # # Create a collection of images of artifacts
    # my_artifacts = []
    # for region in artifact_clusters:
    #     # Select only those artifacts that are big
    #     if region.area >= min_area_threshold_pixels:            
    #         artifacts_binary = np.zeros(binary_image.shape, dtype=bool)
    #         coords = region.coords  # Coordinates of the current region
    #         artifacts_binary[coords[:, 0], coords[:, 1]] = True
    #         # Find rows and columns that are all False
    #         # rows_to_keep = np.any(artifacts_binary, axis=1)
    #         # cols_to_keep = np.any(artifacts_binary, axis=0)
    #         # Crop the array based on the identified rows and columns
    #         # cropped_arr = artifacts_binary[rows_to_keep][:, cols_to_keep]
    #         # Save the array of each individual artifact and its area
    #         artifact_info = [artifacts_binary, region.area]
    #         my_artifacts.append(artifact_info)
    # # plt.imshow(my_artifacts[25][0]);

    # # Analyze each artifact individually
    # separated_artifacts = []
    # for artifact in tqdm(my_artifacts, desc="Processing artifacts", unit="artifact"):
    #     # Calculate the distance to the edge
    #     distance_artifact = ndi.distance_transform_edt(artifact[0]) # Select the array
    #     distance_normalized_artifact = normalize_array(distance_artifact)
    #     # plt.imshow(distance_normalized);
    #     # Select only the center/s of the artifact
    #     threshold_artifact = 0.8 * 255
    #     binary_artifact = distance_normalized_artifact < threshold_artifact
    #     # plt.imshow(binary_artifact);
    #     # Count the number and sizes of the centers
    #     labeled_array_artifact, num_clusters_artifact = label(~binary_artifact)  # Invert the array because we want to label False values
    #     regions_artifact = regionprops(labeled_array_artifact)
    #     # Calculate the poderated mean of the area of the artifact by the area of its centers
    #     # Consider only if passes the min_area
    #     # Then append the coordinates of each
    #     total = sum(region.area for region in regions_artifact)
    #     for region in regions_artifact:
    #         factor = region.area/total
    #         separated_area = artifact[1] * factor
    #         # Suposing that the separated is circular, we can calculate the perimeter
    #         separated_perimeter = 2 * math.pi * math.sqrt((2 * separated_area) / math.pi)
            
    #         colors_to_calculate_mean = []
    #         coords = region.coords  # Coordinates of the current region
    #         for coord in coords:
    #             x, y = coord
    #             colors_to_calculate_mean.append(blurred_normalized[x][y])
            
    #         if log_reg.predict([[separated_area,
    #                              separated_perimeter,
    #                              np.mean(colors_to_calculate_mean),
    #                              # circularity
    #                              ]]) == 1:
            
    #             separated_artifacts.append(region)
    #     time.sleep(0.1)
    
    # output_coords = []
    # for circular_cluster in circular_clusters:
    #     output_coords.append(circular_cluster.coords)
    # for separated_artifact in separated_artifacts:
    #     output_coords.append(separated_artifact.coords)
    
    # coords_to_plot = []
    # for circular_cluster in circular_clusters:
    #     coords_to_plot.append(circular_cluster.coords)
    # for artifact_cluster in artifact_clusters:
    #     coords_to_plot.append(artifact_cluster.coords)

    return output_coords

def calculate_roi_area(roi, ratio):
    # Ensure the input array has the correct shape
    if roi.shape[0] != 1 or roi.shape[2] != 2:
        raise ValueError("Input array should have shape (1, n, 2)")

    # Extract the coordinates from the array
    x_coords = roi[0, :, 0]
    y_coords = roi[0, :, 1]

    # Apply the Shoelace formula to calculate the area
    area = 0.5 * np.abs(np.dot(x_coords, np.roll(y_coords, 1)) - np.dot(y_coords, np.roll(x_coords, 1)))
    converted_area = area * (ratio ** 2)
    return converted_area
    
# =============================================================================
# Run some functions together
# =============================================================================

def create_dict_rois(directory):
    
    dict_rois = {}
    n = 0
    for filename in os.listdir(directory):
        if filename.endswith(".nd2"):
            n += 1
            print('Analyzing image ' + str(n))
            file_path = os.path.join(directory, filename)
            image = nd2.imread(file_path)
            layers = split_layers(image)
            
            for layer_name, layer_data in layers.items():
                roi_coords = draw_ROI(layer_data, filename[:-4])
                if roi_coords is not None:
                    roi = np.array([roi_coords], dtype=np.int32)
                    dict_rois[filename[:-4]] = [roi, layers]
                    break
            else:
                # This block will execute if the "for" loop completes without a break
                print('The whole image will be used as a ROI')
                dimensions = list(layers.values())[0].shape
                custom_values = np.array([[[0, 0], [0, dimensions[1]], [dimensions[0], dimensions[1]], [dimensions[0], 0]]], dtype=np.int32)
                roi = np.array(custom_values, dtype=np.int32)
                dict_rois[filename[:-4]] = [roi, layers]
    return dict_rois


def create_dict_of_binary(dict_rois, layer):
    dict_of_binary = {}
    
    for tag, roi in dict_rois.items():
        roi_data = roi[0]
        layers = roi[1]
        binary_roi_elbow_cfos_cropped = image_to_binary(roi_data, layers, layer)
        dict_of_binary[tag] = binary_roi_elbow_cfos_cropped
    
    return dict_of_binary


def compiler(directory, dict_of_binary, ratio, layer, hyperparameters):
        
    file_name = []
    background_threshold = []
    num_cells = []
    roi_surface = []
    cells_per_squared_mm = []
    
    n=0
    for key, value in dict_of_binary.items():
        n+=1
        print('Analyzing image ' + str(n) + '/' + str(len(dict_of_binary)) + ':' + str(key))
        binary_image = ~value[0]
        blurred_normalized = value[4]
        output_coords = apply_watershed_cfos(binary_image, blurred_normalized, ratio, hyperparameters)
        
        # name = key + '_watershed.tif'
        # file_path = os.path.join(directory, name)
        # plt.imshow(output_coords)
        # plt.savefig(file_path, dpi=300)
        # plt.close()
        
        print(f"Number of Cells - {len(output_coords)}")
        roi_area = calculate_roi_area(value[1], ratio)
        # print(f"{key[:-4]}: ROI Area in µm^2 - {roi_area}")
        cells_mm_2 = ((10**6)*len(output_coords))/roi_area
        # print(f"{key[:-4]}: Cells per square millimeter - {cells_mm_2}")
        
        file_name.append(key)
        background_threshold.append(value[2])
        num_cells.append(len(output_coords))
        roi_surface.append(roi_area)
        cells_per_squared_mm.append(cells_mm_2)
        
        # Create an image with the results
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # ======== 1st panel: original image with the ROI
        mask = np.zeros_like(value[3])
        cv2.fillPoly(mask, value[1], 255)
        layer_roi = np.where(mask == 255, value[3], 0)#.astype(int)
        rows_to_keep = np.any(layer_roi, axis=1)
        cols_to_keep = np.any(layer_roi, axis=0)
        to_plot = value[3][rows_to_keep][:, cols_to_keep]
        axes[0].imshow(to_plot, cmap='grey')
        axes[0].set_title('Original figure')
        # roi_polygon = Polygon(value[1].squeeze(), edgecolor='red', linewidth=2, facecolor='none')
        # axes[0].add_patch(roi_polygon)
        # Hide the axis ticks and values
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].set_xticklabels([])
        axes[0].set_yticklabels([])
        
        # ======== 2nd panel: threshold (elbow)
        hist, bins = np.histogram(value[4][value[4] != 0], bins=64, range=(1, 256)) 
        peaks, _ = find_peaks(hist)
        closest_peak_index = np.argmax(hist[peaks])
        subset_hist = hist[peaks[closest_peak_index]:]
        subset_bins = bins[peaks[closest_peak_index]:-1]
        knee = KneeLocator(subset_bins, subset_hist, curve='convex', direction='decreasing')
        elbow_value = knee.elbow
        axes[1].plot(bins[:-1], hist)
        axes[1].axvline(x=bins[peaks[closest_peak_index]], color='g', linestyle='--', label=f'Closest Peak to 255: {bins[peaks[closest_peak_index]]:.2f}')
        axes[1].axvline(x=elbow_value, color='b', linestyle='--', label=f'Threshold: {elbow_value:.2f}')
        axes[1].set_title('Threshold')
        axes[1].set_xlabel('Pixel Value')  # Corrected from axes[1].xlabel('Pixel Value')
        axes[1].set_ylabel('Frequency')    # Corrected from axes[1].ylabel('Frequency')
        axes[1].legend()
        
        # ======== 3rd panel: identified cells
        number_list = [x for x in range(len(output_coords))]
        number_list.remove(0)
        number_list = random.choices(number_list, k=len(number_list))
        
        artificial_binary = np.zeros(binary_image.shape)
        for coords, number in zip(output_coords, number_list):
            artificial_binary[coords.coords[:, 0], coords.coords[:, 1]] = number
        
        artificial_binary = artificial_binary[rows_to_keep][:, cols_to_keep]
        artificial_binary[artificial_binary == 0] = np.nan
        
        axes[2].set_facecolor('black')
        axes[2].imshow(artificial_binary, cmap='gist_rainbow')
        
        axes[2].set_title('Identified cells')
        # Hide the axis ticks and values
        axes[2].set_xticks([])
        axes[2].set_yticks([])
        axes[2].set_xticklabels([])
        axes[2].set_yticklabels([])
        
        # Adjust layout to prevent overlapping
        plt.tight_layout()
        
        # Save the image as a JPEG file
        name = key + '_' + layer + '.jpg'
        file_path = os.path.join(directory, name)
        plt.savefig(file_path)
        plt.close()
    
    # Creating a DataFrame
    output_dict = {'file_name': file_name,
                 'background_threshold': background_threshold,
                 'num_cells': num_cells,
                 'roi_area': roi_surface,
                 'cells_per_squared_mm': cells_per_squared_mm}
    df = pd.DataFrame(output_dict)
    df_path = os.path.join(directory, 'results.csv')
    df.to_csv(df_path, index=False)
    
    # return artificial_binary

# =============================================================================
# Genetic Algorithm for hyperparameter optimization
# =============================================================================

def evaluate(dict_of_binary, ratio, hyperparameters, actual_values):
    
    predicted_values = {}
    for key, value in dict_of_binary.items():
        binary_image = ~value[0]
        blurred_normalized = value[4]
        cells = apply_watershed_cfos(binary_image, blurred_normalized, ratio, hyperparameters)
        predicted_values[key] = len(cells)

    common_keys = set(actual_values.keys()) & set(predicted_values.keys())

    # Calculate RMSE
    squared_diff = [(actual_values[key] - predicted_values[key])**2 for key in common_keys]
    mse = sum(squared_diff) / len(common_keys)
    rmse = math.sqrt(mse)
    
    # Calculate MAE
    absolute_diff = [abs(actual_values[key] - predicted_values[key]) for key in common_keys]
    mae = sum(absolute_diff) / len(common_keys)
    
    # Calculate the relative error
    relative_error = [abs(actual_values[key] - predicted_values[key]) / actual_values[key] for key in common_keys]
    re_mean = sum(relative_error) / len(common_keys)

    return re_mean, predicted_values

def random_search(dict_of_binary, num_iterations, ratio, actual_values):
    best_loss = float('inf')
    best_hyperparameters = None
    best_predicted_values = None

    for _ in tqdm(range(num_iterations), desc="Performing iterations", unit="iterations"):
        
        # Randomly sample hyperparameters from the search space
        hyperparameters = {
            'distance_2': random.uniform(7, 10), # (between 7-10)
            'distance_3': random.uniform(3, 5), # (between 3-5)
            'distance_4': random.uniform(3, 20), # max range! (between 3-20)
        }
        
        loss, predicted_values = evaluate(dict_of_binary, ratio, hyperparameters, actual_values)
                
        # Update best accuracy and hyperparameters if current iteration is better
        if loss < best_loss:
            best_loss = loss
            best_hyperparameters = hyperparameters
            best_predicted_values = predicted_values
        time.sleep(0.1)

    return best_loss, best_hyperparameters, best_predicted_values

def initialize_population(population_size, hyperparameter_ranges):
    population = []
    for _ in range(population_size):
        individual = {}
        for hyperparameter, (min_value, max_value) in hyperparameter_ranges.items():
            # Initialize each hyperparameter with a random value within its specified range
            individual[hyperparameter] = random.uniform(min_value, max_value)
        population.append(individual)
    return population

def crossover(parent1, parent2):
    # Perform crossover between two parents to create a child
    child = copy.deepcopy(parent1)
    for key in child:
        if random.random() > 0.5:
            child[key] = parent2[key]
    return child

def mutate(individual, hyperparameter_ranges, mutation_rate):
    mutated_individual = individual.copy()  # Make a copy to avoid modifying the original individual
    
    # Iterate through hyperparameters and apply mutation with a certain probability
    for hyperparameter, value in mutated_individual.items():
        if random.random() < mutation_rate:
            # Calculate a mutation value based on the range of the hyperparameter
            mutation_value = random.uniform(-0.1 * (hyperparameter_ranges[hyperparameter][1] - hyperparameter_ranges[hyperparameter][0]),
                                            0.1 * (hyperparameter_ranges[hyperparameter][1] - hyperparameter_ranges[hyperparameter][0]))
            
            # Apply mutation to the hyperparameter, respecting the specified limits
            mutated_value = value + mutation_value
            mutated_value = max(hyperparameter_ranges[hyperparameter][0], min(hyperparameter_ranges[hyperparameter][1], mutated_value))
            
            mutated_individual[hyperparameter] = mutated_value
    
    return mutated_individual

def genetic_algorithm(dict_of_binary, num_iterations, ratio, actual_values, population_size=10, initial_mutation_rate=0.7, mutation_rate_decay=0.5):
    hyperparameter_ranges = {
        'distance_2': (3, 20),
        # 'eccentricity_1': (0.5, 0.6),
        'distance_3': (3, 20),
        # 'color_1': (0.8, 1),
        'distance_4': (3, 20),
    }
    
    population = initialize_population(population_size, hyperparameter_ranges)
    best_loss = float('inf')
    best_hyperparameters = None
    best_predicted_values = None
    
    no_improvement_counter = 0
    patience=20
    loss_list = []
    
    # Initialize the mutation rate
    mutation_rate = initial_mutation_rate

    for generation in tqdm(range(num_iterations), desc="Going through generations", unit="generations"):
        # Evaluate fitness for each individual in the population
        fitness = []
        for individual in population:
            loss, predicted_values = evaluate(dict_of_binary, ratio, individual, actual_values)
            fitness.append((loss, individual, predicted_values))

        # Sort by fitness and select the top individuals
        fitness.sort(key=lambda x: x[0])
        top_individuals = fitness[:population_size // 2]

        # Update best accuracy and hyperparameters if current generation is better
        loss_list.append(top_individuals[0][0])
        if top_individuals[0][0] < best_loss:
            best_loss, best_hyperparameters, best_predicted_values = top_individuals[0]
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1

        # Early stopping check
        # if no_improvement_counter >= patience:
        #     print(f"Stopping early as there is no improvement for {patience} iterations.")
        #     break

        # Create the next generation through crossover and mutation
        new_population = [individual for (_, individual, _) in top_individuals]
        while len(new_population) < population_size:
            parent1, parent2 = random.choices(top_individuals, k=2)
            child = crossover(parent1[1], parent2[1])
            mutated_child = mutate(child, hyperparameter_ranges, mutation_rate)
            new_population.append(mutated_child)

        population = new_population
        
        # Decay the mutation rate
        mutation_rate *= mutation_rate_decay
        if mutation_rate < 0.2:
            mutation_rate = 0.2

        time.sleep(0.1)

    return best_loss, best_hyperparameters, best_predicted_values, loss_list


# =============================================================================
# Master functions
# =============================================================================

def train_model(dict_rois, actual_values, layer, ratio):
    dict_of_binary = create_dict_of_binary(dict_rois, layer)
    best_loss, best_hyperparameters, best_predicted_values, loss_list = genetic_algorithm(dict_of_binary, 50,  ratio, actual_values)
    print("Best Loss:", best_loss)
    print("Best Hyperparameters:", best_hyperparameters)
    print("Best Predicted Values:", best_predicted_values)
    
    plt.plot(range(1, len(loss_list) + 1), loss_list, marker='o', linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
    return best_loss, best_hyperparameters, best_predicted_values

def plot_correlation(actual_values, best_predicted_values):
    
    # Extract keys and values
    keys = list(actual_values.keys())
    values1 = [actual_values[key] for key in keys]
    values2 = [best_predicted_values[key] for key in keys]
    
    # Plot
    plt.scatter(values1, values2)
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.title('Correlation Plot')
    plt.grid(True)
    
    # Calculate correlation coefficient
    correlation_coefficient = np.corrcoef(values1, values2)[0, 1]
    print("Correlation coefficient:", correlation_coefficient)
    
    # Plot correlation line
    x_values = np.array(values1)
    y_values = correlation_coefficient * x_values + np.mean(values2) - correlation_coefficient * np.mean(values1)
    plt.plot(x_values, y_values, color='red', label=f'Correlation line: y = {correlation_coefficient:.2f}x + {np.mean(values2) - correlation_coefficient * np.mean(values1):.2f}')
    
    # Plot correlation value and equation
    plt.text(0.1, 0.9, f'Correlation coefficient: {correlation_coefficient:.2f}', transform=plt.gca().transAxes)
    plt.text(0.1, 0.85, f'Correlation equation: y = {correlation_coefficient:.2f}x + {np.mean(values2) - correlation_coefficient * np.mean(values1):.2f}', transform=plt.gca().transAxes)

    plt.show()
    
def analyze_images(dict_rois, directory, layer_dict, ratio):
    # my_binaries = {}
    for layer, hyperparameters in layer_dict.items():
        print('Analyzing ' + layer )
        dict_of_binary = create_dict_of_binary(dict_rois, layer)
        artifical_binary = compiler(directory, dict_of_binary, ratio, layer, hyperparameters)
        # my_binaries[layer] = artifical_binary
    # return my_binaries
    




















