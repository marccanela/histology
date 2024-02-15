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
from math import pi
from scipy.ndimage import label
from scipy import ndimage as ndi
import os
from matplotlib.patches import Polygon
import random
import math
from tqdm import tqdm
import time

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
    print('To use the next layer to draw the ROI, just close this window.')

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

def image_to_binary(image, tag, layer):
    layers = split_layers(image)
    
    for layer_name, layer_data in layers.items():
        roi_coords = draw_ROI(layer_data, tag)
        if roi_coords is not None:
            roi = np.array([roi_coords], dtype=np.int32)
            break
    
    # Apply the defined ROI
    cfos = layers[layer]  # Select layer_0, layer_1, layer_2, etc.
    mask = np.zeros_like(cfos)
    cv2.fillPoly(mask, roi, 255)
    layer_roi = np.where(mask == 255, cfos, 0).astype(int)
    # plt.imshow(layer_roi, cmap='grey');
    
    # Apply Gaussian blur to reduce noise
    # layer_roi = cv2.GaussianBlur(layer_roi, (5, 5), 0)
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
    
    return [binary_image, roi, elbow_value, cfos, blurred_normalized]


def logistic_regression(dict_of_binary, ratio):
    # Calculate the minimum area in pixels using the conversion factor
    min_area_threshold_micron = 10  # Minimum area in µm^2
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
    # circularity_values = []
    color_values = []
    my_cells_uncroped = []   
    
    for selected_region in tqdm(selected_regions, desc="Processing inputs", unit="input"):
        region = selected_region[0]
        binary_image = selected_region[1]
        blurred_normalized = selected_region[2]
        
        area_values.append(region.area)
        perimeter_values.append(region.perimeter)
        # if region.perimeter != 0:
        #     circularity = 4 * pi * region.area / (region.perimeter ** 2)
        # else:
        #     circularity = 0
        # circularity_values.append(circularity)
        
        new_cfos = np.zeros(binary_image.shape).astype(int)
        new_blurred = blurred_normalized // 3
        colors_to_calculate_mean = []
        coords = region.coords  # Coordinates of the current region
        for coord in coords:
            x, y = coord
            new_blurred[x][y] = 255
            
            new_cfos[x][y] = 1
            # Find rows and columns that are all False
            rows_to_keep = np.any(new_cfos, axis=1)
            cols_to_keep = np.any(new_cfos, axis=0)
            # Crop the array based on the identified rows and columns
            cropped_arr = blurred_normalized[rows_to_keep][:, cols_to_keep]
            
            colors_to_calculate_mean.append(blurred_normalized[x][y])
    
        color_values.append(np.mean(colors_to_calculate_mean))
        my_cells_uncroped.append([blurred_normalized, new_blurred, cropped_arr])
        time.sleep(0.1)
    
    X = np.column_stack((area_values,
                         perimeter_values,
                         color_values,
                         # circularity_values
                         ))
    
    # Display and label images
    y = []
    n = 0
    for image in my_cells_uncroped:
        n += 1
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
        max_edge = max([image[2].shape[0], image[2].shape[1]])
        ax1.imshow(image[0], cmap='gray', extent=[0, max_edge, 0, max_edge])
        ax2.imshow(image[1], cmap='gray', extent=[0, max_edge, 0, max_edge])
        ax3.imshow(image[2], cmap='gray')
        plt.title('Image ' + str(n) + '/100')
        plt.draw()  # Redraw the figure
                
        plt.pause(0.1)
        # Take user input for each image
        user_input = input("Enter a value (0 for non-positive or 1 for positive): ")
        # Validate user input (optional)
        while user_input not in ['0', '1']:
            print("Invalid input. Please enter 0 or 1.")
            user_input = input("Enter a value (0 for non-positive or 1 for positive): ")
    
        # Convert user input to int and append to the list
        y.append(int(user_input))
        plt.close()
    
    # Training a logistic regressor
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Applying a PCA to reduce multicollineality, followed by training the logistic regression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    pipeline = Pipeline([
        ('standarize', StandardScaler()),
        ('pca', PCA(n_components=2)),  # You can specify the number of components
        ('log_reg', LogisticRegression())
    ])
    
    pipeline.fit(X_train, y_train)
        
    return pipeline, X_train, X_test, y_train, y_test

def plot_boundary(log_reg, X, y):
    regressor = log_reg.named_steps['log_reg']
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

def watershed(binary_image, blurred_normalized, log_reg, ratio):
    # Label connected clusters
    labeled_array, num_clusters = label(~binary_image)  # Invert the array because we want to label False values

    # Find properties of each labeled region
    regions = regionprops(labeled_array)
    
    # Parameters for filtering
    threshold_circularity = 0.75  # Circularity threshold

    # Calculate the minimum area in pixels using the conversion factor
    min_area_threshold_micron = 10  # Minimum area in µm^2
    min_area_threshold_pixels = min_area_threshold_micron * (ratio ** 2)
        
    # Filter elliptical/circular clusters based on circularity
    circular_clusters = []
    artifact_clusters = []
    
    for region in tqdm(regions, desc="Processing inputs", unit="input"):
        colors_to_calculate_mean = []
        coords = region.coords  # Coordinates of the current region
        for coord in coords:
            x, y = coord
            colors_to_calculate_mean.append(blurred_normalized[x][y])
            
        if region.area >= min_area_threshold_pixels:
            # Calculate circularity: 4 * pi * area / (perimeter^2)
            if region.perimeter != 0:
                circularity = 4 * pi * region.area / (region.perimeter ** 2)
            else:
                circularity = 0
                    
            if log_reg.predict([[region.area,
                                 region.perimeter,
                                 np.mean(colors_to_calculate_mean),
                                 # circularity
                                 ]]) == 1:
                
                # Check circularity and minimum area
                if circularity >= threshold_circularity:
                        circular_clusters.append(region)
                elif circularity < threshold_circularity:
                    artifact_clusters.append(region)
        time.sleep(0.1)

    # Create a collection of images of artifacts
    my_artifacts = []
    for region in artifact_clusters:
        # Select only those artifacts that are big
        if region.area >= min_area_threshold_pixels:     
            artifacts_binary = np.zeros(binary_image.shape, dtype=bool)
            coords = region.coords  # Coordinates of the current region
            artifacts_binary[coords[:, 0], coords[:, 1]] = True
            # Find rows and columns that are all False
            # rows_to_keep = np.any(artifacts_binary, axis=1)
            # cols_to_keep = np.any(artifacts_binary, axis=0)
            # Crop the array based on the identified rows and columns
            # cropped_arr = artifacts_binary[rows_to_keep][:, cols_to_keep]
            # Save the array of each individual artifact and its area
            artifact_info = [artifacts_binary, region.area]
            my_artifacts.append(artifact_info)
    # plt.imshow(my_artifacts[25][0]);

    # Analyze each artifact individually
    separated_artifacts = []
    for artifact in tqdm(my_artifacts, desc="Processing artifacts", unit="artifact"):
        # Calculate the distance to the edge
        distance_artifact = ndi.distance_transform_edt(artifact[0]) # Select the array
        distance_normalized_artifact = normalize_array(distance_artifact)
        # plt.imshow(distance_normalized);
        # Select only the center/s of the artifact
        threshold_artifact = 0.8 * 255
        binary_artifact = distance_normalized_artifact < threshold_artifact
        # plt.imshow(binary_artifact);
        # Count the number and sizes of the centers
        labeled_array_artifact, num_clusters_artifact = label(~binary_artifact)  # Invert the array because we want to label False values
        regions_artifact = regionprops(labeled_array_artifact)
        # Calculate the poderated mean of the area of the artifact by the area of its centers
        # Consider only if passes the min_area
        # Then append the coordinates of each
        total = sum(region.area for region in regions_artifact)
        for region in regions_artifact:
            factor = region.area/total
            separated_area = artifact[1] * factor
            # Suposing that the separated is circular, we can calculate the perimeter
            separated_perimeter = 2 * math.pi * math.sqrt((2 * separated_area) / math.pi)
            
            colors_to_calculate_mean = []
            coords = region.coords  # Coordinates of the current region
            for coord in coords:
                x, y = coord
                colors_to_calculate_mean.append(blurred_normalized[x][y])
            
            if log_reg.predict([[separated_area,
                                 separated_perimeter,
                                 np.mean(colors_to_calculate_mean),
                                 # circularity
                                 ]]) == 1:
            
                separated_artifacts.append(region)
        time.sleep(0.1)
    
    output_coords = []
    for circular_cluster in circular_clusters:
        output_coords.append(circular_cluster.coords)
    for separated_artifact in separated_artifacts:
        output_coords.append(separated_artifact.coords)
    
    coords_to_plot = []
    for circular_cluster in circular_clusters:
        coords_to_plot.append(circular_cluster.coords)
    for artifact_cluster in artifact_clusters:
        coords_to_plot.append(artifact_cluster.coords)

    return output_coords, coords_to_plot


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
# Run the whole script
# =============================================================================

def create_dict_of_binary(directory, layer):
    dict_of_binary = {}
    
    for filename in os.listdir(directory):
        if filename.endswith(".nd2"):
            file_path = os.path.join(directory, filename)
            image = nd2.imread(file_path)
            binary_roi_elbow_cfos_cropped = image_to_binary(image, filename[:-4], layer)
            dict_of_binary[filename] = binary_roi_elbow_cfos_cropped
    
    return dict_of_binary

def compiler(directory, dict_of_binary, ratio, log_reg):
        
    file_name = []
    background_threshold = []
    num_cells = []
    roi_surface = []
    cells_per_squared_mm = []
    
    n=0
    for key, value in dict_of_binary.items():
        n+=1
        print('Analyzing image ' + str(key[:-4]) + ' (' + str(n) + '/' + len(dict_of_binary) + ')')
        binary_image = value[0]
        blurred_normalized = value[4]
        output_coords, plot_coords = watershed(binary_image, blurred_normalized, log_reg, ratio)
        print(f"Number of Cells - {len(output_coords)}")
        roi_area = calculate_roi_area(value[1], ratio)
        # print(f"{key[:-4]}: ROI Area in µm^2 - {roi_area}")
        cells_mm_2 = ((10**6)*len(output_coords))/roi_area
        # print(f"{key[:-4]}: Cells per square millimeter - {cells_mm_2}")
        
        file_name.append(key[:-4])
        background_threshold.append(value[2])
        num_cells.append(len(output_coords))
        roi_surface.append(roi_area)
        cells_per_squared_mm.append(cells_mm_2)
        
        # Create an image with the results
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
        # 1st panel: original image with the ROI
        axes[0].imshow(value[3], cmap='grey')
        axes[0].set_title('Original figure')
        roi_polygon = Polygon(value[1].squeeze(), edgecolor='red', linewidth=2, facecolor='none')
        axes[0].add_patch(roi_polygon)
        # Hide the axis ticks and values
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].set_xticklabels([])
        axes[0].set_yticklabels([])
        
        # 2nd panel: threshold (elbow)
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
        
        # 3rd panel: identified cells
        artifical_binary = np.zeros(value[3].shape, dtype=bool)
        for coords in plot_coords:
            artifical_binary[coords[:, 0], coords[:, 1]] = True
        axes[2].imshow(artifical_binary, cmap='grey')
        axes[2].set_title('Identified cells')
        # Hide the axis ticks and values
        axes[2].set_xticks([])
        axes[2].set_yticks([])
        axes[2].set_xticklabels([])
        axes[2].set_yticklabels([])
        
        # Adjust layout to prevent overlapping
        plt.tight_layout()
        
        # Save the image as a JPEG file
        name = key[:-4] + '.jpg'
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


























