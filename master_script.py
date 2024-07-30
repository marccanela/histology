"""
Created on Tue Jan  9 09:50:17 2024
@author: mcanela
"""

import copy
import math
import os
import random
import time

import cv2
import matplotlib.pyplot as plt
import nd2
import numpy as np
import pandas as pd
from kneed import KneeLocator
from matplotlib.patches import Polygon
from matplotlib.widgets import PolygonSelector
from scipy import ndimage as ndi
from scipy.ndimage import label
from scipy.signal import find_peaks
from skimage import morphology
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import watershed
from tqdm import tqdm


def split_layers(image):
    layers = {}
    for n in range(image.shape[0]):
        my_layer = image[n]
        layers["layer_" + str(n)] = my_layer
    return layers


def draw_ROI(layer_data, tag):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(layer_data, cmap="gray")
    ax.set_title(tag)
    print("Draw a ROI on this layer and click on the initial dot to finish.")
    print("To use the next layer to draw the ROI, just close this layer.")
    print("Close all layers to use the whole image.")

    polygon_coords = []  # List to store selected points

    def onselect(verts):
        polygon_coords.append(verts)
        print("The ROI has been correctly saved.")

    # Set the facecolor parameter to 'r' for red pointer
    polygon_selector = PolygonSelector(
        ax, onselect, props=dict(color="r", linestyle="-", linewidth=2, alpha=0.5)
    )

    plt.show(block=True)

    if len(polygon_coords) == 0:
        return None
    else:
        return polygon_coords[-1]


def calculate_elbow(hist, bins):
    peaks, _ = find_peaks(hist)
    closest_peak_index = np.argmax(hist[peaks])

    # Create a subset of histogram and bins values between the identified peak and the end
    subset_hist = hist[peaks[closest_peak_index] :]
    subset_bins = bins[peaks[closest_peak_index] : -1]

    # Find the elbow using KneeLocator on the subset
    knee = KneeLocator(subset_bins, subset_hist, curve="convex", direction="decreasing")
    elbow_value = knee.elbow
    return elbow_value


def background_threshold(blurred_normalized):
    hist, bins = np.histogram(
        blurred_normalized[blurred_normalized != 0], bins=64, range=(1, 256)
    )
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
    layer_roi = np.where(mask == 255, cfos, 0)  # .astype(int)
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


def micrometer_to_pixels(distance, ratio):
    converted = distance * ratio
    return converted


def diameter_to_area(diameter):
    area = math.pi * (diameter / 2) ** 2
    return area


def apply_watershed_cfos(
    binary_image,
    blurred_normalized,
    ratio,
    hyperparameters=None,
    min_nuclear_diameter=3,  # um
    min_hole_diameter=5,  # um
):

    if hyperparameters is not None:
        distance_2 = hyperparameters["distance_2"]
        color_1 = hyperparameters["color_1"]
        distance_4 = hyperparameters["distance_4"]
        # eccentricity_1 = hyperparameters['eccentricity_1']
        # distance_3 = hyperparameters['distance_3']

    # =================== Dectect artifacts to separate ============
    nuclear_area_px = diameter_to_area(
        micrometer_to_pixels(min_nuclear_diameter, ratio)
    )

    remove_holes = morphology.remove_small_holes(
        binary_image,
        int(diameter_to_area(micrometer_to_pixels(min_hole_diameter, ratio))),
    )

    labeled_array, num_clusters = label(remove_holes)
    regions = regionprops(labeled_array)

    artifacts = []
    for region in regions:
        if region.area > nuclear_area_px:
            artifacts.append(region)

    # areas_artifacts = np.array([artifact.??? for artifact in artifacts])
    # standardized_areas = [(x - np.mean(areas_artifacts)) / np.std(areas_artifacts) for x in areas_artifacts]
    # plt.hist(standardized_areas);

    cell_landscape = np.zeros(binary_image.shape, dtype=bool)
    for artifact in artifacts:
        coords = artifact.coords  # Coordinates of the current region
        cell_landscape[coords[:, 0], coords[:, 1]] = True
    # plt.imshow(cell_landscape);
    distance_ndi = ndi.distance_transform_edt(cell_landscape)

    diameters = np.array([artifact.equivalent_diameter_area for artifact in artifacts])
    # min_distance = np.mean(diameters) + np.std(diameters) * distance_2
    min_distance = np.mean(diameters) * distance_2
    if min_distance < 1:
        min_distance = 1

    mask = np.zeros(distance_ndi.shape, dtype=bool)
    coords = peak_local_max(
        distance_ndi,
        min_distance=int(min_distance),
        exclude_border=False,
        labels=cell_landscape,
        # num_peaks_per_label=artifact.area // nuclear_area_px
    )
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance_ndi, markers, mask=cell_landscape)
    # plt.imshow(labels);

    non_artifacts = []
    separated_regions = regionprops(labels)
    for region in separated_regions:
        if region.area > nuclear_area_px:
            non_artifacts.append(region)

    # ====================== Identify actual cells =====================

    # Identify global color of the cells
    colors_global = []
    for region in non_artifacts:
        coords = region.coords
        for coord in coords:
            x, y = coord
            colors_global.append(blurred_normalized[x][y])
    # min_color = np.mean(colors_global) + np.std(colors_global) * color_1
    min_color = np.mean(colors_global) * color_1
    if min_color < 0:
        min_color = 0
    elif min_color > 255:
        min_color = 255
    min_color = int(color_1)

    measures = np.array([non_artifact.area for non_artifact in non_artifacts])
    # min_area = np.mean(measures) + np.std(measures) * distance_4
    min_area = np.mean(measures) * distance_4
    if min_area < 0:
        min_area = 0

    actual_cells = []
    for region in non_artifacts:
        coords = region.coords
        colors = []
        for coord in coords:
            x, y = coord
            colors.append(blurred_normalized[x][y])

        if np.mean(colors) > min_color:
            if region.area > min_area:
                actual_cells.append(region)

    return actual_cells


def apply_watershed(binary_image, blurred_normalized, log_reg, ratio, image_type):

    if image_type == "tdt":
        min_diameter_threshold_micron = 10  # Minimum diameter of a neuron in µm
    elif image_type == "cfos":
        min_diameter_threshold_micron = 3  # Minimum diameter of a neuron nucleus in µm
    min_area_threshold_micron = math.pi * (min_diameter_threshold_micron / 2) ** 2
    min_diameter_threshold_pixels = min_diameter_threshold_micron * ratio
    min_area_threshold_pixels = min_area_threshold_micron * (ratio**2)

    # Apply logistic regression
    actual_cells = []
    labeled_array, num_clusters = label(
        ~binary_image
    )  # Invert the array because we want to label False values
    regions = regionprops(labeled_array)
    for region in regions:
        if region.area >= min_area_threshold_pixels:

            colors = []
            coords = region.coords  # Coordinates of the current region
            for coord in coords:
                x, y = coord
                colors.append(blurred_normalized[x][y])

            # Find background color
            square_coordinates = get_square_coordinates(
                region.bbox[0], region.bbox[1], region.bbox[2], region.bbox[3]
            )
            coordinates = np.array(
                [
                    coord
                    for coord in square_coordinates
                    if not np.any(np.all(coord == coords, axis=1))
                ]
            )
            background_colors = []
            for coord in coordinates:
                x, y = coord
                background_colors.append(blurred_normalized[x][y])

            if (
                log_reg.predict(
                    [
                        [
                            # region.area,
                            # region.perimeter,
                            region.perimeter / region.area,
                            np.mean(background_colors) / max(colors),
                            region.eccentricity,
                            # max(colors),
                            # np.mean(colors),
                            # max(background_colors),
                            # np.mean(background_colors),
                        ]
                    ]
                )
                == 1
            ):
                actual_cells.append(region)

    # median_area = np.median([region.area for region in actual_cells])
    # median_diameter = np.median([region.equivalent_diameter_area for region in actual_cells])

    output_coords = []
    for region in tqdm(actual_cells, desc="Processing inputs", unit="input"):
        cell_landscape = np.zeros(binary_image.shape, dtype=bool)
        coords = region.coords
        cell_landscape[coords[:, 0], coords[:, 1]] = True

        remove_holes = morphology.remove_small_holes(
            cell_landscape, int(region.equivalent_diameter_area)
        )
        labeled_array, num_clusters = label(remove_holes)
        new_region = regionprops(labeled_array)

        if len(new_region) != 1:
            print("More than one region found after removing holes.")
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
                    square_coordinates = get_square_coordinates(
                        new_region.bbox[0],
                        new_region.bbox[1],
                        new_region.bbox[2],
                        new_region.bbox[3],
                    )
                    coordinates = np.array(
                        [
                            coord
                            for coord in square_coordinates
                            if not np.any(np.all(coord == coords, axis=1))
                        ]
                    )
                    background_colors = []
                    for coord in coordinates:
                        x, y = coord
                        background_colors.append(blurred_normalized[x][y])

                    if (
                        log_reg.predict(
                            [
                                [
                                    # new_region.area,
                                    # new_region.perimeter,
                                    new_region.perimeter / new_region.area,
                                    np.mean(background_colors) / max(colors),
                                    new_region.eccentricity,
                                    # max(colors),
                                    # np.mean(colors),
                                    # max(background_colors),
                                    # np.mean(background_colors),
                                ]
                            ]
                        )
                        == 1
                    ):
                        output_coords.append(new_region)

            if new_region.eccentricity >= 0.8:
                distance = ndi.distance_transform_edt(remove_holes)
                coords = peak_local_max(
                    distance,
                    footprint=np.ones((3, 3)),
                    labels=remove_holes,
                    min_distance=int(min_diameter_threshold_pixels),
                )
                mask = np.zeros(distance.shape, dtype=bool)
                mask[tuple(coords.T)] = True
                markers, _ = ndi.label(mask)
                labels = watershed(-distance, markers, mask=remove_holes)
                # plt.imshow(labels);

                separated_regions = regionprops(labels)
                for separated_region in separated_regions:
                    if separated_region.area >= min_area_threshold_pixels:
                        colors = []
                        coords = (
                            separated_region.coords
                        )  # Coordinates of the current region
                        for coord in coords:
                            x, y = coord
                            colors.append(blurred_normalized[x][y])
                        # Find background color
                        square_coordinates = get_square_coordinates(
                            separated_region.bbox[0],
                            separated_region.bbox[1],
                            separated_region.bbox[2],
                            separated_region.bbox[3],
                        )
                        coordinates = np.array(
                            [
                                coord
                                for coord in square_coordinates
                                if not np.any(np.all(coord == coords, axis=1))
                            ]
                        )
                        background_colors = []
                        for coord in coordinates:
                            x, y = coord
                            background_colors.append(blurred_normalized[x][y])
                        if (
                            log_reg.predict(
                                [
                                    [
                                        # separated_region.area,
                                        # separated_region.perimeter,
                                        separated_region.perimeter
                                        / separated_region.area,
                                        np.mean(background_colors) / max(colors),
                                        separated_region.eccentricity,
                                        # max(colors),
                                        # np.mean(colors),
                                        # max(background_colors),
                                        # np.mean(background_colors),
                                    ]
                                ]
                            )
                            == 1
                        ):
                            output_coords.append(separated_region)
        time.sleep(0.1)

    return output_coords


def calculate_roi_area(roi, ratio):
    # Ensure the input array has the correct shape
    if roi.shape[0] != 1 or roi.shape[2] != 2:
        raise ValueError("Input array should have shape (1, n, 2)")

    # Extract the coordinates from the array
    x_coords = roi[0, :, 0]
    y_coords = roi[0, :, 1]

    # Apply the Shoelace formula to calculate the area
    area = 0.5 * np.abs(
        np.dot(x_coords, np.roll(y_coords, 1)) - np.dot(y_coords, np.roll(x_coords, 1))
    )
    converted_area = area * (ratio**2)
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
            print("Analyzing image " + str(n))
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
                print("The whole image will be used as a ROI")
                dimensions = list(layers.values())[0].shape
                custom_values = np.array(
                    [
                        [
                            [0, 0],
                            [0, dimensions[1]],
                            [dimensions[0], dimensions[1]],
                            [dimensions[0], 0],
                        ]
                    ],
                    dtype=np.int32,
                )
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

    n = 0
    for key, value in dict_of_binary.items():
        n += 1
        print(
            "Analyzing image "
            + str(n)
            + "/"
            + str(len(dict_of_binary))
            + ":"
            + str(key)
        )
        binary_image = ~value[0]
        blurred_normalized = value[4]
        output_coords = apply_watershed_cfos(
            binary_image, blurred_normalized, ratio, hyperparameters
        )

        # name = key + '_watershed.tif'
        # file_path = os.path.join(directory, name)
        # plt.imshow(output_coords)
        # plt.savefig(file_path, dpi=300)
        # plt.close()

        print(f"Number of Cells - {len(output_coords)}")
        roi_area = calculate_roi_area(value[1], ratio)
        # print(f"{key[:-4]}: ROI Area in µm^2 - {roi_area}")
        cells_mm_2 = ((10**6) * len(output_coords)) / roi_area
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
        layer_roi = np.where(mask == 255, value[3], 0)  # .astype(int)
        rows_to_keep = np.any(layer_roi, axis=1)
        cols_to_keep = np.any(layer_roi, axis=0)
        to_plot = value[3]  # [rows_to_keep][:, cols_to_keep]
        axes[0].imshow(to_plot, cmap="grey")
        axes[0].set_title("Original figure")
        roi_polygon = Polygon(
            value[1].squeeze(), edgecolor="red", linewidth=1, facecolor="none"
        )
        axes[0].add_patch(roi_polygon)
        # Hide the axis ticks and values
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].set_xticklabels([])
        axes[0].set_yticklabels([])

        # ======== 2nd panel: threshold (elbow)
        hist, bins = np.histogram(value[4][value[4] != 0], bins=64, range=(1, 256))
        peaks, _ = find_peaks(hist)
        closest_peak_index = np.argmax(hist[peaks])
        subset_hist = hist[peaks[closest_peak_index] :]
        subset_bins = bins[peaks[closest_peak_index] : -1]
        knee = KneeLocator(
            subset_bins, subset_hist, curve="convex", direction="decreasing"
        )
        elbow_value = knee.elbow
        axes[1].plot(bins[:-1], hist)
        axes[1].axvline(
            x=bins[peaks[closest_peak_index]],
            color="g",
            linestyle="--",
            label=f"Closest Peak to 255: {bins[peaks[closest_peak_index]]:.2f}",
        )
        axes[1].axvline(
            x=elbow_value,
            color="b",
            linestyle="--",
            label=f"Threshold: {elbow_value:.2f}",
        )
        axes[1].set_title("Threshold")
        axes[1].set_xlabel(
            "Pixel Value"
        )  # Corrected from axes[1].xlabel('Pixel Value')
        axes[1].set_ylabel("Frequency")  # Corrected from axes[1].ylabel('Frequency')
        axes[1].legend()

        # ======== 3rd panel: identified cells
        number_list = [x for x in range(len(output_coords))]
        number_list.remove(0)
        number_list = random.choices(number_list, k=len(number_list))

        artificial_binary = np.zeros(binary_image.shape)
        for coords, number in zip(output_coords, number_list):
            artificial_binary[coords.coords[:, 0], coords.coords[:, 1]] = number

        # artificial_binary = artificial_binary[rows_to_keep][:, cols_to_keep]
        artificial_binary[artificial_binary == 0] = np.nan

        axes[2].set_facecolor("black")
        axes[2].imshow(artificial_binary, cmap="gist_rainbow")

        axes[2].set_title("Identified cells")
        # Hide the axis ticks and values
        axes[2].set_xticks([])
        axes[2].set_yticks([])
        axes[2].set_xticklabels([])
        axes[2].set_yticklabels([])

        # Adjust layout to prevent overlapping
        plt.tight_layout()

        # Save the image as a JPEG file
        name = key + "_" + layer + ".jpg"
        file_path = os.path.join(directory, name)
        plt.savefig(file_path)
        plt.close()

    # Creating a DataFrame
    output_dict = {
        "file_name": file_name,
        "background_threshold": background_threshold,
        "num_cells": num_cells,
        "roi_area": roi_surface,
        "cells_per_squared_mm": cells_per_squared_mm,
    }
    df = pd.DataFrame(output_dict)
    df_path = os.path.join(directory, "results.csv")
    df.to_csv(df_path, index=False)

    return artificial_binary


# =============================================================================
# Genetic Algorithm for hyperparameter optimization
# =============================================================================


def evaluate(dict_of_binary, ratio, hyperparameters, actual_values, loss_metric):

    predicted_values = {}

    for key in dict_of_binary.keys():
        value = dict_of_binary[key]
        binary_image = ~value[0]
        blurred_normalized = value[4]
        cells = apply_watershed_cfos(
            binary_image, blurred_normalized, ratio, hyperparameters
        )
        predicted_values[key] = len(cells)

    common_keys = set(actual_values.keys()) & set(predicted_values.keys())

    # Calculate RMSE
    if loss_metric == "rmse":
        squared_diff = [
            (actual_values[key][0] - predicted_values[key]) ** 2 for key in common_keys
        ]
        mse = sum(squared_diff) / len(common_keys)
        rmse = math.sqrt(mse)
        loss = rmse

    # Calculate MAE
    elif loss_metric == "mae":
        absolute_diff = [
            abs(actual_values[key][0] - predicted_values[key]) for key in common_keys
        ]
        mae = sum(absolute_diff) / len(common_keys)
        loss = mae

    # Calculate mean z-score
    elif loss_metric == "zscore":
        zscores = [
            abs(actual_values[key][0] - predicted_values[key]) / actual_values[key][1]
            for key in common_keys
        ]
        zscore_mean = sum(zscores) / len(zscores)
        loss = zscore_mean

    return loss, predicted_values


def initialize_population(population_size, hyperparameter_ranges):
    population = []
    for _ in range(population_size):
        individual = {}
        for hyperparameter, (min_value, max_value) in hyperparameter_ranges.items():
            # Ensure that distance_3 is aproximately half the distance_2
            # if hyperparameter == 'distance_3':
            #     min_value = individual['distance_2'] * 0.25
            #     max_value = individual['distance_2'] * 0.75

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
    mutated_individual = (
        individual.copy()
    )  # Make a copy to avoid modifying the original individual

    # Iterate through hyperparameters and apply mutation with a certain probability
    for hyperparameter, value in mutated_individual.items():
        if random.random() < mutation_rate:

            # Calculate a mutation value based on the range of the hyperparameter
            # mutation_value = random.uniform(-0.1 * (hyperparameter_ranges[hyperparameter][1] - hyperparameter_ranges[hyperparameter][0]),
            #                                 0.1 * (hyperparameter_ranges[hyperparameter][1] - hyperparameter_ranges[hyperparameter][0]))
            mutation_value = random.uniform(-0.1, 0.1)
            mutated_value = value + mutation_value
            mutated_value = max(
                hyperparameter_ranges[hyperparameter][0],
                min(hyperparameter_ranges[hyperparameter][1], mutated_value),
            )
            mutated_individual[hyperparameter] = mutated_value

    return mutated_individual


def genetic_algorithm(
    dict_of_binary,
    ratio,
    actual_values,
    loss_metric,
    num_iterations=50,
    population_size=50,
    initial_mutation_rate=0.2,
    mutation_rate_decay=0.5,
):
    hyperparameter_ranges = {
        "distance_2": (0, 2),  # %
        "color_1": (0, 2),  # %
        "distance_4": (0, 2),  # %
    }

    population = initialize_population(population_size, hyperparameter_ranges)
    best_loss = float("inf")
    best_hyperparameters = None

    no_improvement_counter = 0
    patience = 6
    loss_list = []

    # Initialize the mutation rate
    mutation_rate = initial_mutation_rate

    # Split the keys into training (80%) and validation (20%) sets
    keys = list(dict_of_binary.keys())
    random.shuffle(keys)
    split_index = int(len(keys) * 0.8)

    training_keys = keys[:split_index]
    validation_keys = keys[split_index:]

    training_set = {key: dict_of_binary[key] for key in training_keys}
    validation_set = {key: dict_of_binary[key] for key in validation_keys}

    for generation in tqdm(
        range(num_iterations), desc="Going through generations", unit="generations"
    ):
        # Evaluate fitness for each individual in the population
        fitness = []
        for individual in population:
            loss, predicted_values = evaluate(
                training_set, ratio, individual, actual_values, loss_metric
            )
            fitness.append((loss, individual, predicted_values))

        # Sort by fitness and select the top individuals (elite)
        elite_size = int(0.2 * population_size)
        fitness.sort(key=lambda x: x[0])
        top_individuals = fitness[:elite_size]

        # Update best accuracy and hyperparameters if current generation is better
        loss_list.append(top_individuals[0][0])
        if top_individuals[0][0] < best_loss:
            best_loss, best_hyperparameters, best_predicted_training = top_individuals[
                0
            ]
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1

        print(f"\nBest loss = {best_loss}")

        # Early stopping check
        if no_improvement_counter >= patience:
            print(
                f"\nStopping early as there is no improvement for {patience} iterations.\n"
            )
            _, best_predicted_validation = evaluate(
                validation_set, ratio, best_hyperparameters, actual_values, loss_metric
            )
            break

        # Create the next generation through crossover and mutation
        new_population = [individual for (_, individual, _) in top_individuals]
        while len(new_population) < population_size:
            parent1, parent2 = random.choices(top_individuals, k=2)
            child = crossover(parent1[1], parent2[1])
            mutated_child = mutate(child, hyperparameter_ranges, mutation_rate)
            new_population.append(mutated_child)

        population = new_population

        # Decay the mutation rate
        # mutation_rate *= mutation_rate_decay
        # if mutation_rate < 0.2:
        #     mutation_rate = 0.2

        time.sleep(0.1)

    return (
        best_hyperparameters,
        best_predicted_training,
        best_predicted_validation,
        loss_list,
    )


# =============================================================================
# Master functions
# =============================================================================


def train_model(dict_rois, actual_values, layer, ratio, loss_metric):
    dict_of_binary = create_dict_of_binary(dict_rois, layer)
    best_loss, best_hyperparameters, best_predicted_values, loss_list = (
        genetic_algorithm(dict_of_binary, ratio, actual_values, loss_metric)
    )

    return best_loss, best_hyperparameters, best_predicted_values, loss_list


def plot_correlation(actual_values, best_predicted_values):

    # Extract keys and values
    keys = set(actual_values.keys()) & set(best_predicted_values.keys())
    values1 = [actual_values[key][0] for key in keys]
    values2 = [best_predicted_values[key] for key in keys]

    # Plot
    plt.scatter(values1, values2)
    plt.xlabel("Actual values")
    plt.ylabel("Predicted values")
    plt.title("Correlation Plot")
    plt.grid(True)

    # Calculate correlation coefficient
    correlation_coefficient = np.corrcoef(values1, values2)[0, 1]
    print("Correlation coefficient:", correlation_coefficient)

    # Plot correlation line
    x_values = np.array(values1)
    y_values = (
        correlation_coefficient * x_values
        + np.mean(values2)
        - correlation_coefficient * np.mean(values1)
    )
    plt.plot(
        x_values,
        y_values,
        color="red",
        label=f"Correlation line: y = {correlation_coefficient:.2f}x + {np.mean(values2) - correlation_coefficient * np.mean(values1):.2f}",
    )

    # Plot correlation value and equation
    plt.text(
        0.1,
        0.9,
        f"Correlation coefficient: {correlation_coefficient:.2f}",
        transform=plt.gca().transAxes,
    )
    plt.text(
        0.1,
        0.85,
        f"Correlation equation: y = {correlation_coefficient:.2f}x + {np.mean(values2) - correlation_coefficient * np.mean(values1):.2f}",
        transform=plt.gca().transAxes,
    )

    plt.show()


def analyze_images(dict_rois, directory, layer_dict, ratio):
    my_binaries = {}
    for layer, hyperparameters in layer_dict.items():
        print("Analyzing " + layer)
        dict_of_binary = create_dict_of_binary(dict_rois, layer)
        artifical_binary = compiler(
            directory, dict_of_binary, ratio, layer, hyperparameters
        )
        my_binaries[layer] = artifical_binary
    return my_binaries
