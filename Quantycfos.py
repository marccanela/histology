"""
Created on Thu Jul  4 15:43:26 2024
@author: mcanela
Output analysis of Quanty-cFOS
"""

import os
import time
import zipfile
import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from read_roi import read_roi_file, read_roi_zip
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import regionprops
from skimage import measure

common_tag = "//folder/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2024-02a04_TRAP2_recalling/microscopi/"
areas_folder = os.path.join(common_tag, "microscope_males/bla/areas/")
image_folder = os.path.join(common_tag, "microscope_males/bla/tdt/")
output_folder = os.path.join(common_tag, "microscope_males/bla/MacroResults_tdt/")
model_folder = os.path.join(common_tag, "test/")

cmap = "Reds"
model = "TRAP2"  # 'Immuno' or 'TRAP2'

#--------------- IMPORTING SVM MODEL ---------------------------------

with open(os.path.join(model_folder, "best_model_svm.pkl"), 'rb') as file:
    best_model = pickle.load(file)

#--------- FOR OVERLAP ONLY ------------------------------------------

rois1 = f"{common_tag}microscope_males/rsc/MacroResults_cfos/"
rois1_images = f"{common_tag}microscope_males/rsc/cfos/"
rois2 = f"{common_tag}microscope_males/rsc/MacroResults_tdt/"
rois2_images = f"{common_tag}microscope_males/rsc/tdt/"
    
#------------ PREDEFINED FUNCTIONS --------------------------------

def change_names(number=300):

    folder_path = "//folder/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2024-02a04_TRAP2_recalling/microscopi/microscopi_batch3/rsc"

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file has the expected format
        if filename.endswith(".nd2"):
            parts = filename.split("_")
            if len(parts) >= 3 and parts[1].isdigit():
                # Extract and modify the second element
                original_number = int(parts[1])
                new_number = original_number + number
                parts[1] = str(new_number)

                # Create the new filename
                new_filename = "_".join(parts)

                # Construct the full file paths
                original_file_path = os.path.join(folder_path, filename)
                new_file_path = os.path.join(folder_path, new_filename)

                # Rename the file
                os.rename(original_file_path, new_file_path)


def get_layer(image):  
    # Identify the layer which contains the information
    non_zero_layer = None
    for i in range(image.shape[2]):
        if np.any(image[:, :, i] != 0):
            non_zero_layer = image[:, :, i]
            break
    return non_zero_layer


def get_cell_mask(layer, coordinates):
    # Create a mask from the ROI coordinates
    mask = np.zeros(layer.shape, dtype=np.uint8)
    cv2.fillPoly(mask, coordinates, 1)  
    return mask     


def crop_cell(layer, x_coords, y_coords): 
    # Cut a square around the cell
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))
    layer_cropped = layer[y_min:y_max+1, x_min:x_max+1]
    return layer_cropped


def crop_cell_large(layer, x_coords, y_coords, padding=None):
    
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    if padding is None:
        x_padding = x_max - x_min
    else:   
        x_padding = padding
    x_min, x_max = x_min - x_padding, x_max + x_padding
    
    y_min, y_max = int(min(y_coords)), int(max(y_coords))
    if padding is None:
        y_padding = y_max - y_min
    else:
        y_padding = padding
    y_min, y_max = y_min - y_padding, y_max + y_padding
    
    layer_height, layer_width = layer.shape[:2]
    x_min = max(0, x_min)
    x_max = min(layer_width - 1, x_max)
    y_min = max(0, y_min)
    y_max = min(layer_height - 1, y_max)
    
    layer_cropped = layer[y_min:y_max+1, x_min:x_max+1]
    x_coords_cropped = [x - x_min for x in x_coords]
    y_coords_cropped = [y - y_min for y in y_coords]
    return layer_cropped, x_coords_cropped, y_coords_cropped


def analyze_all(filename, image_folder, cmap, mode, best_model):

    image = cv2.imread(image_folder + filename, cv2.IMREAD_UNCHANGED)
    layer = get_layer(image)

    # Import ROIs
    for folder in os.listdir(output_folder):
        if folder.endswith(filename[:-4]):
            folderpath = os.path.join(output_folder, folder)
            for file in os.listdir(folderpath):
                if file.endswith(".zip"):
                    zip_path = os.path.join(folderpath, file)
                    rois = read_roi_zip(zip_path) 
    
    keeped = {}
    for roi_name, roi_info in rois.items():
        
        X = []
        
        # Extract coordinates
        x_coords = roi_info["x"]
        y_coords = roi_info["y"]
        coordinates = np.array([list(zip(x_coords, y_coords))], dtype=np.int32)
        
        # Computing ROI intensities
        cell_mask = get_cell_mask(layer, coordinates)
        cell_pixels = layer[cell_mask == 1]         
        mean_intensity = np.mean(cell_pixels)
        median_intensity = np.median(cell_pixels)
        X.append(mean_intensity)
        X.append(median_intensity)
        X.append(np.std(cell_pixels))
        X.append(min(cell_pixels))
        X.append(max(cell_pixels))

        # Computing background intensities
        background_mask = 1 - cell_mask
        background_mask_cropped = crop_cell(background_mask, x_coords, y_coords)
        layer_cropped = crop_cell(layer, x_coords, y_coords)
        background_pixels = layer_cropped[background_mask_cropped == 1]
        mean_background = np.mean(background_pixels)
        median_background = np.median(background_pixels)
        X.append(mean_background)
        X.append(median_background)
        X.append(np.std(background_pixels))
        X.append(min(background_pixels))
        X.append(max(background_pixels))
        
        # Intensity comparisions
        X.append(mean_intensity / (mean_background if mean_background != 0 else 1))
        X.append(mean_intensity - mean_background)
        X.append(median_intensity / (median_background if median_background != 0 else 1))
        X.append(median_intensity - median_background)
     
        # Texture-based features (example with LBP and GLCM)
        layer_masked = layer * cell_mask
        layer_masked_cropped = crop_cell(layer_masked, x_coords, y_coords)
        lbp = local_binary_pattern(layer_masked_cropped, P=8, R=1, method='uniform')
        glcm = graycomatrix(layer_masked, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        X.append(np.mean(lbp))
        X.append(np.std(lbp))
        X.append(graycoprops(glcm, 'contrast')[0, 0])
        X.append(graycoprops(glcm, 'correlation')[0, 0])
        X.append(graycoprops(glcm, 'energy')[0, 0])
        X.append(graycoprops(glcm, 'homogeneity')[0, 0])
                    
        # Shape-based features
        label_img = measure.label(cell_mask)
        props = regionprops(label_img)
        if len(props) > 0:
            prop = props[0]  # assuming one object per ROI
            X.append(prop.area)
            X.append(prop.perimeter)
            X.append(prop.eccentricity)
            X.append(prop.major_axis_length)
            X.append(prop.minor_axis_length)
            X.append(prop.solidity)
            X.append(prop.extent)
        
        # Histogram of Oriented Gradients (HOG)
        hog_descriptor = cv2.HOGDescriptor()
        h = hog_descriptor.compute(layer_masked)
        X.append(np.mean(h))
        X.append(np.std(h))

        # Apply model
        X = np.array(X).reshape(1,-1)
        y = best_model.predict(X)
        
        if y == 1:
            keeped[roi_name] = roi_info

    keeped_tags = list(keeped.keys())

    # Export the .roi files
    output_rois_folder = os.path.join(folderpath, "identified_rois")
    if not os.path.exists(output_rois_folder):
        os.makedirs(output_rois_folder)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_rois_folder)
    all_files = os.listdir(output_rois_folder)
    for file_name in all_files:
        if file_name[:-4] not in keeped_tags:
            file_path = os.path.join(output_rois_folder, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

    # Export the images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # inverted_image = cv2.bitwise_not(image)
    # axes[0].imshow(cv2.cvtColor(inverted_image, cv2.COLOR_BGR2RGB))
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    axes[0].imshow(layer, cmap=cmap, vmin=0, vmax=255)
    axes[0].axis("off")  # Hide the axis

    # axes[0].imshow(image, cmap=cmap)
    # axes[0].axis('off')  # Hide the axis

    canvas = np.zeros(
        image.shape[:2], dtype=np.uint8
    )  # Create zeros with the same size as the image
    for roi_name, roi in keeped.items():
        # for roi_name, roi in filtered_rois.items():
        x_coords = roi["x"]
        y_coords = roi["y"]
        # Create the mask_cells
        coordinates = np.array([list(zip(x_coords, y_coords))], dtype=np.int32)
        cv2.fillPoly(canvas, coordinates, 255)
    axes[1].imshow(canvas, cmap=cmap)
    axes[1].axis("off")  # Hide the axis

    plt.tight_layout()

    output_image_folder = os.path.join(output_folder, "output_images")
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    name = filename[:-4] + ".png"
    file_path = os.path.join(output_image_folder, name)
    plt.savefig(file_path)
    plt.close()

    # return filtered_areas, difference_dict
    return keeped


def filter_images(image_folder):
    
    mean_intensity = []
    median_intensity = []
    sd_intensity = []
    min_intensity = []
    max_intensity = []
    
    for filename in tqdm(os.listdir(image_folder), desc="Filtering images", unit="image"):
        if filename.endswith(".tif"):
    
            image = cv2.imread(image_folder + filename, cv2.IMREAD_UNCHANGED)
            layer = get_layer(image)
            layer_values = layer[layer != 0]
            
            mean_intensity.append(np.mean(layer_values))
            median_intensity.append(np.median(layer_values))
            sd_intensity.append(np.std(layer_values))
            min_intensity.append(min(layer_values))
            max_intensity.append(max(layer_values))
    
    X = np.column_stack((mean_intensity, median_intensity, sd_intensity, min_intensity, max_intensity))


# ANALYZE CFOS AND TDTOMATO
def analyze_image(image_folder):       
    
    file_name = []
    num_cells = []
    roi_area = []
    cells_per_squared_mm = []
    for filename in tqdm(
        os.listdir(image_folder), desc="Processing images", unit="image"
    ):
        if filename.endswith(".tif"):
            
            new_filtered_rois = analyze_all(filename, image_folder, cmap, model, best_model)
            # Export the countings per area
            tag = filename[3:-4] + ".nd2.txt"
            area = float(
                pd.read_csv(os.path.join(areas_folder, tag), delimiter="\t")["Area"]
            )
            cells_mm_2 = ((10**6) * len(new_filtered_rois)) / area
            file_name.append(filename[3:-4])
            num_cells.append(len(new_filtered_rois))
            roi_area.append(area)
            cells_per_squared_mm.append(cells_mm_2)
        time.sleep(0.1)
    # Raw dataframe
    df = pd.DataFrame(
        {
            "file_name": file_name,
            "num_cells": num_cells,
            "roi_area": roi_area,
            "cells_per_squared_mm": cells_per_squared_mm,
        }
    )
    df_path = os.path.join(output_folder, "results.csv")
    df.to_csv(df_path, index=False)
    # Friendly dataframe
    # df = pd.read_csv('//folder/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2024-02a04_TRAP2_recalling/microscopi/microscope_females/bla/results_cfos/results.csv')
    split_df = df["file_name"].str.split("_", expand=True)
    df[["group", "id", "brain", "replica"]] = split_df
    grouped_df = (
        df.groupby(["group", "id", "brain"])["cells_per_squared_mm"]
        .mean()
        .reset_index()
    )
    grouped_df.columns = ["group", "id", "brain", "mean_cells_per_squared_mm"]
    grouped_df_path = os.path.join(output_folder, "results_friendly.xlsx")
    grouped_df.to_excel(grouped_df_path, index=False)


# Calculate overlap
def overlap():

    file_name = []
    num_cells = []
    roi_area = []
    cells_per_squared_mm = []

    for filename in tqdm(
        os.listdir(rois1_images), desc="Processing images", unit="image"
    ):
        if filename.endswith(".tif"):
            image = cv2.imread(rois1_images + filename, cv2.IMREAD_UNCHANGED)
            dimensions = image.shape[:2]
            tag = filename[3:-4]
            overlapped = []

            # Read ROIs for the current image from rois1
            for folder1 in os.listdir(rois1):
                if folder1.endswith(
                    tag
                ):  # find the image folder inside the results analysis
                    roi_path1 = os.path.join(rois1, folder1, "identified_rois")
                    rois1_data = []
                    for file in os.listdir(roi_path1):
                        if file.endswith(".roi"):
                            roi1 = read_roi_file(os.path.join(roi_path1, file))
                            rois1_data.append(roi1)

            # Read ROIs for the current image from rois2
            for folder2 in os.listdir(rois2):
                if folder2.endswith(
                    tag
                ):  # find the image folder inside the results analysis
                    roi_path2 = os.path.join(rois2, folder2, "identified_rois")
                    for file in os.listdir(roi_path2):
                        if file.endswith(".roi"):
                            roi2 = read_roi_file(os.path.join(roi_path2, file))

                            for roi_info2 in roi2.values():
                                x_coords2 = roi_info2["x"]
                                y_coords2 = roi_info2["y"]
                                area_roi2 = calculate_polygon_area(x_coords2, y_coords2)
                                coordinates_roi2 = np.array(
                                    [list(zip(x_coords2, y_coords2))], dtype=np.int32
                                )

                                mask_cells_roi2 = np.zeros(dimensions, dtype=np.uint8)
                                cv2.fillPoly(mask_cells_roi2, coordinates_roi2, 1)

                                for roi1 in rois1_data:
                                    for roi_info1 in roi1.values():
                                        x_coords1 = roi_info1["x"]
                                        y_coords1 = roi_info1["y"]
                                        area_roi1 = calculate_polygon_area(
                                            x_coords1, y_coords1
                                        )
                                        coordinates_roi1 = np.array(
                                            [list(zip(x_coords1, y_coords1))],
                                            dtype=np.int32,
                                        )

                                        mask_cells_roi1 = np.zeros(
                                            dimensions, dtype=np.uint8
                                        )
                                        cv2.fillPoly(
                                            mask_cells_roi1, coordinates_roi1, 1
                                        )

                                        overlap = np.logical_and(
                                            mask_cells_roi1, mask_cells_roi2
                                        )
                                        overlap_area = np.sum(overlap)
                                        smaller_roi = min(area_roi1, area_roi2)
                                        if overlap_area >= 0.8 * smaller_roi:
                                            overlapped.append(roi_info2)
                                            break  # Exit inner loop if overlap found
                                    if overlap_area >= 0.8 * smaller_roi:
                                        break  # Exit outer loop if overlap found

            # Export the images
            fig, axes = plt.subplots(1, 3, figsize=(13, 5))

            # IMAGE_1
            for file_1 in os.listdir(rois1_images):
                if file_1.endswith(tag + ".tif"):
                    image_1 = cv2.imread(rois1_images + file_1, cv2.IMREAD_UNCHANGED)
                    image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
                    axes[0].imshow(image_1, cmap="Greens")
                    axes[0].axis("off")  # Hide the axis

            # IMAGE_2
            for file_2 in os.listdir(rois2_images):
                if file_2.endswith(tag + ".tif"):
                    image_2 = cv2.imread(rois2_images + file_2, cv2.IMREAD_UNCHANGED)
                    image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
                    axes[1].imshow(image_2, cmap="Reds")
                    axes[1].axis("off")  # Hide the axis

            # IMAGE_3
            canvas = np.zeros(
                dimensions, dtype=np.uint8
            )  # Create zeros with the same size as the image
            for roi in overlapped:
                x_coords = roi["x"]
                y_coords = roi["y"]
                coordinates = np.array([list(zip(x_coords, y_coords))], dtype=np.int32)
                cv2.fillPoly(canvas, coordinates, 255)
            axes[2].imshow(canvas, cmap="Blues")
            axes[2].axis("off")  # Hide the axis

            plt.tight_layout()

            export_image_folder = os.path.join(rois1, "overlap")
            if not os.path.exists(export_image_folder):
                os.makedirs(export_image_folder)
            name = tag + ".png"
            file_path = os.path.join(export_image_folder, name)
            plt.savefig(file_path)
            plt.close()

            # Export results
            area_tag = tag + ".nd2.txt"
            area = float(
                pd.read_csv(os.path.join(areas_folder, area_tag), delimiter="\t")[
                    "Area"
                ]
            )
            cells_mm_2 = ((10**6) * len(overlapped)) / area
            file_name.append(tag)
            num_cells.append(len(overlapped))
            roi_area.append(area)
            cells_per_squared_mm.append(cells_mm_2)
        time.sleep(0.1)

    # Raw dataframe
    df = pd.DataFrame(
        {
            "file_name": file_name,
            "num_cells": num_cells,
            "roi_area": roi_area,
            "cells_per_squared_mm": cells_per_squared_mm,
        }
    )
    df_path = os.path.join(rois1, "overlap.csv")
    df.to_csv(df_path, index=False)
    # Friendly dataframe
    # df = pd.read_csv('//folder/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2024-02a04_TRAP2_recalling/microscopi/microscope_females/bla/results_cfos/results.csv')
    split_df = df["file_name"].str.split("_", expand=True)
    df[["group", "id", "brain", "replica"]] = split_df
    grouped_df = (
        df.groupby(["group", "id", "brain"])["cells_per_squared_mm"]
        .mean()
        .reset_index()
    )
    grouped_df.columns = ["group", "id", "brain", "mean_cells_per_squared_mm"]
    grouped_df_path = os.path.join(rois1, "overlap_friendly.xlsx")
    grouped_df.to_excel(grouped_df_path, index=False)





















