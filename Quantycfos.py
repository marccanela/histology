"""
Created on Thu Jul  4 15:43:26 2024
@author: mcanela
Output analysis of Quanty-cFOS
"""

from pathlib import Path
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
from skimage.measure import regionprops, label
from shapely.geometry import Polygon

common_tag = Path("//folder/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/")
areas_folder = common_tag / "Fotos BLA marc script/areas/"
image_folder = common_tag / "Fotos BLA marc script/cfos/"
output_folder = common_tag / "Fotos BLA marc script/MacroResults_cfos/"
model_folder = common_tag / '1) SOC/2024-02a04_TRAP2_recalling/microscopi/test/tdt_training_results/'

cmap = "Greens"

#--------------- IMPORTING SVM MODEL ---------------------------------

with open(model_folder / "best_model_svm.pkl", 'rb') as file:
    best_model = pickle.load(file)

best_model = None

#--------- FOR OVERLAP ONLY ------------------------------------------

rois1 = common_tag / "Fotos BLA marc script/MacroResults_cfos_1.1/"
rois1_images = common_tag / "Fotos BLA marc script/cfos/"
rois2 = common_tag / "Fotos BLA marc script/MacroResults_tdt/"
rois2_images = common_tag / "Fotos BLA marc script/tdt/"

#------------ PREDEFINED FUNCTIONS --------------------------------

def change_names(number: int = 300):
    
    folder_path = Path("//folder/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2024-02a04_TRAP2_recalling/microscopi/microscopi_batch3/rsc")

    # Iterate over all files in the folder
    for filename in folder_path.iterdir():
        if filename.is_file() and filename.suffix == ".nd2":
            parts = filename.stem.split("_")
            if len(parts) >= 3 and parts[1].isdigit():
                try:
                    # Extract and modify the second element
                    original_number = int(parts[1])
                    new_number = original_number + number
                    parts[1] = str(new_number)

                    # Create the new filename
                    new_filename = "_".join(parts) + filename.suffix

                    # Construct the full file paths
                    new_file_path = filename.with_name(new_filename)

                    # Rename the file
                    filename.rename(new_file_path)
                    
                except Exception as e:
                    print(f"Error processing {filename.name}: {e}")


def get_layer(image):
    """
    Extracts the first non-empty layer from a 3D image array.    
    This function uses vectorized operations to find the first layer
    that contains any non-zero values and returns it. If all layers are empty
    (i.e., all values are zero), the function returns `None`.
    
    Parameters:
    ----------
    image : numpy.ndarray
        A 3D NumPy array representing the image. The shape of the array
        should be (height, width, num_layers).
    
    Returns:
    -------
    numpy.ndarray or None
        The first non-empty layer of the image if such a layer exists,
        otherwise `None`. The returned layer is a 2D NumPy array with
        shape (height, width).
    """
    layer_sums = np.sum(image, axis=(0, 1))
    non_zero_layer_indices = np.nonzero(layer_sums)[0]
    
    if non_zero_layer_indices.size > 0:
        return image[:, :, non_zero_layer_indices[0]]
    else:
        return None


def get_cell_mask(layer, coordinates):
    '''
    Create a binary mask from the given Region of Interest (ROI) coordinates.
    This function creates a binary mask where the specified ROI is filled with
    ones (1) and the rest of the mask is zeros (0). The mask is created
    based on the coordinates provided, which define the polygonal region
    to be filled.

    Parameters:
    ----------
    layer : numpy.ndarray
        A 2D NumPy array representing the image layer. The shape of the array
        (height, width) determines the dimensions of the resulting mask.

    coordinates : list of numpy.ndarray
        A list where each element is a NumPy array of shape (N, 2)
        representing the vertices of a polygon. These coordinates specify
        the regions to be filled in the mask.

    Returns:
    -------
    numpy.ndarray
        A binary mask of the same shape as `layer`. Pixels inside the defined
        regions are set to `1`, and all other pixels are set to `0`.
    '''
    mask = np.zeros(layer.shape, dtype=np.uint8)
    cv2.fillPoly(mask, coordinates, 1)  
    return mask     


def crop_cell(layer, x_coords, y_coords):
    '''
    Crop a rectangular region from an image layer based on given coordinates.
    This function extracts a rectangular subregion from the provided image
    `layer`. The rectangle is defined by the minimum and maximum x and y
    coordinates. The subregion is extracted by slicing the `layer` array.

    Parameters:
    ----------
    layer : numpy.ndarray
        A 2D NumPy array representing the image layer from which the region is
        to be cropped. The shape of the array should be (height, width).

    x_coords : list or numpy.ndarray
        A list or array of x-coordinates defining the boundary of the
        rectangular region to crop. These coordinates determine the horizontal
        extent of the region.

    y_coords : list or numpy.ndarray
        A list or array of y-coordinates defining the boundary of the 
        rectangular region to crop. These coordinates determine the vertical 
        extent of the region.

    Returns:
    -------
    numpy.ndarray
        A 2D NumPy array representing the cropped region of the image layer.
    '''
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))
    return layer[y_min:y_max+1, x_min:x_max+1]


def filter_roi(layer, rois, cell_background_threshold=None):
    '''
    Filter Regions of Interest (ROIs) based on their characteristics and
    background ratios. This function evaluates and filters ROIs based on
    their size and the ratio of cell pixel values to background pixel values.
    ROIs are initially pre-selected based on their area, and then further
    filtered based on a ratio threshold if specified.

    Parameters:
    ----------
    layer : numpy.ndarray
        A 2D NumPy array representing the image layer from which ROIs are
        extracted. The array should be of shape (height, width).

    rois : dict
        A dictionary where keys are ROI names and values are dictionaries with, 
        at least, the following keys:
        - "x": A list or array of x-coordinates of the ROI vertices.
        - "y": A list or array of y-coordinates of the ROI vertices.

    cell_background_threshold : float, optional
        A threshold ratio for the cell-to-background pixel mean value. Only 
        ROIs with a mean ratio above this threshold are kept. If None, all 
        pre-selected ROIs are returned.

    Returns:
    -------
    dict
        A dictionary where keys are ROI names and values are dictionaries 
        containing ROI information.

    '''
    first_keeped = {}
    cell_background_ratios = {}
    final_keeped = {}
    
    for roi_name, roi_info in rois.items():
        x_coords, y_coords = roi_info["x"], roi_info["y"]
        coordinates = np.array([list(zip(x_coords, y_coords))], dtype=np.int32)
        cell_mask = get_cell_mask(layer, coordinates)
        
        # Pre-select ROIs based on area
        prop = regionprops(label(cell_mask))[0]
        my_area = prop.area
        if my_area < 1000:
            
            # Crop to the bounding box of the ROI
            cell_pixels = layer[cell_mask == 1]
            layer_cropped = crop_cell(layer, x_coords, y_coords)
            background_mask_cropped = crop_cell(1 - cell_mask, x_coords, y_coords)
            background_pixels = layer_cropped[background_mask_cropped == 1]
            
            # Calculate the mean ratio
            cell_pixels_mean = np.mean(cell_pixels)
            background_pixels_mean = np.mean(background_pixels)
            mean_ratio = cell_pixels_mean / (background_pixels_mean if background_pixels_mean != 0 else 1)
            if mean_ratio > 0:
                first_keeped[roi_name] = roi_info
                cell_background_ratios[roi_name] = mean_ratio
                
    if cell_background_threshold == None:
        return first_keeped
    else:
        for roi_name, mean_ratio in cell_background_ratios.items():
            if mean_ratio >= cell_background_threshold:
                final_keeped[roi_name] = first_keeped[roi_name]
        return final_keeped
        

def analyze_roi(layer, roi_name, roi_info, best_model):
    '''
    This function processes a given ROI within an image layer by calculating
    several statistical and texture features, and then uses a pretrained model
    to determine if the ROI meets certain criteria.

    Parameters:
    ----------
    layer : numpy.ndarray
        A 2D NumPy array representing the image layer from which the ROI is
        extracted. The shape of the array should be (height, width).

    roi_name : str
        The name or identifier of the ROI being analyzed.

    roi_info : dict
        A dictionary containing the coordinates of the ROI. It should have, at
        least, the following keys:
        - "x": A list or array of x-coordinates of the ROI vertices.
        - "y": A list or array of y-coordinates of the ROI vertices.

    best_model : object
        A trained model with a `predict` method that takes a feature array as 
        input and returns a prediction.

    Returns:
    -------
    str or None
        The ROI name if the `best_model` predicts that the ROI meets the
        criteria (label == 1); otherwise, `None`.
    '''
    x_coords, y_coords = roi_info["x"], roi_info["y"]
    coordinates = np.array([list(zip(x_coords, y_coords))], dtype=np.int32)
    cell_mask = get_cell_mask(layer, coordinates)
    
    prop = regionprops(label(cell_mask))[0]
      
    # Exclude large ROIs
    my_area = prop.area
    if my_area > 1000:
        return None    
    
    # Crop the bounding box of the ROI
    cell_pixels = layer[cell_mask == 1]
    layer_cropped = crop_cell(layer, x_coords, y_coords)
    background_mask_cropped = crop_cell(1 - cell_mask, x_coords, y_coords)
    background_pixels = layer_cropped[background_mask_cropped == 1]
    
    # Compute statistics
    cell_pixels_mean = np.mean(cell_pixels)
    cell_pixels_median = np.median(cell_pixels)
    background_pixels_mean = np.mean(background_pixels)
    background_pixels_median = np.median(background_pixels)
    
    # Pre-select the ROIs based on mean difference
    mean_dif = cell_pixels_mean - background_pixels_mean
    if mean_dif < 0:
        return None
    
    # Pre-select the ROIs based on median difference
    median_dif = cell_pixels_median - background_pixels_median
    if median_dif < 0:
        return None
    
    # Calculate additional statistics and texture features
    cell_pixels_std = np.std(cell_pixels)
    cell_pixels_min = np.min(cell_pixels)
    cell_pixels_max = np.max(cell_pixels)
    background_pixels_std = np.std(background_pixels)
    background_pixels_min = np.min(background_pixels)
    background_pixels_max = np.max(background_pixels)
    
    features = np.array([
        cell_pixels_mean, cell_pixels_median, cell_pixels_std, cell_pixels_min, cell_pixels_max,
        background_pixels_mean, background_pixels_median, background_pixels_std, background_pixels_min, background_pixels_max,
        cell_pixels_mean / (background_pixels_mean if background_pixels_mean != 0 else 1),
        mean_dif,
        cell_pixels_median / (background_pixels_median if background_pixels_median != 0 else 1),
        median_dif
    ])
    
    # Calculate LBP and GLCM features
    layer_masked = layer * cell_mask
    layer_masked_cropped = crop_cell(layer_masked, x_coords, y_coords)
    lbp = local_binary_pattern(layer_masked_cropped, P=8, R=1, method='uniform')
    glcm = graycomatrix(layer_masked, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    glcm_props = [graycoprops(glcm, prop)[0, 0] for prop in ('contrast', 'correlation', 'energy', 'homogeneity')]
    
    # Combine features
    features = np.concatenate([
        features,
        [np.mean(lbp), np.std(lbp)],
        glcm_props,
        [my_area, prop.perimeter, prop.eccentricity, prop.major_axis_length, prop.minor_axis_length, prop.solidity, prop.extent]
    ])
    
    # HOG features
    hog_descriptor = cv2.HOGDescriptor()
    h = hog_descriptor.compute(layer_masked).flatten()  # Flatten the HOG descriptor
    features = np.concatenate([features, [np.mean(h), np.std(h)]])

    return roi_name if best_model.predict([features]) == 1 else None


def analyze_image(tif_path, cmap, best_model, output_folder, root):
    '''
    Analyze an image, extract and process ROIs, and visualize results..

    Parameters:
    ----------
    tiff_path : Path
        The name of the image file Path to be analyzed.

    cmap : matplotlib.colors.Colormap
        The colormap to be used for visualization of the image and ROIs.

    best_model : object or None
        A trained model with a `predict` method, used to classify ROIs. 
        If `None`, the function uses a default filtering approach.

    output_folder : str
        The path to the folder where the ROIs are stored and images will be saved.
    
    root : str
        Exclusive identifier of the image.
        
    Returns:
    -------
    dict
        A dictionary of ROIs that are considered positive according to the
        model or filtering criteria. The dictionary keys are ROI names, and
        the values are dictionaries containing the ROI information.
    '''
    # Load the image and layer
    image = cv2.imread(str(tif_path), cv2.IMREAD_UNCHANGED)
    layer = get_layer(image)
        
    # Find the correct zip file path
    zip_path = None
    for folder in output_folder.iterdir(): # Folder is a Path
        if folder.stem.endswith(root) and folder.is_dir():
            for zip_path in folder.iterdir():
                if zip_path.suffix == '.zip':
                    rois = read_roi_zip(str(zip_path))
                    break
            if zip_path:
                break
    
    if zip_path is None:
        raise FileNotFoundError("No ROI zip file found for the given filename.")
    
    # Process ROIs
    if best_model:       
        results = {roi_name: analyze_roi(layer, roi_name, roi_info, best_model) 
                   for roi_name, roi_info in tqdm(rois.items(), desc="Processing ROIs", unit="ROI")}
        keeped = {result: rois[result] for result in results.values() if result}
        
    else:
        keeped = filter_roi(layer, rois, 1.1)
    
    keeped_tags = set(keeped.keys())
    
    # Manage the output folder for identified ROIs
    output_rois_folder = folder / "identified_rois"
    output_rois_folder.mkdir(parents=True, exist_ok=True)
    
    # Extract and clean up ROIs
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_rois_folder)
    
    for file_path in output_rois_folder.iterdir():
        if file_path.stem not in keeped_tags:
            file_path.unlink()  # This removes the file
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axes[0].imshow(layer, cmap=cmap, vmin=0, vmax=255)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # ROIs
    canvas = np.zeros(layer.shape, dtype=np.uint8)
    for roi in keeped.values():
        coordinates = np.array([list(zip(roi["x"], roi["y"]))], dtype=np.int32)
        cv2.fillPoly(canvas, coordinates, 255)
    axes[1].imshow(canvas, cmap=cmap)
    axes[1].set_title("Identified ROIs")
    axes[1].axis("off")
    
    plt.tight_layout()
    output_images_folder = output_folder / "output_images"
    output_images_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_images_folder / f"{root}.png")
    plt.close()

    return keeped


# ANALYZE CFOS AND TDTOMATO
def analyze_folder(image_folder, cmap, best_model, output_folder, areas_folder):
    '''
    Processes each TIFF image in the specified folder by:
    1. Identifying positive Regions of Interest (ROIs) using `analyze_image`.
    2. Calculating the number of ROIs, the total area, and the density of
    cells per square millimeter.
    3. Aggregating and saving the results into CSV.

    Parameters:
    ----------
    image_folder : str
        The path to the folder containing TIFF images and ROI area files.

    cmap : matplotlib.colors.Colormap
        The colormap to be used for visualization of images.

    best_model : object or None
        A trained model with a `predict` method, used to classify ROIs. If
        `None`, the function uses a default filtering approach.

    output_folder : str
        The path to the folder where there are the ROIs and results will be saved.

    areas_folder : str
        The path to the folder containing area files in `.txt` format.
        
    Returns:
    -------
    None
        This function does not return a value but saves the results to CSV.

    Notes:
    -----
    - The function assumes that each TIFF image file has a corresponding area
    file with a `.txt` extension.
    - The area file should be located in `areas_folder` and should have a
    tab-delimited format with a column named "Area".
    - The results are saved as a CSV file named "results.csv" in the `output_folder`.

    '''
    results = []
    tif_paths = list(image_folder.glob("*.tif"))
    
    for tif_path in tqdm(tif_paths, desc="Processing images", unit="image"):
        try:
            root = tif_path.stem[3:]
            
            # Analyze the image and get ROIs
            keeped = analyze_image(tif_path, image_folder, cmap, best_model, output_folder, root)
            
            # Find the corresponding area file
            area_file = root + ".nd2.txt"
            area_file_path = areas_folder / area_file
        
            if area_file_path.is_file():
                area_data = pd.read_csv(area_file_path, delimiter="\t")
                if 'Area' in area_data.columns:
                    area = area_data["Area"].astype(float).values[0]
                else:
                    raise ValueError(f"Column 'Area' not found in {root}.")
            else:
                raise FileNotFoundError(f"Area file {root} not found.")
    
            final_count = len(keeped)
            cells_mm_2 = (10**6 * final_count) / area
            results.append((root, final_count, area, cells_mm_2))
    
        except Exception as e:
            print(f"Error processing {root}: {e}")
    
    df = pd.DataFrame(results, columns=["file_name", "num_cells", "roi_area", "cells_per_squared_mm"])
    df.to_csv(output_folder / "results.csv", index=False)


# Calculate overlap
def overlap(rois1, rois1_images, rois2, rois2_images, areas_folder):
    """
    This function processes TIFF images from two sets of identified ROIs, 
    compares them to find overlaps, and exports the results as images and
    CSV file. The overlap is determined based on an 80% area overlap
    criterion.

    Parameters:
    ----------
    rois1 : str
        Path to the first folder containing subfolders with identified ROIs.
    
    rois1_images : str
        Path to the folder containing images corresponding to the first set of ROIs.
    
    rois2 : str
        Path to the second folder containing subfolders with identified ROIs.
    
    rois2_images : str
        Path to the folder containing images corresponding to the second set of ROIs.
    
    areas_folder : str
        Path to the folder containing area information files for the images.

    Returns:
    -------
    None
        This function does not return a value but saves the overlapping ROIs 
        results as images and data files.

    Notes:
    -----
    - The function assumes that each TIFF image file in `rois1_images` and 
    `rois2_images` has a corresponding area file
      with a `.txt` extension in `areas_folder`.
    - The ROI data is expected to be in `.roi` format.
    - The overlap images and results are saved in the "overlap" subfolder 
    within `rois1`.
    """
    file_name = []
    num_cells = []
    roi_area = []
    cells_per_squared_mm = []

    for tif_path  in tqdm(list(rois1_images.glob("*.tif")), desc="Processing images", unit="image"):
        
        # Select an image from rois1_images
        tag = tif_path.stem[3:]
        overlapped = {}
        
        # Find the corresponding ROIs folders
        rois1_folder = next(folder for folder in rois1.iterdir() if folder.name.endswith(tag))
        rois1_path = rois1_folder / "identified_rois"
        rois2_folder = next(folder for folder in rois2.iterdir() if folder.name.endswith(tag))
        rois2_path = rois2_folder / "identified_rois"
        
        # Load and index ROIs from rois2 for quick lookup
        rois2_indexed = {}
        for roi2_path in rois2_path.glob("*.roi"):
            roi2_data = read_roi_file(str(roi2_path))
            for roi2_info in roi2_data.values():
                x_coords2 = roi2_info["x"]
                y_coords2 = roi2_info["y"]
                polygon2 = Polygon(zip(x_coords2, y_coords2))
                rois2_indexed[roi2_path.name] = polygon2
        
        # Process ROIs from rois1 and compare with indexed ROIs from rois2
        for roi1_path in rois1_path.glob("*.roi"):
            roi1_data = read_roi_file(str(roi1_path))
            for roi1_info in roi1_data.values():
                x_coords1 = roi1_info["x"]
                y_coords1 = roi1_info["y"]
                polygon1 = Polygon(zip(x_coords1, y_coords1))
                found_overlap = False
                
                # Compare with each ROI from rois2
                for roi2_name, polygon2 in rois2_indexed.items():
                    intersection = polygon1.intersection(polygon2)
                    intersection_area = intersection.area
        
                    if intersection_area > 0:
                        area_roi1 = polygon1.area
                        area_roi2 = polygon2.area
                        smaller_roi = min(area_roi1, area_roi2)
                        if intersection_area >= 0.8 * smaller_roi:
                            overlapped.update(read_roi_file(str(rois2_path / roi2_name)))
                            found_overlap = True
                            break
                if found_overlap:
                    break
        
        # Plot the results
        fig, axes = plt.subplots(1, 3, figsize=(13, 5))
        
        image1_path = next(image for image in rois1_images.glob(f"*{tag}.tif"))
        image_1 = cv2.imread(str(image1_path), cv2.IMREAD_UNCHANGED)
        image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
        axes[0].imshow(image_1, cmap="Greens")
        axes[0].axis("off")  # Hide the axis
        
        image2_path = next(image for image in rois2_images.glob(f"*{tag}.tif"))
        image_2 = cv2.imread(str(image2_path), cv2.IMREAD_UNCHANGED)
        image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
        axes[1].imshow(image_2, cmap="Reds")
        axes[1].axis("off")  # Hide the axis

        canvas = np.zeros(image_1.shape[:2], dtype=np.uint8)  # Create zeros with the same size as the image
        for roi_info in overlapped.values():
            coordinates = np.array([list(zip(roi_info['x'], roi_info['y']))], dtype=np.int32)
            cv2.fillPoly(canvas, coordinates, 255)
        axes[2].imshow(canvas, cmap="Blues")
        axes[2].axis("off")  # Hide the axis
            
        plt.tight_layout()
        
        export_image_folder = rois1 / "overlap"
        export_image_folder.mkdir(parents=True, exist_ok=True)
        plt.savefig(export_image_folder / f"{tag}.png")
        plt.close()

        # Export the numerical results
        area_file_path = areas_folder / f"{tag}.nd2.txt"
        if area_file_path.is_file():
            area_data = pd.read_csv(area_file_path, delimiter="\t")
            if 'Area' in area_data.columns:
                area = area_data["Area"].astype(float).values[0]
            else:
                raise ValueError(f"Column 'Area' not found in {tag}.")
        else:
            raise FileNotFoundError(f"Area file {tag} not found.")

        cells_mm_2 = ((10**6) * len(overlapped)) / area
        file_name.append(tag)
        num_cells.append(len(overlapped))
        roi_area.append(area)
        cells_per_squared_mm.append(cells_mm_2)

    # Raw dataframe
    df = pd.DataFrame(
        {
            "file_name": file_name,
            "num_cells": num_cells,
            "roi_area": roi_area,
            "cells_per_squared_mm": cells_per_squared_mm,
        }
    )
    df.to_csv(rois1 / "overlap.csv", index=False)


# Calculate overlap
def old_overlap(rois1, rois1_images, rois2, rois2_images, areas_folder):
    """
    This function processes TIFF images from two sets of identified ROIs, 
    compares them to find overlaps, and exports the results as images and
    CSV file. The overlap is determined based on an 80% area overlap
    criterion.

    Parameters:
    ----------
    rois1 : str
        Path to the first folder containing subfolders with identified ROIs.
    
    rois1_images : str
        Path to the folder containing images corresponding to the first set of ROIs.
    
    rois2 : str
        Path to the second folder containing subfolders with identified ROIs.
    
    rois2_images : str
        Path to the folder containing images corresponding to the second set of ROIs.
    
    areas_folder : str
        Path to the folder containing area information files for the images.

    Returns:
    -------
    None
        This function does not return a value but saves the overlapping ROIs 
        results as images and data files.

    Notes:
    -----
    - The function assumes that each TIFF image file in `rois1_images` and 
    `rois2_images` has a corresponding area file
      with a `.txt` extension in `areas_folder`.
    - The ROI data is expected to be in `.roi` format.
    - The overlap images and results are saved in the "overlap" subfolder 
    within `rois1`.
    """
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







df = pd.read_csv("//folder/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/Fotos BLA marc script/MacroResults_cfos_1.1/overlap.csv")
split_df = df["file_name"].str.split("_", expand=True)
df["id"] = split_df[0]
grouped_df = df.groupby("id")["num_cells"].mean().reset_index()
grouped_df.to_excel(output_folder / "results_overlap_friendly.xlsx", index=False)


# split_df = df["file_name"].str.split("_", expand=True)
# df[["group", "id", "brain", "replica"]] = split_df
# grouped_df = df.groupby(["group", "id", "brain"])["cells_per_squared_mm"].mean().reset_index()
# grouped_df.columns = ["group", "id", "brain", "mean_cells_per_squared_mm"]
# grouped_df.to_excel(os.path.join(output_folder, "results_friendly.xlsx"), index=False)















