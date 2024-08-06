"""
Created on Thu Jul 25 13:07:53 2024
@author: mcanela
TRAINING AN SVM TO CLASSIFY CELLS
"""

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from read_roi import read_roi_file, read_roi_zip

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import regionprops
from skimage import measure
from scipy import ndimage as ndi

test_folder = "//folder/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2024-02a04_TRAP2_recalling/microscopi/test"
image_folder = f"{test_folder}/tdt"
quanty_folder = f"{test_folder}/MacroResults_tdt"

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


mean_intensity = []
median_intensity = []
sd_intensity = []
min_intensity = []
max_intensity = []

mean_background = []
median_background = []
sd_background = []
min_background = []
max_background = []

lbp_mean = []
lbp_std = []
contrast = []
correlation = []
energy = []
homogeneity = []

area = []
perimeter = []
eccentricity = []
major_axis_length = []
minor_axis_length = []
solidity = []
extent = []

hog_mean = []
hog_std = []

label = []


for filename in os.listdir(image_folder):
    if filename.endswith(".tif"):
        name = filename[:-4]
        
        # Open the image
        image = cv2.imread(f'{image_folder}/{filename}', cv2.IMREAD_UNCHANGED)
        layer = get_layer(image)
        
        # Import ROI
        for folder in os.listdir(quanty_folder):
            if folder.endswith(name):
                folderpath = os.path.join(quanty_folder, folder)
                for file in os.listdir(folderpath):
                    if file.endswith(".zip"):
                        zip_path = os.path.join(folderpath, file)
                        rois = read_roi_zip(zip_path)
        
        # Extract information of the ROI
        for roi_name, roi_info in rois.items():
            
            # Extract coordinates
            x_coords = roi_info["x"]
            y_coords = roi_info["y"]
            coordinates = np.array([list(zip(x_coords, y_coords))], dtype=np.int32)
            
            # Computing ROI intensities
            cell_mask = get_cell_mask(layer, coordinates)
            cell_pixels = layer[cell_mask == 1]         
            mean_intensity.append(np.mean(cell_pixels))
            median_intensity.append(np.median(cell_pixels))
            sd_intensity.append(np.std(cell_pixels))
            min_intensity.append(min(cell_pixels))
            max_intensity.append(max(cell_pixels))

            # Computing background intensities
            background_mask = 1 - cell_mask
            background_mask_cropped = crop_cell(background_mask, x_coords, y_coords)
            layer_cropped = crop_cell(layer, x_coords, y_coords)
            background_pixels = layer_cropped[background_mask_cropped == 1]
            mean_background.append(np.mean(background_pixels))
            median_background.append(np.median(background_pixels))
            sd_background.append(np.std(background_pixels))
            min_background.append(min(background_pixels))
            max_background.append(max(background_pixels))
            
            # Texture-based features (example with LBP and GLCM)
            layer_masked = layer * cell_mask
            layer_masked_cropped = crop_cell(layer_masked, x_coords, y_coords)
            lbp = local_binary_pattern(layer_masked_cropped, P=8, R=1, method='uniform')
            glcm = graycomatrix(layer_masked, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            lbp_mean.append(np.mean(lbp))
            lbp_std.append(np.std(lbp))
            contrast.append(graycoprops(glcm, 'contrast')[0, 0])
            correlation.append(graycoprops(glcm, 'correlation')[0, 0])
            energy.append(graycoprops(glcm, 'energy')[0, 0])
            homogeneity.append(graycoprops(glcm, 'homogeneity')[0, 0])
                        
            # Shape-based features
            label_img = measure.label(cell_mask)
            props = regionprops(label_img)
            if len(props) > 0:
                prop = props[0]  # assuming one object per ROI
                area.append(prop.area)
                perimeter.append(prop.perimeter)
                eccentricity.append(prop.eccentricity)
                major_axis_length.append(prop.major_axis_length)
                minor_axis_length.append(prop.minor_axis_length)
                solidity.append(prop.solidity)
                extent.append(prop.extent)
            
            # Histogram of Oriented Gradients (HOG)
            hog_descriptor = cv2.HOGDescriptor()
            h = hog_descriptor.compute(layer_masked)
            hog_mean.append(np.mean(h))
            hog_std.append(np.std(h))
            
            # Identify as cell (1) or non-cell (0)
            fig, axes = plt.subplots(1, 5, figsize=(16, 5))
            
            axes[0].imshow(layer, cmap="Reds")
            axes[0].plot(x_coords, y_coords, 'b-', linewidth=1)
            axes[0].axis("off")  # Hide the axis
            
            layer_cropped_large, x_coords_cropped, y_coords_cropped = crop_cell_large(layer, x_coords, y_coords, 500)
            axes[1].imshow(layer_cropped_large, cmap="Reds")
            axes[1].plot(x_coords_cropped, y_coords_cropped, 'b-', linewidth=1)
            axes[1].axis("off")  # Hide the axis

            axes[2].imshow(layer_cropped_large, cmap="Reds")
            axes[2].axis("off")  # Hide the axis
            
            layer_cropped_small, x_coords_cropped, y_coords_cropped = crop_cell_large(layer, x_coords, y_coords, 250)
            axes[3].imshow(layer_cropped_small, cmap="Reds")
            axes[3].plot(x_coords_cropped, y_coords_cropped, 'b-', linewidth=1)
            axes[3].axis("off")  # Hide the axis
            
            axes[4].imshow(layer_cropped_small, cmap="Reds")
            axes[4].axis("off")  # Hide the axis
            
            plt.tight_layout()
            plt.show()
            plt.pause(0.1)
            
            # Ask for user input
            user_input = input("Please enter 1 or 0: ")   
            while user_input not in ["1", "0"]:
                user_input = input("Invalid input. Please enter 1 or 0: ")
            user_value = int(user_input)
            label.append(user_value)
            plt.close(fig)
            

# Create a DataFrame from these lists

mean_ratio = [x / (y if y != 0 else 1) for x, y in zip(mean_intensity, mean_background)]
mean_difference = [x - y for x, y in zip(mean_intensity, mean_background)]

median_ratio = [x / (y if y != 0 else 1) for x, y in zip(median_intensity, median_background)]
median_difference = [x - y for x, y in zip(median_intensity, median_background)]

data = {
    'mean_intensity': mean_intensity,
    'median_intensit': median_intensity,
    'sd_intensity': sd_intensity,
    'min_intensity': min_intensity,
    'max_intensity': max_intensity,

    'mean_background': mean_background,
    'median_background': median_background,
    'sd_background': sd_background,
    'min_background': min_background,
    'max_background': max_background,
    
    'mean_ratio': mean_ratio,
    'mean_difference': mean_difference,
    'median_ratio': median_ratio,
    'median_difference': median_difference,

    'lbp_mean': lbp_mean,
    'lbp_std': lbp_std,
    'contrast': contrast,
    'correlation': correlation,
    'energy': energy,
    'homogeneity': homogeneity,

    'area': area,
    'perimeter': perimeter,
    'eccentricity': eccentricity,
    'major_axis_length': major_axis_length,
    'minor_axis_length': minor_axis_length,
    'solidity': solidity,
    'extent': extent,

    'hog_mean': hog_mean,
    'hog_std': hog_std,

    'label': label
}

df = pd.DataFrame(data)            
df['min_intensity'] = df['min_intensity'].astype(int)
df['max_intensity'] = df['max_intensity'].astype(int)
df['min_background'] = df['min_background'].astype(int)
df['max_background'] = df['max_background'].astype(int)
df['hog_mean'] = df['hog_mean'].astype(float)
df['hog_std'] = df['hog_std'].astype(float)

df.to_csv(f"{test_folder}/df_3.csv", index=False)

df = pd.read_csv(f"{test_folder}/tdt_training_results/df_3.csv")

from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
X_train = df_train.iloc[:, :-1].values
y_train = df_train.iloc[:, -1].values
X_test = df_test.iloc[:, :-1].values
y_test = df_test.iloc[:, -1].values

# =============================================================================
# SVM
# =============================================================================

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(random_state=42)),
    ('svm', SVC(kernel="rbf", random_state=42))
        ])

param_dist = {
    'pca__n_components': uniform(0.5, 0.5),
    'svm__C': uniform(1, 100),
    'svm__gamma': uniform(0.001, 0.1)
}

random_search = RandomizedSearchCV(pipeline,
                                   param_dist,
                                   n_iter=100,
                                   cv=5,
                                   random_state=42,
                                   n_jobs=-1)

random_search.fit(X_train, y_train)

print("Best parameters found: ", random_search.best_params_)
print("Best cross-validation score: ", random_search.best_score_)

# Best model from RandomizedSearchCV
best_model = random_search.best_estimator_

import pickle
with open(f"{test_folder}/best_model_svm.pkl", 'wb') as file:
    pickle.dump(best_model, file)
with open(f"{test_folder}/best_model_svm.pkl", 'rb') as file:
    best_model = pickle.load(file)

# =============================================================================
# Random forest
# =============================================================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# Random Forest
rf = RandomForestClassifier(random_state=42)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
param_dist = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Setting up RandomizedSearchCV
random_search = RandomizedSearchCV(rf, param_dist, n_iter=100, cv=5, random_state=42, n_jobs=-1, verbose=2)

# Fitting the RandomizedSearchCV
random_search.fit(X_train, y_train)

# Printing the best parameters and score
print("Best parameters found: ", random_search.best_params_)
print("Best cross-validation score: ", random_search.best_score_)

# Best model from RandomizedSearchCV
best_model = random_search.best_estimator_

import pickle
with open(f"{test_folder}/best_model_rf.pkl", 'wb') as file:
    pickle.dump(best_model, file)
with open(f"{test_folder}/best_model_rf.pkl", 'rb') as file:
    best_model = pickle.load(file)

# =============================================================================
# Logistic regression
# =============================================================================

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(random_state=42)),
    ('log_reg', LogisticRegression(random_state=42)),
        ])

param_dist = {
    'pca__n_components': uniform(0.5, 0.5),
    'log_reg__C': uniform(1, 100),
}

random_search = RandomizedSearchCV(pipeline,
                                   param_dist,
                                   n_iter=100,
                                   cv=5,
                                   random_state=42,
                                   n_jobs=-1)

random_search.fit(X_train, y_train)

print("Best parameters found: ", random_search.best_params_)
print("Best cross-validation score: ", random_search.best_score_)

# Best model from RandomizedSearchCV
best_model = random_search.best_estimator_

import pickle
with open(f"{test_folder}/best_model_logreg.pkl", 'wb') as file:
    pickle.dump(best_model, file)
with open(f"{test_folder}/best_model_logreg.pkl", 'rb') as file:
    best_model = pickle.load(file)

# =============================================================================
# Model evaluation training dataset
# =============================================================================

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
y_train_pred = cross_val_predict(best_model, X_train, y_train, cv=3)
confusion_matrix(y_train, y_train_pred)

from sklearn.metrics import precision_score, recall_score, f1_score
precision_score(y_train, y_train_pred)
recall_score(y_train, y_train_pred)
f1_score(y_train, y_train_pred)

# As the positive class is rare, we'll use the PR curve instead of the ROC curve

from sklearn.metrics import precision_recall_curve

y_scores = cross_val_predict(best_model, X_train, y_train, cv=3, method="decision_function") #SVM

y_scores = cross_val_predict(best_model, X_train, y_train, cv=3, method="predict_proba") #RandForest
y_scores = y_scores[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)

plt.plot(recalls, precisions, linewidth=2, label="Precision/Recall curve")
plt.xlabel('Recall TP/(TP+FN)')
plt.ylabel('Precision TP/(TP+FP)')
plt.show()

from sklearn.metrics import roc_auc_score
print("ROC-AUC score: ", roc_auc_score(y_train, y_scores))

# =============================================================================
# Model evaluation testing dataset
# =============================================================================

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
y_test_pred = cross_val_predict(best_model, X_test, y_test, cv=3)
confusion_matrix(y_test, y_test_pred)

from sklearn.metrics import precision_score, recall_score, f1_score
precision_score(y_test, y_test_pred)
recall_score(y_test, y_test_pred)
f1_score(y_test, y_test_pred)

# As the positive class is rare, we'll use the PR curve instead of the ROC curve

from sklearn.metrics import precision_recall_curve

y_scores = cross_val_predict(best_model, X_test, y_test, cv=3, method="decision_function") #SVM

y_scores = cross_val_predict(best_model, X_test, y_test, cv=3, method="predict_proba") #RandForest
y_scores = y_scores[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)

plt.plot(recalls, precisions, linewidth=2, label="Precision/Recall curve")
plt.xlabel('Recall TP/(TP+FN)')
plt.ylabel('Precision TP/(TP+FP)')
plt.show()

from sklearn.metrics import roc_auc_score
print("ROC-AUC score: ", roc_auc_score(y_test, y_scores))










































