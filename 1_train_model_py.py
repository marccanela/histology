import pickle as pkl
from master_script import train_model
from master_script import plot_correlation

# Enter the directory where you store your .ND2 images and your roi_dict.pkl
directory = '//folder/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2024-01a02 - cFos/microscope/hippocampus/'

# Import your ROIs
with open(directory + 'rois_dg.pkl', 'rb') as file:
    dict_rois = pkl.load(file)

file_names = [
    "noshock_13_hip_001", "noshock_13_hip_002", "noshock_13_hip_004",
    "noshock_14_hip_001", "noshock_14_hip_002", "noshock_14_hip_003", "noshock_14_hip_004",
    "noshock_15_hip_001", "noshock_15_hip_002", "noshock_15_hip_003", "noshock_15_hip_004",
    "noshock_16_hip_001", "noshock_16_hip_002", "noshock_16_hip_003", "noshock_16_hip_004",
    "noshock_17_hip_001", "noshock_17_hip_002", "noshock_17_hip_003", "noshock_17_hip_004",
    "noshock_18_hip_001", "noshock_18_hip_002", "noshock_18_hip_003", "noshock_18_hip_004",
    "paired_1_hip_001", "paired_1_hip_002", "paired_1_hip_003", "paired_1_hip_004",
    "paired_2_hip_001", "paired_2_hip_002", "paired_2_hip_003", "paired_2_hip_004", "paired_2_hip_005",
    "paired_3_hip_001", "paired_3_hip_002", "paired_3_hip_003", "paired_3_hip_004",
    "paired_4_hip_002", "paired_4_hip_003",
    "paired_5_hip_001", "paired_5_hip_002", "paired_5_hip_003", "paired_5_hip_004",
    "paired_6_hip_001", "paired_6_hip_002", "paired_6_hip_003", "paired_6_hip_004",
    "unpaired_7_hip_001", "unpaired_7_hip_002", "unpaired_7_hip_003", "unpaired_7_hip_004", "unpaired_7_hip_005",
    "unpaired_8_hip_001", "unpaired_8_hip_002", "unpaired_8_hip_003", "unpaired_8_hip_004",
    "unpaired_9_hip_001", "unpaired_9_hip_002", "unpaired_9_hip_003", "unpaired_9_hip_004", "unpaired_9_hip_005",
    "unpaired_10_hip_001_2", "unpaired_10_hip_002", "unpaired_10_hip_003", "unpaired_10_hip_004", "unpaired_10_hip_005",
    "unpaired_11_hip_001", "unpaired_11_hip_002",
    "unpaired_12_hip_001", "unpaired_12_hip_002", "unpaired_12_hip_003", "unpaired_12_hip_004", "unpaired_12_hip_005"
]
manual_counts = [
    28, 41, 25, 32, 33, 41, 53, 61, 59, 40, 
    52, 31, 26, 33, 48, 63, 45, 53, 41, 56, 
    45, 57, 63, 53, 39, 43, 38, 38, 33, 43, 
    32, 29, 53, 77, 86, 55, 40, 32, 36, 42, 
    41, 37, 65, 60, 37, 45, 37, 44, 48, 42, 
    33, 58, 40, 51, 43, 57, 39, 60, 48, 51, 
    53, 33, 37, 40, 31, 45, 34, 41, 44, 29, 
    39, 32
]
actual_values = dict(zip(file_names, manual_counts))
    
best_loss, best_hyperparameters, best_predicted_values = train_model(dict_rois, actual_values, layer = 'layer_1', ratio = 1.55)
with open(directory + 'best_hyperparameters.pkl', 'wb') as file:
    pkl.dump(best_hyperparameters, file)

plot_correlation(actual_values, best_predicted_values)

# =============================================================================
# Calculate the distribution of errors (if Gaussian, use RMSE instead of MAE)
import numpy as np
from scipy.stats import kstest, probplot
import matplotlib.pyplot as plt
# Calculate errors
errors = []
for key in actual_values:
    errors.append(actual_values[key] - best_predicted_values.get(key, 0))
# Convert errors to numpy array
errors = np.array(errors)
# Perform Kolmogorov-Smirnov test
statistic, p_value = kstest(errors, 'norm')

# Interpret the result
print("Kolmogorov-Smirnov Test:")
print(f"Test Statistic: {statistic}")
print(f"P-value: {p_value}")

if p_value > 0.05:
    print("The errors follow a Gaussian distribution (fail to reject the null hypothesis)")
    print("Better use RMSE as a loss function.")
else:
    print("The errors do not follow a Gaussian distribution (reject the null hypothesis)")
    print("Better use MAE as a loss function.")

# Create Q-Q plot
plt.figure(figsize=(8, 6))
probplot(errors, dist="norm", plot=plt)
plt.title("Q-Q Plot of Errors")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")
plt.grid(True)
plt.show()



























