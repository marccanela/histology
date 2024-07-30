import csv
import pickle as pkl

import matplotlib.pyplot as plt

from master_script import plot_correlation, train_model

# Enter the directory where you store your .ND2 images and your roi_dict.pkl
directory = "//folder/becell/Macro tests/"
directory = "/run/user/1000/gvfs/smb-share:server=172.20.4.47,share=becell/Macro tests/"

# Import your ROIs
with open(directory + "dict_rois.pkl", "rb") as file:
    dict_rois = pkl.load(file)


def csv_to_dict(filename):
    data_dict = {}
    with open(filename, "r") as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if len(row) == 3:  # Ensure there are two columns in each row
                key, value1, value2 = row
                data_dict[key] = [float(value1), float(value2)]
    return data_dict


actual_values = csv_to_dict(directory + "counting_sd.csv")

best_hyper, predicted_training, predicted_validation, loss_list = train_model(
    dict_rois, actual_values, "layer_1", 1.55, "zscore"
)
with open(directory + "best_hyperparameters_zscore.pkl", "wb") as file:
    pkl.dump(best_hyper, file)

plt.plot(range(1, len(loss_list) + 1), loss_list, marker="o", linestyle="-")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

plot_correlation(actual_values, predicted_training)
plot_correlation(actual_values, predicted_validation)

import matplotlib.pyplot as plt

# =============================================================================
# Calculate the distribution of errors (if Gaussian, use RMSE instead of MAE)
import numpy as np
from scipy.stats import kstest, probplot

# Calculate errors
errors = []
for key in actual_values:
    errors.append(actual_values[key] - predicted_training.get(key, 0))
# Convert errors to numpy array
errors = np.array(errors)
# Perform Kolmogorov-Smirnov test
statistic, p_value = kstest(errors, "norm")

# Interpret the result
print("Kolmogorov-Smirnov Test:")
print(f"Test Statistic: {statistic}")
print(f"P-value: {p_value}")

if p_value > 0.05:
    print(
        "The errors follow a Gaussian distribution (fail to reject the null hypothesis)"
    )
    print("Better use RMSE as a loss function.")
else:
    print(
        "The errors do not follow a Gaussian distribution (reject the null hypothesis)"
    )
    print("Better use MAE as a loss function.")

# Create Q-Q plot
plt.figure(figsize=(8, 6))
probplot(errors, dist="norm", plot=plt)
plt.title("Q-Q Plot of Errors")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")
plt.grid(True)
plt.show()
