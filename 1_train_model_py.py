import pickle as pkl
from master_script import train_model
from master_script import plot_correlation
import csv
import matplotlib.pyplot as plt

# Enter the directory where you store your .ND2 images and your roi_dict.pkl
directory = '/run/user/1000/gvfs/smb-share:server=172.20.4.47,share=becell/Macro tests/'

# Import your ROIs
with open(directory + 'dict_rois.pkl', 'rb') as file:
    dict_rois = pkl.load(file)

def csv_to_dict(filename):
    data_dict = {}
    with open(filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if len(row) == 2:  # Ensure there are two columns in each row
                key, value = row
                data_dict[key] = float(value)
    return data_dict

actual_values = csv_to_dict(directory + 'counting.csv')

    
best_loss, best_hyperparameters, best_predicted_values, loss_list = train_model(dict_rois, actual_values, layer = 'layer_1', ratio = 1.55)
with open(directory + 'best_hyperparameters.pkl', 'wb') as file:
    pkl.dump(best_hyperparameters, file)

plt.plot(range(1, len(loss_list) + 1), loss_list, marker='o', linestyle='-')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

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



























