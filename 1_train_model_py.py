import pickle as pkl
from master_script import train_model
from master_script import plot_correlation

# Enter the directory where you store your .ND2 images and your roi_dict.pkl
directory = '//folder/becell/Macro tests/images_cfos/'

# Import your ROIs
with open(directory + 'dict_rois.pkl', 'rb') as file:
    dict_rois = pkl.load(file)

file_names = ['Male_1_ca3_001', 'Male_1_ca3_002', 'Male_1_Cg_001', 'Male_1_Cg_002', 'Male_1_dg_001', 'Male_1_dg_002', 
              'Male_1_PFC_003', 'Male_1_PFC_004', 'Male_1_RSC_001', 'Male_1_RSC_002', 'Male_4_ca3_003', 'Male_4_ca3_004', 
              'Male_4_Cg_003', 'Male_4_Cg_004', 'Male_4_dg_002', 'Male_4_dg_003', 'Male_4_PFC_002', 'Male_4_PFC_003', 
              'Male_4_PFC_006', 'Male_4_RSC_001', 'noshock_13_hip_003', 'noshock_17_bla_005', 'paired_2_hip_002', 
              'paired_2_hip_003', 'paired_6_bla_002', 'paired_6_hip_004', 'unpaired_7_bla_003']
manual_counts = [68, 71.5, 225, 153.5, 88, 92.5, 164, 203, 136.5, 162.5, 69, 56.5, 103, 150, 26.5, 46.5, 175.5, 74.5, 
                 148, 123.5, 61, 77, 83, 75, 63, 63, 102]
actual_values = dict(zip(file_names, manual_counts))
    
best_loss, best_hyperparameters, best_predicted_values = train_model(dict_rois, actual_values, layer = 'layer_1', ratio = 1.55)
with open(directory + 'best_hyperparameters.pkl', 'wb') as file:
    pkl.dump(best_hyperparameters, file)

plot_correlation(actual_values, best_predicted_values)