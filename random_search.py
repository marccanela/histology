import random
from master_script import create_dict_rois
from master_script import create_dict_of_binary
from master_script import apply_watershed_cfos
from tqdm import tqdm
import time
import numpy as np
import math
import copy
import matplotlib.pyplot as plt

directory = 'C:/Users/mcanela/Desktop/hipocampus/'
# dict_rois = create_dict_rois(directory)

import pickle as pk
# with open(directory + 'dict_rois.pkl', 'wb') as file:
#     pk.dump(dict_rois, file)
with open(directory + 'dict_rois_jose.pkl', 'rb') as file:
    dict_rois = pk.load(file)

# dict_of_binary = create_dict_of_binary(dict_rois, 'layer_1')


def evaluate(dict_of_binary, ratio, hyperparameters, actual_values):
    
    predicted_values = {}
    for key, value in dict_of_binary.items():
        binary_image = ~value[0]
        blurred_normalized = value[4]
        cells = apply_watershed_cfos(binary_image, blurred_normalized, ratio, 3, 5,
                                     **hyperparameters)
        predicted_values[key] = len(cells)

    common_keys = set(actual_values.keys()) & set(predicted_values.keys())

    # Calculate RMSE
    squared_diff = [(actual_values[key] - predicted_values[key])**2 for key in common_keys]
    mse = sum(squared_diff) / len(common_keys)
    rmse = math.sqrt(mse)
    
    # Calculate MAE
    absolute_diff = [abs(actual_values[key] - predicted_values[key]) for key in common_keys]
    mae = sum(absolute_diff) / len(common_keys)

    return mae, predicted_values

def random_search(dict_of_binary, num_iterations, ratio, actual_values):
    best_loss = float('inf')
    best_hyperparameters = None
    best_predicted_values = None

    for _ in tqdm(range(num_iterations), desc="Performing iterations", unit="iterations"):
        
        # Randomly sample hyperparameters from the search space
        hyperparameters = {
            'distance_2': random.uniform(7, 10), # (between 7-10)
            'eccentricity_1': random.uniform(0.5, 0.6), # (between 0.5-0.6)
            'distance_3': random.uniform(3, 5), # (between 3-5)
            'color_1': random.uniform(0.8, 1), # (between 0.8-1)
            'distance_4': random.uniform(3, 20), # max range! (between 3-20)
        }
        
        # # Add constraints to the hyperparameters
        # while not hyperparameters['min_diameter_threshold_micron'] > hyperparameters['nucleus_diameter']:
        #     # If constraints are not met, resample hyperparameters
        #     hyperparameters = {
        #         'min_diameter_threshold_micron': random.uniform(10, 18),
        #         'min_eccentricity': random.uniform(0.5, 1),
        #         'color_ratio_threshold': random.uniform(0.6, 1),
        #         'nucleus_diameter': random.uniform(3, 10),
        #     }
        
        loss, predicted_values = evaluate(dict_of_binary, ratio, hyperparameters, actual_values)
                
        # Update best accuracy and hyperparameters if current iteration is better
        if loss < best_loss:
            best_loss = loss
            best_hyperparameters = hyperparameters
            best_predicted_values = predicted_values
        time.sleep(0.1)

    return best_loss, best_hyperparameters, best_predicted_values

# best_rmse, best_hyperparams, best_predicted_values = random_search(dict_of_binary, 200, 1.55)
# print("Best Loss:", best_loss)
# print("Best Hyperparameters:", best_hyperparams)
# print("Best Predicted values:", best_predicted_values)


# =============================================================================
# Genetic Algorithm for hyperparameter optimization
# =============================================================================

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

# Example usage:
# best_loss, best_hyperparameters, best_predicted_values, loss_list = genetic_algorithm(dict_of_binary, 50, 1.55)
# print("Best Loss:", best_loss)
# print("Best Hyperparameters:", best_hyperparameters)
# print("Best Predicted Values:", best_predicted_values)

plt.plot(range(1, len(loss_list) + 1), loss_list, marker='o', linestyle='-')
plt.title('Loss as a function of Iterations')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
































