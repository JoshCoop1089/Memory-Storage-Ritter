from contextual_choice_sl import run_experiment

"""
Notes:
Barcode size needs to be at least 3 times as large as hamming_clustering
training_epochs cannot be less than 10
pulls_per_episode has to be more than 2

noise_epochs will be for one level of noise, the program will run through 4 levels of noise automatically
i'm not sure what will happen if num_barcodes isn't an integer multiple of num_arms

"""

###### Change These!!! ###########
# Experiment Type and Length
exp_types = ['context', 'embedding']
training_epochs = 750
noise_epochs = 50

# Experiment Difficulty
hamming_clustering = 1      #Create evenly distributed clusters based on arms/barcodes
sim_threshold = 0           #Create one cluster regardless of arms/barcodes
num_arms = 2
num_barcodes = 6
barcode_size = 16
pulls_per_episode = 10

# Randomized seed changes to average for returns graph
num_repeats = 3

###### NO MORE CHANGES!!!!!!!! ##########

exp_base = exp_types, training_epochs, noise_epochs, num_repeats
exp_difficulty = hamming_clustering, num_arms, num_barcodes, barcode_size, pulls_per_episode, sim_threshold
run_experiment(exp_base, exp_difficulty)