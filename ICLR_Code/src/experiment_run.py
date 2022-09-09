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
#context, embedding, hidden, L2RL

exp_types = ['context', 'embedding', 'hidden', 'L2RL']
training_epochs = 2500
noise_epochs = 250
noise_train_percent = 0

# Experiment Difficulty
hamming_clustering = 1      #Create evenly distributed clusters based on arms/barcodes
sim_threshold = 0           #Create one cluster regardless of arms/barcodes
num_arms = 10
num_barcodes = 10
barcode_size = 20
pulls_per_episode = 10

# Randomized seed changes to average for returns graph
num_repeats = 1

# Modify this to fit your machines save paths
figure_save_location = "..\\Memory-Storage-Ritter\\ICLR_Code\\figs\\"
###### NO MORE CHANGES!!!!!!!! ##########

exp_base = exp_types, training_epochs, noise_epochs, noise_train_percent, num_repeats, figure_save_location
exp_difficulty = hamming_clustering, num_arms, num_barcodes, barcode_size, pulls_per_episode, sim_threshold
run_experiment(exp_base, exp_difficulty)


"""
Tests to run:

2a4b, 2a8b, 4a8b, 4a12b, 6a12b
size 16, 24, 32


Barcode Number
Arm Number
Barcode Size
Baseline vs 1 Hamming Cluster

Train with noise?



"""