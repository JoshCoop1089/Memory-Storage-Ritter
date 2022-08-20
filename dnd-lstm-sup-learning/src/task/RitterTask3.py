"""
---- Task Setup  ----

assume 400 contexts of size 20

create bag of 400 unique contexts
shuffle the context list

split bag in half
assign half to low prob
assign half to high prob

create reward_accumulator with context as key and 0 as value

create duplicates bag
5x copies of all low prob bcs
5x copies of all high prob bcs

shuffle the duplicate bags individually

one episode is a random draw w/o replacement from both bags

randomize the location of each prob and create inputs for epoch
for low,high in zip(low_dup, high_dup):
    episode = np.zeros((2,context_size))
    chance = np.zeros(2)
    loc = random.randint(0,2)
    episode[loc] = low
    episode[1-loc] = high
    chance[loc] = 0.1
    chance[1-loc] = 0.9
        
turn episode_list in tensors of size [episodes, 2*context_size]
    
output = episode_list, chance_list, reward_accumulator
    
---- LSTM ----
take episode list and choose either left or right context to check against memory
loc = random.randint(0,2)
action, (all the other stuff) = agent(episode_list[episode][loc])

look up passed in context in reward_mapping to get percent chance
increment reward value in reward_acculumator for average return calc at end

loss calc
backprop


--- DNDLSTM ---
input is the same as ritter1
memsearch/store is the same as ritter1
"""

"""
General Improvements
    Set up savepoints for memory keys at multiple points during training
        First epoch, 10%, 50%, end?
    Use the t-sne graph to create a 2x2 output to display the time skips on embedding seperations


Noisy Task
    After a full training run on regular embedder training, freeze model
    Add "noise" boolean around the backprop loops


Multiple Barcodes Per Arm Task
    manual set of exp_settings['num_barcodes'] to something more than num_arms
    how to train embedder
        use distinct arm pulls instead of barcodes?
    how to gauge accuracy of memory recall
        set up two trackers
            one for barcode prediction
            one for arm pull predictions
        

"""