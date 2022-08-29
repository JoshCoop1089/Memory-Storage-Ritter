"""
Contextual Bandit Task

One task -> one barcode

Assume three arms to pull
If barcode is 001:
    arm3 gives 90% chance of reward
    arm1 and arm2 give 10% chance of reward

Generate barcode mapping to arm pull probability
dict where key is barcode and value is what arm has 90% chance of reward

From Ritter Paper:
    Each episode consisted of a series of pulls, throughout which the agent should efficiently
    find the best arm (explore) and pull it as many times as possible (exploit).
    During each episode, a context was presented which identified the reward probabilities.

# One episode is a sequence of pulls using only one barcode, in this case 10 pulls, rewards calculated via barcode mapping
# Create 10 episodes per unique context
# Thus, one epoch is a sequence of 100 episodes, allowing for 10 repeats of 10 pulls in one context

We sampled the sequence of tasks for each epoch as follows:
we first sampled a set of unique contexts, and paired
each element of that set randomly with one of the possible
rewarding arm positions b, ensuring that each rewarding
arm position was paired with at least one context. We then
created a bag S in which each (c; b) pair was duplicated
10 times. Finally, we sampled the task sequence for the
epoch by repeatedly sampling uniformly without replacement
tasks tn = (cn; bn)  unif(S). There were 100
episodes per epoch and 10 unique contexts per epoch. Thus,
each context was presented 10 times per epoch. There were
10 arms, and episodes were 10 trials long.

LSTM Input Format:
Single Trial of 10 pulls for one barcode
Trial is a sequence of 10 one hot encoded pulls indicating the pulled arm

[100, barcode2, r:0] would be one input to the LSTM
"""
import torch
import numpy as np
from numpy.linalg import norm
import random

class ContextualBandit():
    """
    Create a Contextual Bandit Task with either deterministic arm rewards or a 90%/10% reward chance
    """

    def __init__(self, 
                    pulls_per_episode, episodes_per_epoch, 
                    num_arms, num_barcodes, barcode_size,
                    reset_barcode_mapping, reset_arms_per_epoch,
                    sim_threshold, hamming_threshold, 
                    device, perfect_info = False):

        # Task Specific
        self.device = device
        self.episodes_per_epoch = episodes_per_epoch
        self.reset_barcode_mapping = reset_barcode_mapping
        self.reset_arms_per_epoch = reset_arms_per_epoch

        # Looping the LSTM inputs, so only need one pull per episode to start it off
        self.pulls_per_episode = 1

        # Arm Specific
        self.num_arms = num_arms
        self.num_barcodes = num_barcodes
        self.barcode_size = barcode_size
        self.perfect_info = perfect_info

        # Arm Clustering (Forcing barcodes to be close to each other, measured by cosine similarity)
        self.sim_threshold = sim_threshold
        self.hamming_threshold = hamming_threshold
        self.cluster_lists = []
    
        # Barcode to Arm Mapping generated on init, can be regenerated by sample if needed
        if self.hamming_threshold:
            self.epoch_mapping = self.generate_barcode_clusters()
        else:
            self.epoch_mapping = self.generate_barcode_mapping()

    def sample(self, to_torch=True):
        """
        Get a single epochs worth of observations and rewards for input to LSTM

        Args (Defined at program runtime):
            self.reset_barcode_mapping (Boolean): Whether you recreate the set of randomized barcodes
            self.reset_arms_per_epoch (Boolean): Whether you reset the distinct mapping of barcode to good arm, while not changing the original barcode values

        Returns:
            obs_barcodes_rewards (Tensor): All pulls/barcodes/rewards for the epoch
            self.epoch_mapping (Dict (String -> Int)): What arm is best for a barcode
            barcode_strings (Numpy Array): Replication of barcodes as string for easier referencing later in program
            barcode_tensors (Tensor): Replication of barcode tiled to match length of epoch
        """

        # Generate a new random set of barcodes to execute pulls from
        if self.reset_barcode_mapping:
            self.epoch_mapping = self.generate_barcode_mapping()

        # Reassign arms randomly within current barcode scheme
        if self.reset_arms_per_epoch:
            if self.hamming_threshold:
                self.epoch_mapping = self.map_arms_to_barcode_clusters(self.cluster_lists)
            else:
                self.epoch_mapping = self.map_arms_to_barcodes(self.epoch_mapping)

        observation_p1, reward_p1, barcode_p1, barcode_strings, barcode_id, arm_id = self.generate_trials_info(self.epoch_mapping)
        obs_barcodes_rewards = np.dstack([observation_p1, barcode_p1, reward_p1])

        # to pytorch form
        if to_torch:
            obs_barcodes_rewards = to_pth(obs_barcodes_rewards).to(self.device)
            barcode_tensors = to_pth(barcode_p1).to(self.device)
            barcode_ids = to_pth(barcode_id, pth_dtype=torch.long).to(self.device)
            arm_ids = to_pth(arm_id, pth_dtype=torch.long).to(self.device)
        return obs_barcodes_rewards, self.epoch_mapping, barcode_strings, barcode_tensors, barcode_ids, arm_ids

    def hamming_distance(self, barcode1, barcode2):
        return sum(c1 != c2 for c1, c2 in zip(barcode1, barcode2))

    def generate_barcode_clusters(self):
        barcode_bag = set()
        mapping = {}
        new_seed = False
        assert self.barcode_size - 2*self.hamming_threshold > self.hamming_threshold

        # Generate seperated seed barcodes
        while len(barcode_bag) < self.num_barcodes/self.num_arms:
            barcode = np.random.randint(0, 2, (self.barcode_size))
            if np.sum(barcode) == 0:
                continue
            if len(barcode_bag) == 0:
                seed_bc = barcode

            elif self.hamming_threshold:
                for seed_bc in barcode_bag:
                    h_d = self.hamming_distance(seed_bc, barcode)

                    # If the cluster seeds are too similar, throw it out
                    if h_d < self.barcode_size - 2*self.hamming_threshold:
                        new_seed = True
                        break

            # Current Seed is too close to other seeds
            if new_seed:
                new_seed = False
                continue

            # barcode -> string starts out at '[1 1 0]', thus the reductions on the end
            barcode_string = np.array2string(barcode)[1:-1].replace(" ", "")
            barcode_bag.add(barcode_string)

        # Barcode_bag now holds distinct cluster start points, create close clusters around those points
        bc_clusters = list(barcode_bag)
        for cluster in bc_clusters:
            mini_cluster_bag = set()
            mini_cluster_bag.add(cluster)
            cluster = np.asarray(list(cluster), dtype=int)
            while len(mini_cluster_bag) < self.num_arms:
                barcode = np.copy(cluster)

                # Noise can be used to simulate hamming distance
                noise_idx = np.random.choice(range(self.barcode_size), self.hamming_threshold, replace=False)
                
                # There's probably a nicer numpy way to do this but i cannot figure it out
                # Flipping the value at the noise location
                for idx in noise_idx:
                    barcode[idx] = barcode[idx]==0

                # Avoid cosine similarity bug with barcode of all 0's
                if np.sum(barcode) == 0:
                    continue

                elif self.hamming_threshold:
                    h_d = self.hamming_distance(cluster, barcode)
                    if h_d > self.hamming_threshold:
                        continue
                    else:
                        # barcode -> string starts out at '[1 1 0]', thus the reductions on the end
                        barcode_string = np.array2string(barcode)[1:-1].replace(" ", "")
                        mini_cluster_bag.add(barcode_string)

            # Need to store individual cluster for arm reshuffling at every epoch
            cluster_list = list(mini_cluster_bag)
            self.cluster_lists.append(cluster_list)
            cluster_mapping = self.map_arms_to_barcodes(barcode_list = cluster_list)
            mapping = mapping | cluster_mapping
        
        return mapping

    def map_arms_to_barcode_clusters(self, cluster_list):
        mapping = {}
        for cluster in cluster_list:
            mapping = mapping | self.map_arms_to_barcodes(mapping = None, barcode_list=cluster)
        return mapping

    def generate_barcode_mapping(self):
        barcode_bag = set()
        mapping = {}
        seed_reset = 0

        # Create a set of unique binary barcodes
        # Array2String allows barcodes to be hashable in set to get uniqueness guarantees
        while len(barcode_bag) < self.num_barcodes:
            if seed_reset > 10000:
                barcode_bag = set()
                print("stuck on old seed, reseting initial location")
                seed_reset = 0
            
            barcode = np.random.randint(0, 2, (self.barcode_size))
            seed_reset += 1

            # Avoid cosine similarity bug with barcode of all 0's
            if np.sum(barcode) == 0:
                continue

            if len(barcode_bag) == 0:
                seed_bc = barcode

            if self.sim_threshold:
                similarity = np.dot(seed_bc, barcode)/(norm(seed_bc)*norm(barcode))
                if similarity < self.sim_threshold:
                    continue

            # barcode -> string starts out at '[1 1 0]', thus the reductions on the end
            barcode_string = np.array2string(barcode)[1:-1].replace(" ", "")
            barcode_bag.add(barcode_string)

        barcode_bag_list = list(barcode_bag)
        mapping = self.map_arms_to_barcodes(mapping = None, barcode_list = barcode_bag_list)
        return mapping

    def map_arms_to_barcodes(self, mapping = None, barcode_list = None):
        if mapping:
            barcode_list = list(mapping.keys())
        else: #barcode_list != None is required
            mapping = {}

        # Generate mapping of barcode to good arm
        for barcode in barcode_list:
            arm = random.randint(0, self.num_arms-1)
            mapping[barcode] = arm

        # At least one barcode for every arm gets guaranteed
        unique_guarantees = random.sample(barcode_list, self.num_arms)
        for arm, guarantee in enumerate(unique_guarantees):
            mapping[guarantee] = arm

        return mapping

    def generate_trials_info(self, mapping):

        """
        LSTM Input Format:
        Trial is a sequence of X one hot encoded pulls indicating the pulled arm
        [1001010] -> human reads this as [100, barcode2, 0] would be one pull in one trial for barcode2
        this would be a pull on arm0, and based on the mapping of barcode2, returns a reward of 0
        
        one episode is a sequence of 10 trials drawn for a single barcode instance from barcode bag
        one epoch is the full contents of barcode bag
        """

        # Create the trial sample bag with num_barcode instances of each unique barcode
        # 4 unique barcodes -> 16 total barcodes in bag, 4 copies of each unique barcode

        # # Ishani's Grouping method for unsup learning
        # trial_barcode_bag = []
        # for i in range(self.num_barcodes):
        #     temp_list = []
        #     for barcode in mapping:
        #         temp_list.append(barcode)
        #     random.shuffle(temp_list)
        #     for k in temp_list:
        #         trial_barcode_bag.append(k)

        # Josh's general randomization method
        trial_barcode_bag = []
        for barcode in mapping:
            for _ in range(self.num_barcodes):
                trial_barcode_bag.append(barcode)
        random.shuffle(trial_barcode_bag)

        self.sorted_bcs = sorted(list(mapping.keys()))

        observations = np.zeros((self.num_barcodes**2, self.pulls_per_episode, self.num_arms))
        rewards = np.zeros((self.num_barcodes**2, self.pulls_per_episode, 1))
        barcodes = np.zeros((self.num_barcodes**2, self.pulls_per_episode, self.barcode_size))
        barcodes_strings = np.zeros((self.num_barcodes**2, self.pulls_per_episode, 1), dtype=object)
        barcodes_id = np.zeros((self.num_barcodes**2, 1))
        arms_id = np.zeros((self.num_barcodes**2, 1))

        for episode_num, barcode in enumerate(trial_barcode_bag):
            lstm_inputs, pre_comp_tensors = self.generate_one_episode(barcode, mapping)
            observations[episode_num], rewards[episode_num], barcodes[episode_num] = lstm_inputs
            barcodes_strings[episode_num], barcodes_id[episode_num], arms_id[episode_num] = pre_comp_tensors

        return observations, rewards, barcodes, barcodes_strings, barcodes_id, arms_id

    def generate_one_episode(self, barcode, mapping):
        """
        Create a single series of pulls, with rewards specified under the input barcode
        Args:
            barcode (String): Context Label for Arm ID
            mapping (Dict(String -> Int)): What arm is best for a barcode

        Returns:
            trial_pulls (Numpy Array): Contains all distinct arm pulls in order
            trial_rewards (Numpy Array): All rewards for arm pulls under input barcode
            bar_ar (Numpy Array): Input barcode tiled to match length of trial pulls
            bar_strings (Numpy Array): Input barcode as string for easier referencing later in program   
            bar_id (Numpy Array): Sorted ID's for Embedder loss calculations         
        """
        # Generate arm pulling sequence for single episode
        # Creates an Arms X Pulls matrix, using np.eye to onehotencode arm pulls
        trial_pulls = np.eye(self.num_arms)[np.random.choice(
            self.num_arms,self.pulls_per_episode)]

        # Get reward for trial pulls
        # First pull barcode from mapping to ID good arm
        best_arm = mapping[barcode]
        trial_rewards = np.zeros((self.pulls_per_episode, 1), dtype=np.float32)
        pull_num, arm_chosen = np.where(trial_pulls == 1)

        # Good arm has 90% chance of reward, all others have 10% chance
        for pull, arm in zip(pull_num, arm_chosen):
            if self.perfect_info == False:
                if arm == best_arm:
                    reward = int(np.random.random() < 0.9)
                else:
                    reward = int(np.random.random() < 0.1)
        
            # Deterministic Arm Rewards (for debugging purposes)
            # Make sure to change get_reward_from_assumed_barcode in utils.py as well
            else:  # self.perfect_info == True
                reward = int(arm==best_arm)

            trial_rewards[pull] = float(reward)
        
        # Tile the barcode for all pulls in the episode
        bar_strings = np.zeros((self.pulls_per_episode, 1), dtype = object)
        bar_ar = np.zeros((self.pulls_per_episode, self.barcode_size))
        for num in range(self.pulls_per_episode):
            bar_strings[num] = barcode
            for id, val in enumerate(barcode):
                bar_ar[num][id] = int(val)

        bar_id = self.sorted_bcs.index(barcode)

        lstm_inputs = trial_pulls, trial_rewards, bar_ar
        pre_comp_tensors = bar_strings, bar_id, best_arm

        return lstm_inputs, pre_comp_tensors

def to_pth(np_array, pth_dtype=torch.FloatTensor):
    return torch.as_tensor(np_array).type(pth_dtype)

if __name__ == '__main__':

    pulls = 1
    episodes = 1
    noise = 0
    num_arms = 2
    num_barcodes = 2
    barcode_size = 2
    perfect_info = True
    reset_barcode_mapping = True

    device = torch.device('cpu')
    task = ContextualBandit(pulls, episodes, num_arms, num_barcodes, 
                            barcode_size, reset_barcode_mapping, 
                            noise, device, perfect_info)

    obs, bar, reward, map = task.sample()
    outp = np.dstack([obs, bar, reward])
    print("Obs:", obs)
    print("Bar:", bar)
    print("Reward:", reward)
    print("Mapping:", task.epoch_mapping)
    print("Concat:", outp)
    obs, bar, reward, map = task.sample()
    outp = np.dstack([obs, bar, reward])
    # print("Obs:", obs)
    # print("Bar:", bar)
    # print("Reward:", reward)
    print("Mapping:", task.epoch_mapping)
    print("Concat:", outp)

    """
    Observation:
        Onehot encoded arm pull
            (ie, pull arm 2, input is 010)
    Barcode
        (example case above (001 -> 90% arm3 reward))
    Reward:
        based on barcode, apply percent chance to observation
            (in this case, 10% chance due to pulling arm2 in arm3 barcode)

    Epoch/Sample Generation:
        Number of times a barcode repeats over a single trial = # total barcodes



    LSTM:
        inputs:
            Observation, Barcode
                might not pass both into model, but both are required for embedder training

        generate hidden state:
            pass HS through embedder to check for task in memory (dnd.get_memory)
                Embedder would output both the embedding and the softmax distribution of what it thinks the task is
                    Barcode one hot encoding as class labels for outputs?
                this is also where embedder training happens with actual barcode used for cross-entropy loss
            return LSTM c-state and reintegrate

            pass HS into RL algo
        RL:
            hidden state of LSTM -> a2c
            a2c gives policy
            policy into pick action
            pick action gives what arm to pull

            output:
                what arm to pull

        LSTM Outputs:
            arm to pull, embedding assumed barcode
        

    Reward for LSTM:
        agent will have to also export the embedding assumed task barcode to ID arm chances
        given export barcode, pull lever based on action choice from RL
        barcode gives prob over arms
        Reward is based on distrib over arms/arm choice


    *** Need new reward function for pulling arm
    """