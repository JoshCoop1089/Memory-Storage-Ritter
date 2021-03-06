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
import random

class ContextualBandit():

    def __init__(self, 
                    pulls_per_episode, episodes_per_epoch, 
                    num_arms, num_barcodes, barcode_size,
                    noise_observations,
                    perfect_info = False):

        # Task Specific
        self.pulls_per_episode = pulls_per_episode
        self.episodes_per_epoch = episodes_per_epoch
        self.noise_observations = noise_observations

        # Arm Specific
        self.num_arms = num_arms
        self.num_barcodes = num_barcodes
        self.barcode_size = barcode_size
        self.perfect_info = perfect_info

        self.epoch_mapping = self.generate_barcode_mapping()
    
        # input validation
        # Noise not implemented yet in observations
        assert 0 <= noise_observations < pulls_per_episode

    def sample(self, to_torch=True):
        """
        Get a single epochs worth of observations and rewards for a newly created barcode mapping

        Parameters:
        num_arms: int
            Number of unique arms which can be pulled
        num_barcodes: int
            Number of unique contexts which can map to a specific arm
            (Length of barcode is same as number of arms for now, can be changed)

        Returns:
        observations: tensor
            a sequence of arm pulls
        barcodes: tensor
            the context for each arm pull
        rewards: tensor
            the reward gained from an armpull in that specific context
        epoch_mapping: dict {barcode -> arm}
            shows which arm has the 90% reward chance per barcode
        """


        observation_p1, reward_p1, barcode_p1 = self.generate_trials_info(self.epoch_mapping)

        # Outputs stacked for LSTM
        observations = np.vstack([observation_p1])
        barcodes = np.vstack([barcode_p1])
        rewards = np.vstack([reward_p1])

        # to pytorch form
        if to_torch:
            observations = to_pth(observations)
            barcodes = to_pth(barcodes)
            rewards = to_pth(rewards, pth_dtype=torch.LongTensor)
        return observations, barcodes, rewards

    def generate_barcode_mapping(self):
        barcode_bag = set()
        mapping = {}

        # Create a set of unique binary barcodes
        # Array2String allows barcodes to be hashable in set to get uniqueness guarantees
        while len(barcode_bag) < self.num_barcodes:
            barcode = np.random.randint(0, 2, (1, self.barcode_size))
            # barcode -> string starts out at '[[1 1 0]]', thus the reductions on the end
            barcode_string = np.array2string(barcode)[2:-2].replace(" ", "")
            barcode_bag.add(barcode_string)

        barcode_bag_list = list(barcode_bag)

        # Generate mapping of barcode to good arm
        for barcode in barcode_bag_list:
            arm = random.randint(0, self.num_arms)
            mapping[barcode] = arm

        # At least one barcode for every arm gets guaranteed
        unique_guarantees = random.sample(barcode_bag_list, self.num_arms)
        for arm, guarantee in enumerate(unique_guarantees):
            mapping[guarantee] = arm

        return mapping

    def generate_trials_info(self, mapping):

        """
        LSTM Input Format:
        Single Trial of 10 pulls for one barcode
        Trial is a sequence of 10 one hot encoded pulls indicating the pulled arm

        [100, barcode2, 0] would be one pull in one trial for barcode2
        this would be a pull on arm0, and based on the mapping of barcode2, returns a reward of 0
        
        one episode is a sequence of 10 trials drawn from barcode bag
        one epoch is the full contents of barcode bag
        """

        # Create the trial sample bag with num_barcode instances of each unique barcode
        # 4 unique barcodes -> 16 total barcodes in bag, 4 copies of each unique barcode
        trial_barcode_bag = []
        for barcode in mapping:
            for i in range(self.num_barcodes):
                trial_barcode_bag.append(barcode)
        random.shuffle(trial_barcode_bag)

        observations = np.zeros((self.num_barcodes**2, self.pulls_per_episode, self.num_arms))
        rewards = np.zeros((self.num_barcodes**2, self.pulls_per_episode, 1))
        barcodes = np.zeros((self.num_barcodes**2, self.pulls_per_episode, self.barcode_size))

        for episode_num, barcode in enumerate(trial_barcode_bag):
            observations[episode_num], rewards[episode_num], barcodes[episode_num] = self.generate_one_episode(barcode, mapping)

        return observations, rewards, barcodes

    def generate_one_episode(self, barcode, mapping):

        # Generate arm pulling sequence for single trial
        # Creates an Arms X Pulls matrix, using np.eye to onehotencode arm pulls
        # Unsure if this is actually doing what i want, given docs say it shouldn't be able to random sample rows... and that's what i'm trying to do
        # But also it works when i tried it?  Need to check this more
        trial_pulls = np.eye(self.num_arms)[np.random.choice(
            self.num_arms,self.pulls_per_episode)]

        # Get reward for trial pulls
        # First pull barcode from mapping to ID good arm
        best_arm = mapping[barcode]
        trial_rewards = np.zeros((self.pulls_per_episode, 1))
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
            else:  # self.perfect_info == False
                reward = int(arm==best_arm)

            trial_rewards[pull] = reward

        # After generation of reward, use noise to obscure input for first x pulls of trial
        # Unsure how to implement this, just random generate a one hot encode, make more than one input, keep binary?
        
        # Transform the barcode back to an array for tensor broadcasting
        bar_ar = np.zeros((self.pulls_per_episode, self.barcode_size))
        for num in range(self.pulls_per_episode):
            for id, val in enumerate(barcode):
                bar_ar[num][id] = int(val)

        return trial_pulls, trial_rewards, bar_ar

def to_pth(np_array, pth_dtype=torch.FloatTensor):
    return torch.tensor(np_array).type(pth_dtype)

if __name__ == '__main__':

    pulls = 2
    episodes = 9
    noise = 0
    num_arms = 2
    num_barcodes = 3
    barcode_size = 2
    perfect_info = False
    task = ContextualBandit(pulls, episodes, num_arms, num_barcodes, barcode_size, noise, perfect_info)

    obs, bar, reward = task.sample()
    outp = np.dstack([obs, bar, reward])
    # print("Obs:", obs)
    # print("Bar:", bar)
    # print("Reward:", reward)
    print("Mapping:", task.epoch_mapping)
    print("Concat:", outp)
    obs, bar, reward = task.sample()
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