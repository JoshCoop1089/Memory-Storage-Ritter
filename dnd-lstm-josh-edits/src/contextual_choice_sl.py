"""demo: train a DND LSTM on a contextual choice task
"""
from asyncio import run
import time
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from task.ContextBandits import ContextualBandit
from sl_model import DNDLSTM as Agent
from utils import compute_stats, to_sqnp
from sl_model.DND import compute_similarities
from sl_model.utils import get_reward_from_assumed_barcode, compute_returns, compute_a2c_loss

def run_experiment_sl(exp_settings):
    """
    exp_settings is a dict with parameters as keys:

    randomize: Boolean (for performing multiple trials to average results or not)
    epochs: int (number of times to wipe memory and rerun learning)
    kernel: string (should be either 'l2' or 'cosine')
    agent_input: string (choose between passing obs/context, or only obs into LSTM)
    num_arms: int (number of unique arms to choose from)
    barcode_size: int (dimension of barcode used to specify good arm)
    num_barcodesL: int (number of unique contexts to define)
    pulls_per_episode: int (how many arm pulls are given to each unique barcode)
    noise_percent: float (between 0 and 1 to make certain percent of observations useless)
    embedding_size: int (how big is the embedding model size)
    """

    if not exp_settings['randomize']:
        seed_val = 0
        torch.manual_seed(seed_val)
        np.random.seed(seed_val)
   
    '''init task'''
    # Example: 4 unique barcodes -> 16 total barcodes in epoch, 4 trials of each unique barcode
    num_barcodes = exp_settings['num_barcodes']
    episodes_per_epoch = num_barcodes**2

    # Arm pulls per single barcode episode
    pulls_per_episode = exp_settings['pulls_per_episode']

    # Make a percent of observed pulls useless for prediction
    noise_percent = exp_settings['noise_percent']
    noise_observations = int(pulls_per_episode * noise_percent)

    # Task Choice
    task = ContextualBandit(
        pulls_per_episode,
        episodes_per_epoch,
        noise_observations)


    """
    #Training the LSTM agent
    exp_settings['randomize']
    exp_settings['epochs']
    exp_settings['kernel']
    exp_settings['agent_input']

    # A single epoch of samples
    exp_settings['num_arms']
    exp_settings['barcode_size']
    exp_settings['num_barcodes']
    exp_settings['pulls_per_episode']
    exp_settings['noise_percent']
    """

    '''init model'''
    n_epochs = exp_settings['epochs']
    kernel = exp_settings['kernel']
    agent_input = exp_settings['agent_input']

    # input/output/hidden/memory dim
    num_arms = exp_settings['num_arms'] #LSTM input dim
    #Embedder output dim (not embedding size, just what we're using to get barcode predictions), possible LSTM input
    barcode_size = exp_settings['barcode_size'] 

    # Input to LSTM is only observation
    if agent_input == 'obs':
        dim_input_lstm = num_arms
    
    # Input is obs/context pair
    else:  # agent_input == 'obs/context'
        dim_input_lstm = num_arms + barcode_size

    # set params
    dim_hidden_lstm = 32
    dim_output_lstm = num_arms
    dict_len = num_barcodes #only one memory slot per barcode
    learning_rate = 5e-4

    # init agent / optimizer
    agent = Agent(dim_input_lstm, dim_hidden_lstm, dim_output_lstm,
                     dict_len, exp_settings)
    optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)

    '''train'''
    log_sims = np.zeros(n_epochs,)
    run_time = np.zeros(n_epochs,)
    log_return = np.zeros(n_epochs,)
    log_embedder_accuracy = np.zeros(n_epochs,)
    log_loss_value = np.zeros(n_epochs,)
    log_loss_policy = np.zeros(n_epochs,)

    log_Y = np.zeros((n_epochs, episodes_per_epoch, pulls_per_episode))
    log_Y_hat = np.zeros((n_epochs, episodes_per_epoch, pulls_per_episode))

    # loop over epoch
    for i in range(n_epochs):
        time_start = time.time()

        # Need to decide if the mapping of barcode to arm will change every epoch
        # If this does change, will need to retrain the embedder model every epoch as well
        # get data for this epoch
        observations, barcodes, reward_from_obs, epoch_mapping = task.sample(num_arms, num_barcodes, barcode_size)
        # print(X)
        # flush hippocampus
        agent.reset_memory()
        agent.turn_on_retrieval()

        # loop over the training set
        for m in range(episodes_per_epoch):
            # prealloc
            embedder_accuracy = 0
            cumulative_reward = 0
            probs, rewards, values = [], [], []
            h_t, c_t = agent.get_init_states()

            # Clearing the per trial hidden state buffer
            agent.flush_trial_buffer()

            # # Freeze DNDLSTM Agent here to not interfere with embedding training
            # # HOW TO FREEZE AGENT IDKMYBFFJILL
            for param in agent.parameters():
                param.requires_grad = False
                print(param)

            # loop over time, for one training example
            for t in range(pulls_per_episode):
                # only save memory at the last time point
                agent.turn_off_encoding()
                if t == pulls_per_episode-1 and m < episodes_per_epoch:
                    agent.turn_on_encoding()

                # Flag to turn on gradients in the embedder model
                enable_embedder_layers = False
                if t == 0:
                    enable_embedder_layers = True

                output_t, _ = agent(observations[m][t].view(1, 1, -1), 
                                        barcodes[m][t].view(1, 1, -1),
                                        h_t, c_t,
                                        enable_embedder_layers)
                a_t, assumed_barcode, prob_a_t, v_t, h_t, c_t = output_t

                # compute immediate reward
                r_t = get_reward_from_assumed_barcode(a_t, rewards[m][t], 
                                                    assumed_barcode, epoch_mapping)

                # log
                probs.append(prob_a_t)
                rewards.append(r_t)
                values.append(v_t)
                cumulative_reward += r_t
                log_Y_hat[i, m, t] = a_t.item()

                # Does the embedder predicted context match the actual context?
                # barcode -> string starts out at '[[1 1 0]]', thus the reductions on the end
                barcode_ground = np.array2string(barcodes[m][t].numpy())[2:-2].replace(" ", "")
                embedder_accuracy += (barcode_ground == assumed_barcode)

            # Unfreeze DNDLSTM Agent, after freezing embedding agent in save_memories
            for param in agent.parameters():
                param.requires_grad = True

            returns = compute_returns(rewards)
            loss_policy, loss_value = compute_a2c_loss(probs, values, returns)
            loss = loss_policy + loss_value
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log
            log_Y[i] = np.squeeze(reward_from_obs.numpy())
            log_embedder_accuracy[i] += embedder_accuracy / episodes_per_epoch
            log_return[i] += cumulative_reward / episodes_per_epoch
            log_loss_value[i] += loss_value.item() / episodes_per_epoch
            log_loss_policy[i] += loss_policy.item() / episodes_per_epoch


        # # Memory retrievals above sim threshold
        # good_pull = np.array(agent.dnd.recall_sims) >= sim_threshhold
        # # print(len(agent.dnd.recall_sims), (n_trials-1) * trial_length)
        # valid_pulls[i] = sum(good_pull)/ ((n_trials-1) * trial_length)

        # Avg Similarity between queries and memory
        log_sims[i] += np.mean(agent.dnd.recall_sims)

        # print out some stuff
        time_end = time.time()
        run_time[i] = time_end - time_start
        print(
            'Epoch %3d | return = %.2f | loss: val = %.2f, pol = %.2f | time = %.2f | emb_accuracy = %.2f'%
            (i, log_return[i], log_loss_value[i], log_loss_policy[i], run_time[i], log_embedder_accuracy[i])
        )
    avg_embedder_accuracy = np.mean(log_embedder_accuracy)
    avg_time = np.mean(run_time)
    avg_sim = np.mean(log_sims)
    print(f"-*-*- \n\tAvg Time: {avg_time:.2f} | Embedder Accuracy over Epoch: {avg_embedder_accuracy:.2f}\n-*-*-")

    # Additional returns to graph out of this file
    keys, vals = agent.get_all_mems_josh()

    return log_return, log_loss_value, log_embedder_accuracy, keys, vals


if __name__  == '__main__':
    exp_settings = {}
    exp_settings['randomize'] = False
    exp_settings['epochs'] = 2
    exp_settings['kernel'] = 'cosine'
    exp_settings['noise_percent'] = 0.5
    exp_settings['agent_input'] = 'obs'
    exp_settings['embedding_size'] = 16
    exp_settings['num_arms'] = 3
    exp_settings['barcode_size'] = 3
    exp_settings['num_barcodes'] = 4
    exp_settings['pulls_per_episode'] = 4

    a,b,c,d,e = run_experiment_sl(exp_settings)
