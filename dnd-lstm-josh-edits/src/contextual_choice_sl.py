"""demo: train a DND LSTM on a contextual choice task
"""
from sklearn.manifold import TSNE
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
    ### Noise not implemented yet ###
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
    dim_hidden_lstm = exp_settings['dim_hidden_lstm']
    dim_output_lstm = num_arms
    dict_len = num_barcodes #do i still need this given that i grow the dict with every barcode?
    learning_rate = 5e-4

    # init agent / optimizer
    agent = Agent(dim_input_lstm, dim_hidden_lstm, dim_output_lstm,
                     dict_len, exp_settings)
    optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)

    '''train'''
    log_sims = np.zeros(n_epochs,)
    run_time = np.zeros(n_epochs,)
    log_return = np.zeros((n_epochs,episodes_per_epoch),)
    log_embedder_accuracy = np.zeros((n_epochs, episodes_per_epoch),)
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
        # flush hippocampus
        agent.reset_memory()
        agent.turn_on_retrieval()
        agent.dnd.mapping = epoch_mapping

        # loop over the training set
        for m in range(episodes_per_epoch):
            # prealloc
            embedder_accuracy = 0
            cumulative_reward = 0
            probs, rewards, values = [], [], []
            h_t, c_t = agent.get_init_states()

            # Clearing the per trial hidden state buffer
            agent.flush_trial_buffer()

            barcodes_seen_prior = len(agent.dnd.key_context_map)
            # loop over time, for one training example
            for t in range(pulls_per_episode):
                # only save memory at the last time point
                agent.turn_off_encoding()
                if t == pulls_per_episode-1 and m < episodes_per_epoch:
                    agent.turn_on_encoding()

                output_t, _ = agent(observations[m][t].view(1, 1, -1), 
                                        barcodes[m][t].view(1, 1, -1),
                                        h_t, c_t)
                a_t, assumed_barcode, prob_a_t, v_t, h_t, c_t = output_t

                # compute immediate reward
                r_t = get_reward_from_assumed_barcode(a_t, reward_from_obs[m][t], 
                                                    assumed_barcode, epoch_mapping)
                # if r_t:
                #     print('reward!')

                # log
                probs.append(prob_a_t)
                rewards.append(r_t)
                values.append(v_t)
                cumulative_reward += r_t
                log_Y_hat[i, m, t] = a_t.item()

                # Does the embedder predicted context match the actual context?
                real_bc = np.array2string(barcodes[m][t].numpy())[
                    1:-1].replace(" ", "").replace(".", "")
                # print(real_bc, assumed_barcode)
                embedder_accuracy += int(real_bc == assumed_barcode)
                # if real_bc == assumed_barcode:
                #     print("context match!")

            barcodes_seen_after = len(agent.dnd.key_context_map)
            if barcodes_seen_after != barcodes_seen_prior:
                print("Episode:", m, "\t| Old:", barcodes_seen_prior, "\t| New:", barcodes_seen_after)
            returns = compute_returns(rewards)
            loss_policy, loss_value = compute_a2c_loss(probs, values, returns)
            loss = loss_policy + loss_value
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # log
            log_Y[i] = np.squeeze(reward_from_obs.numpy())

            # Updating avg return per episode
            episode_avg_return = cumulative_reward / pulls_per_episode
            if m == 0:
                log_return[i][m] = episode_avg_return
            else:
                # 0 indexed M counter need a boost
                log_return[i][m] = (log_return[i][m-1]*(m) + episode_avg_return)/(m+1)

            # Updating avg accuracy per episode
            episode_avg_accuracy = embedder_accuracy / pulls_per_episode
            if m == 0:
                log_embedder_accuracy[i][m] = episode_avg_accuracy
            else:
                # 0 indexed M counter need a boost
                log_embedder_accuracy[i][m] = (log_embedder_accuracy[i][m-1]*(m) + episode_avg_accuracy)/(m+1)
            
            log_loss_value[i] += loss_value.item() / episodes_per_epoch
            log_loss_policy[i] += loss_policy.item() / episodes_per_epoch


        # # Memory retrievals above sim threshold
        # good_pull = np.array(agent.dnd.recall_sims) >= sim_threshhold
        # # print(len(agent.dnd.recall_sims), (n_trials-1) * trial_length)
        # valid_pulls[i] = sum(good_pull)/ ((n_trials-1) * trial_length)

        # # Avg Similarity between queries and memory
        # log_sims[i] += np.mean(agent.dnd.recall_sims)

        # print out some stuff
        time_end = time.time()
        run_time[i] = time_end - time_start
        print(
            'Epoch %3d | avg_return = %.2f | loss: val = %.2f, pol = %.2f | time = %.2f | emb_accuracy = %.2f'%
            (i, log_return[i][-1], log_loss_value[i], log_loss_policy[i], run_time[i], log_embedder_accuracy[i][-1])
        )

    final_emb_acc = [log_embedder_accuracy[i][-1] for i in range(len(log_embedder_accuracy))]
    # avg_time = np.mean(run_time)
    # avg_sim = np.mean(log_sims)
    # print(f"-*-*- \n\tAvg Time: {avg_time:.2f}\n-*-*-")

    # Additional returns to graph out of this file
    keys, prediction_mapping = agent.get_all_mems_embedder()

    return log_return, log_loss_value, log_embedder_accuracy, keys, prediction_mapping, epoch_mapping

# Graphing Helper Functions
def expected_return(num_arms):
    perfect = 0.9
    random = 0.9*(1/num_arms) + 0.1*(num_arms-1)/num_arms
    return perfect, random

def get_barcode(mem_id, prediction_mapping):
    for k, v in prediction_mapping.items():
        if v == mem_id:
            return k
    print('Invalid mem_id:', mem_id)
    return ''

# Adapted from https://learnopencv.com/t-sne-for-feature-visualization/
def scale_to_01_range(x):
        value_range = (np.max(x) - np.min(x))
        starts_from_zero = x - np.min(x)
        return starts_from_zero / value_range

def plot_tsne_distribution(keys, labels, fig, axes):
    features = np.array([y.numpy() for y in keys])
    tsne = TSNE(n_components=2).fit_transform(features)
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)
    label_list = [labels[i][0] for i in range(len(labels))]
    start, end = 0,0
    # # for every class, we'll add a scatter plot separately
    for barcode, num, valid in labels:
        start = end
        end = end + num
        # find the samples of the current class in the data
        indices = [i for i in range(start, end)]
        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        marker = 'x' if valid else 'o'
        # add a scatter plot with the corresponding color and label
        # , label=f"B:{barcode} | Valid:{valid} | Num:{num}"
        axes.scatter(current_tx, current_ty, marker = marker)
    return fig, axes

if __name__  == '__main__':

    # Experimental Parameters
    exp_settings = {}
    exp_settings['randomize'] = False
    exp_settings['epochs'] = 1
    exp_settings['kernel'] = 'cosine'
    exp_settings['noise_percent'] = 0.5
    exp_settings['agent_input'] = 'obs'
    exp_settings['dim_hidden_lstm'] = 32
    exp_settings['embedding_size'] = 64
    exp_settings['num_arms'] = 10
    exp_settings['barcode_size'] = 6
    exp_settings['num_barcodes'] = 32
    exp_settings['pulls_per_episode'] = 50

    # Graph Setup
    graph_title = f"LSTM Hidden Dim: {exp_settings['dim_hidden_lstm']} | Embedding Size: {exp_settings['embedding_size']} | Arms: {exp_settings['num_arms']}\n" +\
        f"Barcode Dim: {exp_settings['barcode_size']} | Unique Barcodes: {exp_settings['num_barcodes']} | Pulls per Trial: {exp_settings['pulls_per_episode']}"
    perfect_ret, random_ret = expected_return(exp_settings['num_arms'])
    f, axes = plt.subplots(1, 2, figsize=(8, 5))

    # Only store valid barcode predictions from the embedder
    exp_settings['store_all'] = False
    log_return, log_loss_value, log_embedder_accuracy, keys, prediction_mapping, epoch_mapping = run_experiment_sl(exp_settings)
    for i, data in enumerate(log_return):
        axes[0].plot(data, label = f'Valid Stored')
    for i, data in enumerate(log_embedder_accuracy):
        axes[1].plot(data, label=f"Valid Stored")

    # Store every predicted barcode from embedder
    exp_settings['store_all'] = True
    log_return, log_loss_value, log_embedder_accuracy, keys, prediction_mapping, epoch_mapping = run_experiment_sl(exp_settings)
    # print("P-Bars to KeyID: ", prediction_mapping)
    # print("R-Bars to Arm:", epoch_mapping)
    for i, data in enumerate(log_return):
        axes[0].plot(data, label = f'All Stored')
    for i, data in enumerate(log_embedder_accuracy):
        axes[1].plot(data, label=f"All Stored")

    # Returns
    axes[0].axhline(y=perfect_ret, color='r', linestyle='dashed', label = 'Perfect Pulls')
    axes[0].axhline(y=random_ret, color='b', linestyle='dashed', label = 'Random Pulls')
    axes[0].set_ylabel('Returns')
    axes[0].set_xlabel('Episode')
    axes[0].legend(bbox_to_anchor=(0, -0.2, 1, 0), loc="upper left",
               mode="expand", borderaxespad=0, ncol=2)

    #Embedder Accuracy 
    axes[1].set_ylabel('Accuracy')
    axes[1].set_xlabel('Episode')
    axes[1].legend(bbox_to_anchor=(0, -0.2, 1, 0), loc="upper left",
               mode="expand", borderaxespad=0, ncol=2)
    sns.despine()
    f.tight_layout()
    f.subplots_adjust(top=0.9)
    plt.suptitle(graph_title)

    # T-SNE Mapping Attempts (from https://learnopencv.com/t-sne-for-feature-visualization/)
    labels = []
    for mem_id, barcode_keys in enumerate(keys):
        num_keys = len(barcode_keys)
        if num_keys > 0:
            barcode = get_barcode(mem_id, prediction_mapping)
            valid = False
            if barcode in epoch_mapping.keys():
                valid = True
            labels.append((barcode, num_keys, valid))
    # print(epoch_mapping.keys())
    # print(labels, sum(labels[i][1] for i in range(len(labels))))
    
    import itertools
    flattened_keys = list(itertools.chain.from_iterable(keys))
    # print(len(flattened_keys))
    
    f3, axes3 = plt.subplots(1, 1, figsize=(8, 5))
    f3, axes3 = plot_tsne_distribution(flattened_keys, labels, f3, axes3)
    axes3.xaxis.set_visible(False)
    axes3.yaxis.set_visible(False)
    axes3.set_title("t-SNE on Keys" + 
        "\nX is a real barcode, O is a false predicted barcode" +
        "\nDoesn't indicate if prediction was correct for specific observation" +
        "\nOnly if it was within the set of possibly correct barcodes")
    # axes3.legend(bbox_to_anchor=(0, -0.1, 1, 0), loc="upper left",
    #            mode="expand", borderaxespad=0, ncol=2)
    f3.tight_layout()
    plt.show()
