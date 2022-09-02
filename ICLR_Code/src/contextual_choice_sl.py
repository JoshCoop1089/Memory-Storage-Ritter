"""demo: train a DND LSTM on a contextual choice task
"""
# Win64bit Optimizations for TSNE
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.manifold import TSNE
import time
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from task.ContextBandits import ContextualBandit
from sl_model import DNDLSTM as Agent
from sl_model.utils import get_reward_from_assumed_barcode, compute_returns, compute_a2c_loss

def run_experiment_sl(exp_settings):
    """
    exp_settings is a dict with parameters as keys:

    randomize: Boolean (for performing multiple trials to average results or not)
    epochs: int (number of times to wipe memory and rerun learning)
    kernel: string (should be either 'l2' or 'cosine')
    agent_input: string (choose between passing obs/context, or only obs into LSTM)
    mem_store: string (what is used as keys in the memory)
    num_arms: int (number of unique arms to choose from)
    barcode_size: int (dimension of barcode used to specify good arm)
    num_barcodes: int (number of unique contexts to define)
    pulls_per_episode: int (how many arm pulls are given to each unique barcode episode)
    perfect_info: Boolean (True -> arms only give reward if best arm is pulled, False -> 90%/10% chances on arms as usual)
    noise_percent: float (between 0 and 1 to make certain percent of observations useless)
    embedding_size: int (how big is the embedding model size)
    reset_barcodes_per_epoch: Boolean (create a new random set of barcodes at the start of every epoch, or keep the old one)
    reset_arms_per_epoch: Boolean (remap arms to barcodes at the start of every epoch, or keep the old one)
    lstm_learning_rate: float (learning rate for the LSTM-DND main agent optimizer)
    embedder_learning_rate: float (learning rate for the Embedder-Barcode Prediction optimizer)
    task_version: string (bandit or original QiHong task)
    """
    # Tensorboard viewing
    if exp_settings['tensorboard_logging']:
        tb = SummaryWriter()

    # See Experimental parameters for GPU vs CPU choices
    if exp_settings['torch_device'] == 'CPU':
        device = torch.device('cpu')
    elif exp_settings['torch_device'] == 'GPU':
        device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        raise ValueError("Incorrect Torch Device set")

    if not exp_settings['randomize']:
        seed_val = 0
        torch.manual_seed(seed_val)
        np.random.seed(seed_val)

    # Full training and noise eval length
    n_epochs = exp_settings['epochs'] + exp_settings['noise_eval_epochs']*len(exp_settings['noise_percent'])
   
    '''init task'''
    # input/output/hidden/memory dim
    num_arms = exp_settings['num_arms']             # LSTM input dim
    barcode_size = exp_settings['barcode_size']     # Possible additional LSTM input dim
    num_barcodes = exp_settings['num_barcodes']     # Different number of contexts

    # Arm pulls per single barcode episode
    pulls_per_episode = exp_settings['pulls_per_episode']

    # Arm rewards can be deterministic for debugging
    perfect_info = exp_settings['perfect_info']

    # Cluster barcodes at the start (Only use one per experiment)
    sim_threshold = exp_settings['sim_threshold']
    hamming_threshold = exp_settings['hamming_threshold']
    assert (hamming_threshold == 0) or (hamming_threshold > 0 and 3*hamming_threshold < barcode_size)

    # Task Init
    # Example: 4 unique barcodes -> 16 total barcodes in epoch, 4 trials of each unique barcode
    episodes_per_epoch = num_barcodes**2

    task = ContextualBandit(
        pulls_per_episode, episodes_per_epoch,
        num_arms, num_barcodes, barcode_size,
        sim_threshold, hamming_threshold, device, perfect_info)

    # LSTM Chooses which arm to pull
    dim_output_lstm = num_arms
    dict_len = pulls_per_episode*(num_barcodes**2)
    value_weight = exp_settings['value_error_coef']
    entropy_weight = exp_settings['entropy_error_coef']
    
    # Input is obs/context/reward triplet
    dim_input_lstm = num_arms + barcode_size + 1
    dim_hidden_lstm = exp_settings['dim_hidden_lstm']
    learning_rate = exp_settings['lstm_learning_rate']

    # init agent / optimizer
    agent = Agent(dim_input_lstm, dim_hidden_lstm, dim_output_lstm,
                     dict_len, exp_settings, device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, agent.parameters()), lr=learning_rate)

    # Timing
    run_time = np.zeros(n_epochs,)

    # Results for TB or Graphing
    log_return = np.zeros(n_epochs,)
    log_embedder_accuracy = np.zeros(n_epochs,)
    log_loss_value = np.zeros(n_epochs,)
    log_loss_policy = np.zeros(n_epochs,)
    log_loss_total = np.zeros(n_epochs,)

    one_hot_action = torch.zeros((1,num_arms), dtype=torch.float32, device=device)

    print("\n", "-*-_-*- "*3, "\n")
    # loop over epoch
    for i in range(n_epochs):
        time_start = time.perf_counter()

        # get data for this epoch
        observations_barcodes_rewards, epoch_mapping, barcode_strings, barcode_tensors, barcode_id, arm_id = task.sample()
        agent.dnd.mapping = epoch_mapping

        # flush hippocampus
        agent.reset_memory()
        agent.turn_on_retrieval()

        # How much noise is needed?
        apply_noise = i-exp_settings['epochs']
        if apply_noise >= 0:
            noise_idx = apply_noise//exp_settings['noise_eval_epochs']
            noise_percent = exp_settings['noise_percent'][noise_idx]
            noise_barcode_flip_locs = int(noise_percent*barcode_size)

        # loop over the training set
        for m in range(episodes_per_epoch):

            # prealloc
            embedder_accuracy = 0
            cumulative_reward = 0
            probs, rewards, values, entropies = [], [], [], []
            h_t, c_t = agent.get_init_states()

            # Clearing the per trial hidden state buffer
            agent.flush_trial_buffer()

            # Noisy Barcodes are constant across an episode if needed
            if apply_noise >= 0:
                action = observations_barcodes_rewards[m][0][0:num_arms].view(1,-1)
                noisy_bc = observations_barcodes_rewards[m][0][num_arms:-1].view(1,-1)
                reward = observations_barcodes_rewards[m][0][-1].view(
                    1, -1)
                
                # What indicies need to be randomized?
                idx = torch.multinomial(noisy_bc, noise_barcode_flip_locs)

                # Do we flip the value at that index?
                mask = torch.randint_like(idx, 0, 2)

                # Applying the mask to the barcode
                for idx1, mask1 in zip(idx[0], mask[0]):
                    noisy_bc[0][idx1] = float(torch.ne(mask1,noisy_bc[0][idx1]))

                # Remake the input
                input_to_lstm = torch.cat(
                    (action, noisy_bc, reward.view(1,1)), dim=1)

            # loop over time, for one training example
            for t in range(pulls_per_episode):

                # only save memory at the last time point
                agent.turn_off_encoding()
                if t == pulls_per_episode-1 and m < episodes_per_epoch:
                    agent.turn_on_encoding()

                # First input when not noisy comes from task.sample
                if t == 0 and i < exp_settings['epochs']:
                    input_to_lstm = observations_barcodes_rewards[m]

                # Using the output action and reward of the last step of the LSTM as the next input
                else: #t != 0:
                    input_to_lstm = last_action_output

                    # Reset the one_hot var 
                    one_hot_action[0][a_t] = 0.0

                # What is being stored for Ritter?
                mem_key = barcode_tensors[m] if i < exp_settings['epochs'] else noisy_bc

                output_t, _ = agent(input_to_lstm, barcode_strings[m][0], 
                                        mem_key, barcode_id[m],
                                        h_t, c_t)
                a_t, assumed_barcode_string, prob_a_t, v_t, entropy, h_t, c_t = output_t

                # Always use ground truth bc for reward eval
                real_bc = barcode_strings[m][0][0]

                # compute immediate reward for actor network
                r_t = get_reward_from_assumed_barcode(a_t, real_bc, 
                                                        epoch_mapping, device, perfect_info)

                # Does the predicted context match the actual context?
                embedder_accuracy += int(real_bc == assumed_barcode_string)
                
                probs.append(prob_a_t)
                rewards.append(r_t)
                values.append(v_t)
                entropies.append(entropy)
                cumulative_reward += r_t

                # Inputs to LSTM come from predicted actions and rewards of last time step
                one_hot_action[0][a_t] = 1.0
                next_bc = barcode_tensors[m]

                # Add noise to the barcode at the right moments in experiment
                if i >= n_epochs - exp_settings['noise_eval_epochs']:
                    next_bc = noisy_bc

                # Create next input to feed back into LSTM
                last_action_output = torch.cat((one_hot_action, next_bc, r_t.view(1,1)), dim = 1)

            # LSTM/A2C Loss for Episode
            returns = compute_returns(rewards, device, gamma = 0.0)
            loss_policy, loss_value, entropies_tensor = compute_a2c_loss(probs, values, returns, entropies)
            loss = loss_policy + value_weight*loss_value - entropy_weight*entropies_tensor

            # Only perform model updates during train phase
            if apply_noise < 0:

                if exp_settings['mem_store'] == 'embedding':
                    # Embedder Loss for Episode
                    a_dnd = agent.dnd
                    loss_vals = [x[2] for x in a_dnd.trial_buffer]
                    episode_loss = torch.stack(loss_vals).mean()
                    a_dnd.embedder_loss[i] += (episode_loss/episodes_per_epoch)

                    # Unfreeze Embedder
                    for name, param in a_dnd.embedder.named_parameters():
                        if param.requires_grad:
                            print(name, param.grad)
                        param.requires_grad = True

                    # Freeze LSTM/A2C
                    layers = [agent.i2h, agent.h2h, agent.a2c]
                    for layer in layers:
                        for name, param in layer.named_parameters():
                            param.requires_grad = False 

                    # Embedder Backprop
                    a_dnd.embed_optimizer.zero_grad()
                    episode_loss.backward(retain_graph=True)
                    a_dnd.embed_optimizer.step()
                    a_dnd.embed_optimizer.zero_grad()

                    # Freeze Embedder until next memory retrieval
                    for name, param in a_dnd.embedder.named_parameters():
                        param.requires_grad = False
                    
                    # Unfreeze LSTM/A2C
                    for layer in layers:
                        for name, param in layer.named_parameters():
                            param.requires_grad = True 

                # LSTM and A2C Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            # Updating avg return per episode
            log_return[i] += torch.div(cumulative_reward, (episodes_per_epoch*pulls_per_episode))
           
            # Updating avg accuracy per episode
            log_embedder_accuracy[i] += torch.div(embedder_accuracy, (episodes_per_epoch*pulls_per_episode))
            
            # Loss Logging
            log_loss_value[i] += torch.div(loss_value, episodes_per_epoch)
            log_loss_policy[i] += torch.div(loss_policy, episodes_per_epoch)
            log_loss_total[i] += torch.div(loss, episodes_per_epoch)

        # Tensorboard Stuff
        if exp_settings['tensorboard_logging'] and i%5 == 4:
            tb.add_scalar("LSTM Loss_All", log_loss_total[i], i)
            tb.add_scalar("LSTM Loss_Policy", log_loss_policy[i], i)
            tb.add_scalar("LSTM Loss_Value", log_loss_value[i], i)
            tb.add_scalar("LSTM Returns", log_return[i], i)
            if exp_settings['mem_store'] == 'embedding':
                tb.add_scalar("Embedder Loss",
                            agent.dnd.embedder_loss[i], i)
                tb.add_scalar("Accuracy Embedder Model",
                                log_embedder_accuracy[i], i)

        run_time[i] = time.perf_counter() - time_start

        # Print reports every 10% of the total number of epochs
        if i%(int(n_epochs/10)) == 0 or i == n_epochs-1:
            print(
                'Epoch %3d | avg_return = %.2f | loss: val = %.2f, pol = %.2f, tot = %.2f | time = %.2f'%
                (i, log_return[i], log_loss_value[i], log_loss_policy[i], log_loss_total[i], run_time[i])
            )
            # Accuracy over the last 10 epochs
            if  i > 10:
                avg_acc = log_embedder_accuracy[i-9:i+1].mean()
            else:
                avg_acc = log_embedder_accuracy[:i+1].mean()
            print("\tEmbedder Accuracy: ", round(avg_acc, 4), end = ' | ')
            print("Ritter Baseline: ", round(1-1/exp_settings['num_barcodes'], 4))

        # Store the keys from just before the noise begins
        if i == exp_settings['epochs'] - 1:
            keys, prediction_mapping = agent.get_all_mems_embedder()
    
    # Final Results
    print("- - - "*3)
    final_q = 3*(exp_settings['epochs']//4)
    print("Last Quarter Return Avg: ", round(np.mean(log_return[final_q:]), 3))
    print("Total Time Elapsed:", round(sum(run_time), 1), "secs")
    print("Avg Epoch Time:", round(np.mean(run_time), 2), "secs")
    print("- - - "*3)

    logs = log_return, log_loss_value, log_loss_policy, log_loss_total, log_embedder_accuracy, agent.dnd.embedder_loss
    key_data = keys, prediction_mapping, epoch_mapping

    if exp_settings['tensorboard_logging']:
        tb.flush()
        tb.close()

    return  logs, key_data

### Graphing Helper Functions ###
# Theoretical Min/Max Return Performance
def expected_return(num_arms, perfect_info):
    if not perfect_info:
        perfect = 0.9
        random = 0.9*(1/num_arms) + 0.1*(num_arms-1)/num_arms
    else:
        perfect = 1
        random = 1/num_arms
    return perfect, random

# Adapted from https://learnopencv.com/t-sne-for-feature-visualization/
def scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range

def plot_tsne_distribution(keys, labels, mapping, fig, axes, idx_mem):
    features = np.array([y.cpu().numpy() for y in keys])
    tsne = TSNE(n_components=2).fit_transform(features)
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # Seperate by barcode
    classes = {k:[] for k in mapping.keys()}
    for idx, c_id in enumerate(labels):
        classes[c_id].append(idx)

    marker_list = ['x', '1', 'o', '*', '2', 'v', '.', '^','3', '<', '>', '4', '+', 's']
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    assert len(classes) <= len(marker_list), "Too many distinct barcodes to display with current selection of labels"
    assert max(list(mapping.values())) <= len(color_list), "Too many distinct arms to display with current selection of colors"

    # Map each barcode as a seperate layer on the same scatterplot
    for m_id, (c_id, indices) in enumerate(classes.items()):
        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # Identify the arm of the barcode
        arm = mapping[c_id]

        # Graph arms by color and barcodes by marker
        axes[idx_mem].scatter(current_tx, current_ty, c = color_list[arm], marker = marker_list[m_id])

    return fig, axes

def run_experiment(exp_base, exp_difficulty):

    exp_settings = {}

    ### Hyperparams in BayesOpt ###
    # Set in get_hyperparams function, below values are placeholders
    exp_settings['dim_hidden_a2c'] = 0
    exp_settings['dim_hidden_lstm'] = 0
    exp_settings['entropy_error_coef'] = 0
    exp_settings['lstm_learning_rate'] = 0
    exp_settings['value_error_coef'] = 0
    exp_settings['embedding_size'] = 0
    exp_settings['embedder_learning_rate'] = 0
    ### End Hyperparams in BayesOpt ###

    ### Experimental Parameters ###
    exp_settings['randomize'] = True
    exp_settings['perfect_info'] = False            #Make arms 100%/0% reward instead of 90%/10%
    exp_settings['torch_device'] = 'CPU'            # 'CPU' or 'GPU'

    # Task Info
    exp_settings['kernel'] = 'cosine'               # Cosine, l2
    exp_settings['mem_store'] = 'context'           # Context, embedding, obs/context, obs, hidden (unsure how to do obs, hidden return calc w/o barcode predictions)

    # Task Size and Length
    exp_settings['num_arms'] = 0
    exp_settings['barcode_size'] = 0
    exp_settings['num_barcodes'] = 0
    exp_settings['pulls_per_episode'] = 0
    exp_settings['epochs'] = 0

    # Task Complexity
    exp_settings['noise_percent'] = [0.25, 0.5, 0.75, 0.875]
    exp_settings['noise_eval_epochs'] = 0
    exp_settings['sim_threshold'] = 0         #Cosine similarity threshold for clustering
    exp_settings['hamming_threshold'] = 0       #Hamming distance for clustering

    # Data Logging
    exp_settings['tensorboard_logging'] = False
    ### End of Experimental Parameters ###


    # Forced Hyperparams
    exp_settings['torch_device'] = 'GPU'
    exp_settings['dim_hidden_a2c'] = int(2**8.644)        #400
    exp_settings['dim_hidden_lstm'] = int(2**8.655)       #403
    exp_settings['entropy_error_coef'] = 0.0391
    exp_settings['lstm_learning_rate'] = 10**-3.332       #4.66e-4
    exp_settings['value_error_coef'] = 0.62
    exp_settings['embedding_size'] = int(2**8.629)          #395
    exp_settings['embedder_learning_rate'] = 10**-3.0399    #9.1e-4

    # Experimental Variables
    mem_store_types, exp_settings['epochs'], exp_settings['noise_eval_epochs'], num_repeats, file_loc = exp_base
    exp_settings['hamming_threshold'], exp_settings['num_arms'], exp_settings['num_barcodes'], exp_settings[
        'barcode_size'], exp_settings['pulls_per_episode'], exp_settings['sim_threshold'] = exp_difficulty

    # Safety Assertions
    assert exp_settings['epochs'] > 10, "Training epochs must be greater than 10"
    assert exp_settings['pulls_per_episode'] > 2, "Pulls per episode must be greater than 2"
    assert exp_settings['barcode_size'] > 3*exp_settings['hamming_threshold'], "Barcodes must be greater than 3*Hamming"

    ### Beginning of Experimental Runs ###
    f, axes = plt.subplots(1, 1, figsize=(8, 6))
    f1, axs = plt.subplots(1, 1, figsize=(8, 6))
    f3, axes3 = plt.subplots(1, len(mem_store_types), figsize=(3*len(mem_store_types), 6))

    for idx_mem, mem_store in enumerate(mem_store_types):
        tot_rets = np.zeros(exp_settings['epochs']+exp_settings['noise_eval_epochs']*len(exp_settings['noise_percent']))
        exp_settings['mem_store'] = mem_store
        for i in range(num_repeats):

            print(f"\nNew Run --> Iteration: {i} | Type: {mem_store} | Device: {exp_settings['torch_device']}")
            logs, key_data = run_experiment_sl(exp_settings)
            log_return, log_loss_value, log_loss_policy, log_loss_total, log_embedder_accuracy, embedder_loss = logs
            keys, prediction_mapping, epoch_mapping = key_data 
            tot_rets += log_return/num_repeats
            # print(tot_rets)

        smoothed_rewards = pd.Series.rolling(pd.Series(tot_rets), 30).mean()
        smoothed_rewards = [elem for elem in smoothed_rewards]
        axes.plot(smoothed_rewards, label=f"Mem: {mem_store}")

        # Embedder/Mem Accuracy 
        smoothed_accuracy = pd.Series.rolling(pd.Series(log_embedder_accuracy), 30).mean()
        smoothed_accuracy = [elem for elem in smoothed_accuracy]
        axs.plot(smoothed_accuracy, label=f"Mem: {mem_store}")

        # T-SNE to visualize keys in memory
        embeddings = [x[0] for x in keys]
        labels = [x[1] for x in keys]

        # Artifically boost datapoint count to make tsne nicer
        while len(embeddings) < 100:
            embeddings.extend(embeddings)
            labels.extend(labels)

        f3, axes3 = plot_tsne_distribution(embeddings, labels, epoch_mapping, f3, axes3, idx_mem)
        axes3[idx_mem].xaxis.set_visible(False)
        axes3[idx_mem].yaxis.set_visible(False)
        axes3[idx_mem].set_title(mem_store)

    # Graph Setup
    if exp_settings['hamming_threshold']:
        cluster_info = f"IntraCluster Dist: {exp_settings['hamming_threshold']} | InterCluster Dist: {exp_settings['barcode_size']-2*exp_settings['hamming_threshold']}"
    else:
        cluster_info = f"Similarity: {exp_settings['sim_threshold']}" 

    graph_title = f""" --- Returns ---
    Arms: {exp_settings['num_arms']} | Unique Barcodes: {exp_settings['num_barcodes']} | Barcode Dim: {exp_settings['barcode_size']}
    Pulls per Trial: {exp_settings['pulls_per_episode']} | Clusters: {int(exp_settings['num_barcodes']/exp_settings['num_arms'])}
    {cluster_info}
    """

    # Returns
    # perfect_ret, random_ret = expected_return(exp_settings['num_arms'], exp_settings['perfect_info'])
    # axes.axhline(y=random_ret, color='b', linestyle='dashed', label = 'Random Pulls')
    # axes.axhline(y=perfect_ret, color='k', linestyle='dashed', label = 'Theoretical Max')
    axes.set_ylabel('Returns')
    axes.set_xlabel('Epoch')

    # Noise Partitions
    colors = ['g', 'r', 'c', 'm']
    for idx, noise_percent in enumerate(exp_settings['noise_percent']):
        axes.axvline(x=exp_settings['epochs'] + idx*exp_settings['noise_eval_epochs'], color=colors[idx], linestyle = 'dashed',
            label = f"{int(exp_settings['barcode_size']*noise_percent)} Bits Noisy")
        axs.axvline(x=exp_settings['epochs'] + idx*exp_settings['noise_eval_epochs'], color=colors[idx], linestyle='dashed',
            label = f"{int(exp_settings['barcode_size']*noise_percent)} Bits Noisy")

    # Accuracy
    axs.set_ylabel('Accuracy')
    axs.set_xlabel('Epoch')
    axs.set_title('Model Barcode Prediction Accuracy')
    axs.axhline(y=1/exp_settings['num_barcodes'], color='b', linestyle='dashed', label = 'Random Choice')

    sns.despine()
    axes.legend(bbox_to_anchor=(0, -0.2, 1, 0), loc="upper left",
            mode="expand", borderaxespad=0, ncol=3)
    f.tight_layout()
    f.subplots_adjust(top=0.8)
    f.suptitle(graph_title)
    axs.legend(bbox_to_anchor=(0, -0.2, 1, 0), loc="upper left",
            mode="expand", borderaxespad=0, ncol=2)
    f1.tight_layout()
    f3.tight_layout()
    f3.subplots_adjust(top=0.8)
    f3.suptitle("t-SNE on keys in memory from last training epoch\nIcon indicates barcode, color is best arm choice")

    # Graph Saving
    file_loc = file_loc
    exp_id = f"{exp_settings['num_arms']}a{exp_settings['num_barcodes']}b{exp_settings['pulls_per_episode']}p {exp_settings['hamming_threshold']} hamming {num_repeats} run(s) "
    plot_type = ['returns', 'accuracy', 'tsne']
    for fig_num, figa in enumerate([f, f1, f3]):
        filename = file_loc + exp_id + plot_type[fig_num] +".png"
        figa.savefig(filename)

    plt.show()