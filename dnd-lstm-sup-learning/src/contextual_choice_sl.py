"""demo: train a DND LSTM on a contextual choice task
"""
# Win64bit Optimizations for TSNE
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.manifold import TSNE
import itertools
import time
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as LRScheduler

from task.ContextualChoice import ContextualChoice
from task.ContextBandits import ContextualBandit
from sl_model import DNDLSTM as Agent
from utils import compute_stats, to_sqnp
from sl_model.DND import compute_similarities
from sl_model.utils import get_reward, get_reward_from_assumed_barcode, compute_returns, compute_a2c_loss

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

    n_epochs = exp_settings['epochs'] + exp_settings['noise_eval_epochs']*len(exp_settings['noise_percent'])
    agent_input = exp_settings['agent_input']
   
    '''init task'''
    # input/output/hidden/memory dim
    num_arms = exp_settings['num_arms']             # LSTM input dim
    barcode_size = exp_settings['barcode_size']     # Possible additional LSTM input dim
    num_barcodes = exp_settings['num_barcodes']     # Different number of contexts

    # Do we need a new set of contexts for each epoch, or do we keep it constant
    reset_barcodes_per_epoch = exp_settings['reset_barcodes_per_epoch']

    # Do we shuffle the arm assignments while keeping the barcodes constant?
    # Obvi don't have this and reset_barcodes true at the same time.
    reset_arms_per_epoch = exp_settings['reset_arms_per_epoch']

    # Arm pulls per single barcode episode
    pulls_per_episode = exp_settings['pulls_per_episode']

    # Arm rewards can be deterministic for debugging
    perfect_info = exp_settings['perfect_info']

    # Cluster barcodes at the start (Only use one per experiment)
    sim_threshold = exp_settings['sim_threshold']
    hamming_threshold = exp_settings['hamming_threshold']
    assert (hamming_threshold == 0) or (hamming_threshold > 0 and 3*hamming_threshold < barcode_size)

    # Task Choice
    if exp_settings['task_version'] == 'bandit':
        # Example: 4 unique barcodes -> 16 total barcodes in epoch, 4 trials of each unique barcode
        episodes_per_epoch = num_barcodes**2

        task = ContextualBandit(
            pulls_per_episode, episodes_per_epoch,
            num_arms, num_barcodes, barcode_size,
            reset_barcodes_per_epoch, reset_arms_per_epoch,
            sim_threshold, hamming_threshold, device, perfect_info)

        # LSTM Chooses which arm to pull
        dim_output_lstm = num_arms
        dict_len = pulls_per_episode*(num_barcodes**2)
        value_weight = exp_settings['value_error_coef']
        entropy_weight = exp_settings['entropy_error_coef']

    elif exp_settings['task_version'] == 'original':
        task = ContextualChoice(
            num_arms, num_barcodes, 0.5)
        episodes_per_epoch = 2*num_barcodes
        epoch_mapping = {}

        # LSTM indicates if this is a 0 or a 1 task
        dim_output_lstm = 2
        dict_len = 50
        value_weight = 1
        entropy_weight = 0
    
    # Input to LSTM is only observation + reward
    if agent_input == 'obs':
        dim_input_lstm = num_arms + 1
    
    # Input is obs/context/reward triplet
    else:  # agent_input == 'obs/context'
        dim_input_lstm = num_arms + barcode_size + 1

    dim_hidden_lstm = exp_settings['dim_hidden_lstm']
    learning_rate = exp_settings['lstm_learning_rate']

    # init agent / optimizer
    agent = Agent(dim_input_lstm, dim_hidden_lstm, dim_output_lstm,
                     dict_len, exp_settings, device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, agent.parameters()), lr=learning_rate)
    # scheduler = LRScheduler.OneCycleLR(
    #     optimizer, max_lr=learning_rate, steps_per_epoch=episodes_per_epoch, epochs=n_epochs)

    # Timing
    run_time = np.zeros(n_epochs,)
    task_cumulative, pull_cumulative, episode_cumulative, end_cumulative = 0,0,0,0
    pull_timings, episode_timings = {}, {}

    # Results for TB or Graphing
    log_return = np.zeros(n_epochs,)
    log_embedder_accuracy = np.zeros(n_epochs,)
    log_loss_value = np.zeros(n_epochs,)
    log_loss_policy = np.zeros(n_epochs,)
    log_loss_total = np.zeros(n_epochs,)

    # Legacy, might need to check over spread of action choices?
    log_Y = np.zeros((n_epochs, episodes_per_epoch, pulls_per_episode))
    log_Y_hat = np.zeros((n_epochs, episodes_per_epoch, pulls_per_episode))

    barcode_data = [[[]]]
    one_hot_action = torch.zeros((1,num_arms), dtype=torch.float32, device=device)

    print("\n", "-*-_-*- "*3, "\n")
    # loop over epoch
    for i in range(n_epochs):
        time_start = time.perf_counter()
        if i < n_epochs - 1:
            barcode_data.append([[]])

        # Barcode to arm mapping can be changed per epoch, or kept constant
        # see exp_settings['reset_barcodes_per_epoch']
        # get data for this epoch
        if exp_settings['task_version'] == 'bandit':
            observations_barcodes_rewards, epoch_mapping, barcode_strings, barcode_tensors, barcode_id, arm_id = task.sample()
            agent.dnd.mapping = epoch_mapping
            # print(observations_barcodes)

        elif exp_settings['task_version'] == 'original':
            X, Y, contexts = task.sample(num_barcodes)
        
        if exp_settings['timing']:
            task_cumulative += (time.perf_counter() - time_start)

        # flush hippocampus
        agent.reset_memory()
        agent.turn_on_retrieval()
        # print(sorted(list(epoch_mapping.items())))

        # How much noise is needed?
        apply_noise = i-exp_settings['epochs']
        if apply_noise >= 0:
            noise_idx = apply_noise//exp_settings['noise_eval_epochs']
            noise_percent = exp_settings['noise_percent'][noise_idx]
            noise_barcode_flip_locs = int(noise_percent*barcode_size)

        # loop over the training set
        for m in range(episodes_per_epoch):

            if exp_settings['timing']:
                episode_start = time.perf_counter()

            # prealloc
            embedder_accuracy = 0
            cumulative_reward = 0
            probs, rewards, values, entropies = [], [], [], []
            h_t, c_t = agent.get_init_states()
            if m < episodes_per_epoch - 1:
                barcode_data[i].append([])

            # # Tensorboard the graphs
            # # There are some weird branching logic problems which are preventing this from working right
            # if i == 0 and m == 0:
            #     test_input = observations_barcodes[0][0].view(1, 1, -1)
            #     print(test_input)
            #     h_test, c_test = agent.get_init_states()
            #     inputs = (test_input, h_test, c_test)
            #     tb.add_graph(agent, input_to_model=inputs)
                
            #     # scripted_graph = torch.jit.script(agent, example_inputs=inputs)
            #     # print(scripted_graph)

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
                # print(f"BC Before: {noisy_bc}")
                for idx1, mask1 in zip(idx[0], mask[0]):
                    noisy_bc[0][idx1] = float(int(torch.ne(mask1,noisy_bc[0][idx1])))
                # print(f"BC After: {noisy_bc}")

                # print(compute_similarities(noisy_bc[0], [barcode_tensors[m][0]], metric = 'cosine'))

                # Remake the input
                input_to_lstm = torch.cat(
                    (action, noisy_bc, reward.view(1,1)), dim=1)

            # loop over time, for one training example
            for t in range(pulls_per_episode):
                
                if exp_settings['timing']:
                    pull_start = time.perf_counter()

                # only save memory at the last time point
                agent.turn_off_encoding()
                if t == pulls_per_episode-1 and m < episodes_per_epoch:
                    agent.turn_on_encoding()

                if exp_settings['task_version'] == 'bandit':
                    # Looping LSTM inputs means only the first pull of episode is defined outside model
                    if exp_settings['lstm_inputs_looped']:

                        # First input when not noisy comes from task.sample
                        if t == 0 and i < exp_settings['epochs']:
                            input_to_lstm = observations_barcodes_rewards[m]

                        # Using the output action and reward of the LSTM as the next input
                        else: #t != 0:
                            input_to_lstm = last_action_output

                            # Reset the one_hot var 
                            one_hot_action[0][a_t] = 0.0

                    if exp_settings['embedder_arm_trained']:
                        memory_loss_id = arm_id[m]
                    else:
                        memory_loss_id = barcode_id[m]

                    mem_key = barcode_tensors[m] if i < exp_settings['epochs'] else noisy_bc

                    output_t, cache = agent(input_to_lstm, barcode_strings[m][0], 
                                            mem_key, memory_loss_id,
                                            h_t, c_t)
                    a_t, assumed_barcode_string, prob_a_t, v_t, entropy, h_t, c_t = output_t
                    _, _, _, _, _, timings = cache

                    pull_overhead_start = time.perf_counter()

                    # Ritter uses BC from memory, Embedding using bc from embedding through model predictor
                    # Always use ground truth bc for reward eval
                    real_bc = barcode_strings[m][0][0]

                    # compute immediate reward for actor network
                    r_t = get_reward_from_assumed_barcode(a_t, real_bc, 
                                                            epoch_mapping, device, perfect_info)

                    # Does the predicted context match the actual context?
                    if exp_settings['embedder_arm_trained']:
                        # Training the embedder on arm choice overloads the assumed_barcode_string
                        #  return with the arm prediction for the embedder
                        if exp_settings['mem_store'] == 'embedding':
                            try:
                                real_arm = epoch_mapping[real_bc]
                                assumed_arm = assumed_barcode_string
                                embedder_accuracy += int(real_arm == assumed_arm)
                            except KeyError: #returning empty barcodes at beginning of episode
                                #Yes i know this is dumb, but it's mostly a reminder to me as to what is happening here
                                embedder_accuracy += 0
                        else:
                            embedder_accuracy += int(real_bc == assumed_barcode_string)
                    else:
                        embedder_accuracy += int(real_bc == assumed_barcode_string)
                    # print(real_bc, assumed_barcode_string)

                    # # Confusion Matrix for Embedder Predictions
                    # if exp_settings['mem_store'] == 'embedding' and i == n_epochs - 1:
                    #     barcode_data[i][m].append((real_bc, assumed_barcode_string))

                elif exp_settings['task_version'] == 'original':
                    # print(X[m][t])
                    output_t, _ = agent(X[m][t].view(1, 1, -1), "", None,
                                            h_t, c_t)
                    a_t, assumed_barcode_string, prob_a_t, v_t, entropy, h_t, c_t = output_t
                    
                    # compute immediate reward for actor network
                    r_t = get_reward(a_t, Y[m][t])
                
                probs.append(prob_a_t)
                rewards.append(r_t)
                values.append(v_t)
                entropies.append(entropy)
                cumulative_reward += r_t
                # log_Y_hat[i, m, t] = a_t.item()
                
                if exp_settings['timing']:
                    timings['1g. Rewards'] = time.perf_counter() - pull_overhead_start 
                    input_loop_start = time.perf_counter() 

                # Inputs to LSTM come from predicted actions and rewards of last time step
                if exp_settings['lstm_inputs_looped']:
                    one_hot_action[0][a_t] = 1.0
                    next_bc = barcode_tensors[m]

                    # Add noise to the barcode at the right moments in experiment
                    if i >= n_epochs - exp_settings['noise_eval_epochs']:
                        next_bc = noisy_bc

                    # Create next input to feed back into LSTM
                    last_action_output = torch.cat((one_hot_action, next_bc, r_t.view(1,1)), dim = 1)

                if exp_settings['timing']:
                    timings['1h. Looped Inputs'] = time.perf_counter() - input_loop_start
                    for k,v in timings.items():
                        pull_timings[k] = pull_timings.get(k,0) + v
                    pull_cumulative += (time.perf_counter() - pull_start)
                    
            # print("-- end of ep --")
            episode_overhead_start = time.perf_counter()

            # LSTM/A2C Loss for Episode
            returns = compute_returns(rewards, device, gamma = 0.0)
            loss_policy, loss_value, entropies_tensor = compute_a2c_loss(probs, values, returns, entropies)
            loss = loss_policy + value_weight*loss_value - entropy_weight*entropies_tensor

            if exp_settings['timing']:
                loss_time = time.perf_counter() - episode_overhead_start

            # Only perform model updates during train phase
            if apply_noise < 0:

                 # Embedder Loss for Episode
                if exp_settings['mem_store'] == 'embedding':
                    a_dnd = agent.dnd
                    loss_vals = [x[2] for x in a_dnd.trial_buffer]
                    episode_loss = torch.stack(loss_vals).mean()
                    # print("EmbLoss:", episode_loss)
                    a_dnd.embedder_loss[i] += (episode_loss/episodes_per_epoch)

                # # Testing for gradient leaks between embedder model and lstm model
                # print("Before-a2c o:\n", agent.a2c.critic.weight)
                # print("Before-emb o:\n", agent.dnd.embedder.e2c.weight)

                # Embedder Backprop
                if exp_settings['mem_store'] == 'embedding':
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

                    a_dnd.embed_optimizer.zero_grad()
                    episode_loss.backward(retain_graph=True)
                    a_dnd.embed_optimizer.step()
                    a_dnd.embed_optimizer.zero_grad()

                    # Freeze Embedder until next memory retrieval
                    for name, param in a_dnd.embedder.named_parameters():
                        # print(name, param.grad)
                        param.requires_grad = False
                    
                    # Unfreeze LSTM/A2C
                    for layer in layers:
                        for name, param in layer.named_parameters():
                            param.requires_grad = True 

                # # Testing for gradient leaks between embedder model and lstm model
                # print("Before-a2c o:\n", agent.a2c.critic.weight)
                # print("After-emb o:\n", agent.dnd.embedder.e2c.weight)

                # LSTM and A2C Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # # Testing for gradient leaks between embedder model and lstm model
                # print("After-a2c l:\n", agent.a2c.critic.weight)
                # print("After-emb l:\n", agent.dnd.embedder.e2c.weight)

                # # Learning Rate Scheduler
                # scheduler.step()

            if exp_settings['timing']:
                backprop_time = time.perf_counter() - episode_overhead_start - loss_time

            # log
            # if exp_settings['task_version'] == 'bandit':
            #     log_Y[i] = np.squeeze(reward_from_obs.cpu().numpy())
            # elif exp_settings['task_version'] == 'original':
            #    log_Y[i] += np.squeeze(Y[m][t].cpu().numpy())
            
            # Updating avg return per episode
            log_return[i] += torch.div(cumulative_reward, (episodes_per_epoch*pulls_per_episode))
           
            # Updating avg accuracy per episode
            log_embedder_accuracy[i] += torch.div(embedder_accuracy, (episodes_per_epoch*pulls_per_episode))
            
            # Loss Logging
            log_loss_value[i] += torch.div(loss_value, episodes_per_epoch)
            log_loss_policy[i] += torch.div(loss_policy, episodes_per_epoch)
            log_loss_total[i] += torch.div(loss, episodes_per_epoch)

            if exp_settings['timing']:
                logging_time = time.perf_counter() - episode_overhead_start - loss_time - backprop_time
                ep_timings = {"2b1. Loss": loss_time, "2b2. Backprop": backprop_time, "2b3. Logging": logging_time}
                for k,v in ep_timings.items():
                    episode_timings[k] = episode_timings.get(k,0) + v
                episode_cumulative += (time.perf_counter() - episode_start)

        # Tensorboard Stuff
        tb_start = time.perf_counter() 
        if exp_settings['tensorboard_logging'] and i%5 == 4:
            tb.add_scalar("LSTM Loss_All", log_loss_total[i], i)
            tb.add_scalar("LSTM Loss_Policy", log_loss_policy[i], i)
            tb.add_scalar("LSTM Loss_Value", log_loss_value[i], i)
            tb.add_scalar("LSTM Returns", log_return[i], i)

            # for name, weight in agent.named_parameters():
            #     tb.add_histogram(name, weight, i)
            #     try:
            #         tb.add_histogram(f'{name}.grad', weight.grad, i)
            #         tb.add_histogram(f'{name}_grad_norm', weight.grad.norm(), i)
            #     except Exception:
            #         continue

            if exp_settings['mem_store'] == 'embedding':
                tb.add_scalar("Embedder Loss",
                            agent.dnd.embedder_loss[i], i)
                tb.add_scalar("Accuracy Embedder Model",
                                log_embedder_accuracy[i], i)
                # for name, weight in agent.dnd.embedder.named_parameters():
                #     tb.add_histogram(name, weight, i)
                #     try:
                #         tb.add_histogram(f'{name}.grad', weight.grad, i)
                #         tb.add_histogram(f'{name}_grad_norm', weight.grad.norm(), i)
                #     except Exception:
                #         continue

        if exp_settings['timing']:
            end_cumulative += (time.perf_counter() - tb_start)

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

    # Time Logging
    if exp_settings['timing']:
        tot_epis = episodes_per_epoch*n_epochs
        tot_pulls = pulls_per_episode*tot_epis
        for k in pull_timings:
            if "Save" in k:
                pull_timings[k]/=tot_epis
            else:
                pull_timings[k]/=tot_pulls
        for k in episode_timings:
            episode_timings[k]/=tot_epis
        epoch = np.mean(run_time)
        epi = episode_cumulative/(tot_epis)
        pull = pull_cumulative/(tot_pulls)
        task_init = task_cumulative/n_epochs
        end_of_epoch = end_cumulative/(n_epochs/5)
        time_outs = {   
                        "1. Pull": (pull, 100*pull/pull),
                        "2. Episode Actual": (epi, 100*epi/epi),
                        "2a. All Pulls": (pull*pulls_per_episode, 100*pull*pulls_per_episode/epi),
                        "2b. Episode Overhead": (epi-pull*pulls_per_episode, 100*(epi-pull*pulls_per_episode)/epi),
                        "3. Epoch Actual": (epoch, 100*epoch/epoch),
                        "3a. All Episodes": (epi*episodes_per_epoch, 100*epi*episodes_per_epoch/epoch),
                        "3b. Epoch Overhead": (epoch - epi*episodes_per_epoch, 100*(epoch - epi*episodes_per_epoch)/epoch),
                        "3b1. Task Init": (task_init, 100*task_init/epoch),
                        "3b2. TB (every 5 epochs)": (end_of_epoch, 100*(end_of_epoch/5)/epoch)
                    }

        # Percent of total time per task
        for k,v in pull_timings.items():
            pull_timings[k] = [v, 100*v/pull]
        for k,v in episode_timings.items():
            episode_timings[k] = [v, 100*v/epi]
        
        # Merge all the dictionaries
        time_out = time_outs | pull_timings | episode_timings
        time_out = {key:(round(1000*time_out[key][0], 2), round(time_out[key][1], 1)) for key in time_out}
        timing_df = pd.DataFrame.from_dict(time_out, orient='index', columns = ['Avg Time (ms)', 'Percent of Total']).sort_index()

        print("\nIndividual Part Avg Times:\n")
        print(timing_df)
    print("- - - "*3)

    logs = log_return, log_loss_value, log_loss_policy, log_loss_total, log_embedder_accuracy, agent.dnd.embedder_loss
    key_data = keys, prediction_mapping, epoch_mapping, barcode_data

    if exp_settings['tensorboard_logging']:
        tb.flush()
        tb.close()

    return  logs, key_data

# Graphing Helper Functions
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

def plot_tsne_distribution(keys, labels, arms, mapping, fig, axes, idx_mem):
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
    for m_id, (c_id, indices) in enumerate(classes.items()):
        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # Identify the arm of the barcode
        arm = mapping[c_id]

        # Graph arms by color and barcodes by marker
        axes[idx_mem].scatter(current_tx, current_ty, c = color_list[arm], marker = marker_list[m_id])

    return fig, axes

def get_barcode(mem_id, prediction_mapping):
    for k, v in prediction_mapping.items():
        if v == mem_id:
            return k
    print('Invalid mem_id:', mem_id)
    return ''

def make_confusion_matrix(mapping, barcode_data):
    data = np.zeros((len(mapping), len(mapping)))
    barcode_id_list = sorted(list(mapping.keys()))
    count = 0
    for barcode_data_ep in barcode_data:
     for barcode_data_epi in barcode_data_ep:
        for real, predicted in barcode_data_epi:
            real_id, pred_id = get_barcode_ids(
                barcode_id_list, real, predicted)
            data[real_id][pred_id] += 1
            count += 1
    data = data/count

    # Heatmap of real_id, pred_id
    ax = sns.heatmap(data, annot=True, cmap='Blues')
    ax.set_title('Confusion Matrix for Barcodes\n\n')
    ax.set_xlabel('\nPredicted Barcode')
    ax.set_ylabel('Actual Barcode')

    ids = [x for x in range(len(barcode_id_list))]
    ax.xaxis.set_ticklabels(ids)
    ax.yaxis.set_ticklabels(ids)
    return ax

def get_barcode_ids(barcode_id_list, real, predicted):
    real_id = barcode_id_list.index(real)
    pred_id = barcode_id_list.index(predicted)
    return real_id, pred_id

def get_hyperparams(mem_store, num_arms, exp_settings):
        """
        Quick access to best hyperparams as found by bayesian_opt.py

        Args:
            mem_store (String): What is being stored in memory as a key
            num_arms (Int): How many distinct arms are being pulled
            exp_settings (Dict): All changeable values for experiment

        Returns:
            exp_settings: All changeable values for experiment
        """
        exp_settings['num_arms'] = num_arms
        exp_settings['barcode_size'] = num_arms
        exp_settings['num_barcodes'] = num_arms
        exp_settings['mem_store'] = mem_store

        if mem_store == 'context':
            exp_settings['torch_device'] = 'CPU'

            # Smaller network works for Ritter well, but has early learning peak climb
            if num_arms != 10:
                exp_settings['dim_hidden_a2c'] = int(2**6.909)        #120
                exp_settings['dim_hidden_lstm'] = int(2**5.302)       #39
                exp_settings['entropy_error_coef'] = 0.0641
                exp_settings['lstm_learning_rate'] = 10**-2.668       #2.1e-3
                exp_settings['value_error_coef'] = 0.335

            elif num_arms == 10:
                exp_settings['dim_hidden_a2c'] = int(2**8.508)          #364
                exp_settings['dim_hidden_lstm'] = int(2**6.966)         #125
                exp_settings['entropy_error_coef'] = 0.0544
                exp_settings['lstm_learning_rate'] = 10**-2.852         #1.4e-3
                exp_settings['value_error_coef'] = 0.2767
            
        # Need to do more Bayes passes to find if this can be made better for embedder
        elif mem_store == 'embedding':
            exp_settings['torch_device'] = 'GPU'
            exp_settings['dim_hidden_a2c'] = int(2**8.644)        #400
            exp_settings['dim_hidden_lstm'] = int(2**8.655)       #403
            exp_settings['entropy_error_coef'] = 0.0391
            exp_settings['lstm_learning_rate'] = 10**-3.332       #4.66e-4
            exp_settings['value_error_coef'] = 0.62

            # Embedder Model Info (optimized for 4arms, but can be used for 10arms as well)
            # Need to investigate if there is a better embedding size/structure at a later point
            exp_settings['embedding_size'] = int(2**8.629)          #395
            exp_settings['embedder_learning_rate'] = 10**-3.0399    #9.1e-4
        return exp_settings

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
    exp_settings['reset_barcodes_per_epoch'] = False
    exp_settings['reset_arms_per_epoch'] = True
    exp_settings['lstm_inputs_looped'] = True       # Use action predictions from lstm as next input, instead of predetermined pulls
    exp_settings['torch_device'] = 'CPU'            # 'CPU' or 'GPU'

    # Task Info
    exp_settings['kernel'] = 'cosine'               # Cosine, l2
    exp_settings['agent_input'] = 'obs/context'     # Obs, obs/context
    exp_settings['mem_store'] = 'context'           # Context, embedding, obs/context, obs, hidden (unsure how to do obs, hidden return calc w/o barcode predictions)
    exp_settings['task_version'] = 'bandit'         # Bandit, original

    # Task Size and Length
    exp_settings['num_arms'] = 0
    exp_settings['barcode_size'] = 0
    exp_settings['num_barcodes'] = 0
    exp_settings['pulls_per_episode'] = 10
    exp_settings['epochs'] = 1200

    # Task Complexity
    exp_settings['noise_percent'] = [0.25, 0.5, 0.75, 0.875]
    exp_settings['noise_eval_epochs'] = 5
    exp_settings['sim_threshold'] = 0.5         #Cosine similarity threshold for clustering
    exp_settings['hamming_threshold'] = 5       #Hamming distance for clustering
    exp_settings['embedder_arm_trained'] = False

    # Data Logging
    exp_settings['tensorboard_logging'] = False
    exp_settings['timing'] = False
    ### End of Experimental Parameters ###

    ### Beginning of Experimental Runs ###
    f, axes = plt.subplots(1, 2, figsize=(12, 6))
    f1, axs = plt.subplots(1, 1, figsize=(6, 6))
    f3, axes3 = plt.subplots(1, 2, figsize=(12, 6))

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
    mem_store_types, exp_settings['epochs'], exp_settings['noise_eval_epochs'], num_repeats = exp_base
    exp_settings['hamming_threshold'], exp_settings['num_arms'], exp_settings['num_barcodes'], exp_settings[
        'barcode_size'], exp_settings['pulls_per_episode'], exp_settings['sim_threshold'] = exp_difficulty

    # Safety Assertions
    assert exp_settings['epochs'] > 10, "Training epochs must be greater than 10"
    assert exp_settings['pulls_per_episode'] > 2, "Pulls per episode must be greater than 2"
    assert exp_settings['barcode_size'] > 3*exp_settings['hamming_threshold'], "Barcodes must be greater than 3*Hamming"

    for idx_mem, mem_store in enumerate(mem_store_types):
        tot_rets = np.zeros(exp_settings['epochs']+exp_settings['noise_eval_epochs']*len(exp_settings['noise_percent']))
        exp_settings['mem_store'] = mem_store
        for i in range(num_repeats):

            print(f"\nNew Run --> Iteration: {i} | Type: {mem_store} | Device: {exp_settings['torch_device']}")
            logs, key_data = run_experiment_sl(exp_settings)
            log_return, log_loss_value, log_loss_policy, log_loss_total, log_embedder_accuracy, embedder_loss = logs
            keys, prediction_mapping, epoch_mapping, barcode_data = key_data 
            tot_rets += log_return/num_repeats
            # print(tot_rets)

        smoothed_rewards = pd.Series.rolling(pd.Series(tot_rets), 30).mean()
        smoothed_rewards = [elem for elem in smoothed_rewards]
        axes[0].plot(smoothed_rewards, label=f"Mem: {mem_store}")

        # Only graphing the loss on the final trial if there are multiple repeats
        smoothed_loss = pd.Series.rolling(pd.Series(log_loss_total), 30).mean()
        smoothed_loss = [elem for elem in smoothed_loss]
        axes[1].plot(
            smoothed_loss, label=f"Mem: {mem_store}")

        # Embedder/Mem Accuracy 
        smoothed_accuracy = pd.Series.rolling(pd.Series(log_embedder_accuracy), 30).mean()
        smoothed_accuracy = [elem for elem in smoothed_accuracy]
        axs.plot(smoothed_accuracy, label=f"Mem: {mem_store}")

        # T-SNE to visualize keys in memory
        embeddings = [x[0] for x in keys]
        labels = [x[1] for x in keys]

        # Map the keys to their predicted best arm pull
        if mem_store == 'context' or not exp_settings['embedder_arm_trained']:
            arm_pred = [epoch_mapping[x] for x in labels]
        else:
            arm_pred = labels
            labels = [x[2][0] for x in keys]

        # Artifically boost datapoint count to make tsne nicer?
        while len(embeddings) < 100:
            embeddings.extend(embeddings)
            arm_pred.extend(arm_pred)
            labels.extend(labels)

        f3, axes3 = plot_tsne_distribution(embeddings, labels, arm_pred, epoch_mapping, f3, axes3, idx_mem)
        axes3[idx_mem].xaxis.set_visible(False)
        axes3[idx_mem].yaxis.set_visible(False)
        axes3[idx_mem].set_title(mem_store)

        # # Embedder Barcode Confusion Matrix
        # axs[1] = make_confusion_matrix(epoch_mapping, barcode_data)

    # Put generic ritter trend on graph for quick reference
    if (exp_settings['num_arms'] == 10 and 
        exp_settings['barcode_size'] == 10 and 
        exp_settings['num_barcodes'] == 10 and 
        exp_settings['pulls_per_episode'] == 10 and
        exp_settings['epochs'] >= 2000):
        x = [0, 1000, 2000, 3000, 4000]
        y = [0.2, 0.35, 0.6, 0.7, 0.75]
        axes[0].plot(x,y, linestyle='dashed', label=f'Ritter Paper Returns')

    # # Original Task from QiHong
    # exp_settings['lstm_inputs_looped'] = False
    # exp_settings['task_version'] = 'original'
    # exp_settings['mem_store'] = 'obs/context'
    # exp_settings['epochs'] = 20
    # exp_settings['dim_hidden_lstm'] = 32
    # exp_settings['dim_hidden_a2c'] = 32
    # exp_settings['lstm_learning_rate'] = 5e-4
    # exp_settings['num_arms'] = 32           # Obs_Dim
    # exp_settings['barcode_size'] = 32       # Ctx_Dim
    # exp_settings['num_barcodes'] = 50       # n_unique_examples
    # exp_settings['pulls_per_episode'] = 10  # trial_length

    # logs, key_data = run_experiment_sl(exp_settings)
    # log_return, log_loss_value, log_loss_policy, log_embedder_accuracy, embedder_loss = logs
    # keys, prediction_mapping, epoch_mapping, barcode_data = key_data
    # axes[0].plot(log_return, label=f'Original Task')
    # axes[1].plot(log_loss_value, label=f'Original Value Loss')
    # axes[1].plot(log_loss_policy,
    #                 label=f'Original Policy Loss')

    # Graph Setup
    graph_title = f""" --- Returns and Loss ---
    LSTM Dim: {exp_settings['dim_hidden_lstm']} | A2C Dim: {exp_settings['dim_hidden_a2c']} | LSTM LR: {round(exp_settings['lstm_learning_rate'], 5)} 
    Val Loss Coef: {exp_settings['value_error_coef']}| Entropy Loss Coef: | {exp_settings['entropy_error_coef']}
    Embedding Dim: {exp_settings['embedding_size']} | Embedder LR: {round(exp_settings['embedder_learning_rate'], 5)} 
    Epochs: {exp_settings['epochs']} | Unique Barcodes: {exp_settings['num_barcodes']} | Barcode Dim: {exp_settings['barcode_size']}
    Arms: {exp_settings['num_arms']} | Pulls per Trial: {exp_settings['pulls_per_episode']} | Perfect Arms: {exp_settings['perfect_info']}
    Clusters: {int(exp_settings['num_barcodes']/exp_settings['num_arms'])} | IntraCluster Dist: {exp_settings['hamming_threshold']} | InterCluster Dist: {exp_settings['barcode_size']-2*exp_settings['hamming_threshold']}"""

    # Returns
    if exp_settings['task_version'] == 'bandit':
        num_bc = exp_settings['num_barcodes']
        num_eps = num_bc**2
        perfect_ret, random_ret = expected_return(exp_settings['num_arms'], exp_settings['perfect_info'])
        axes[0].axhline(y = random_ret, color='b', linestyle='dashed', label = 'Random Pulls')
        axes[0].axhline(y=perfect_ret, color='k', linestyle='dashed', label = 'Theoretical Max')
        colors = ['g', 'r', 'c', 'm']
        for idx, noise_percent in enumerate(exp_settings['noise_percent']):
            axes[0].axvline(x=exp_settings['epochs'] + idx*exp_settings['noise_eval_epochs'], color=colors[idx], linestyle = 'dashed',\
                label = f"{int(exp_settings['barcode_size']*noise_percent)} Bits Noisy")
            axs.axvline(x=exp_settings['epochs'] + idx*exp_settings['noise_eval_epochs'], color=colors[idx], linestyle='dashed',
                label = f"{int(exp_settings['barcode_size']*noise_percent)} Bits Noisy")
    axes[0].set_ylabel('Returns')
    axes[0].set_xlabel('Epoch')
    axes[0].legend(bbox_to_anchor=(0, -0.2, 1, 0), loc="upper left",
            mode="expand", borderaxespad=0, ncol=2)

    # Loss for Embedder and LSTM models
    axes[1].set_ylabel('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(bbox_to_anchor=(0, -0.2, 1, 0), loc="upper left",
            mode="expand", borderaxespad=0, ncol=2)

    # Accuracy
    axs.set_ylabel('Accuracy')
    axs.set_xlabel('Epoch')
    axs.set_title('Model Barcode Prediction Accuracy')
    axs.axhline(y=1/exp_settings['num_barcodes'], color='b', linestyle='dashed', label = 'Random Choice')
    axs.legend(bbox_to_anchor=(0, -0.2, 1, 0), loc="upper left",
            mode="expand", borderaxespad=0, ncol=2)

    sns.despine()
    f.tight_layout()
    f.subplots_adjust(top=0.7)
    f.suptitle(graph_title)
    f1.tight_layout()
    f3.tight_layout()
    f3.subplots_adjust(top=0.8)
    f3.suptitle("t-SNE on keys in memory from last training epoch\nIcon indicates barcode, color is best arm choice")
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    file_loc = "..\\Memory-Storage-Ritter\\dnd-lstm-sup-learning\\figs\\Saved_Plots\\"
    exp_id = f"{exp_settings['num_arms']}a{exp_settings['num_barcodes']}b{exp_settings['pulls_per_episode']}p {exp_settings['hamming_threshold']} hamming {num_repeats} run(s) "
    plot_type = ['returns', 'accuracy', 'tsne']
    for fig_num, figa in enumerate([f, f1, f3]):
        filename = file_loc + exp_id + plot_type[fig_num] +".png"
        figa.savefig(filename)
    plt.show()