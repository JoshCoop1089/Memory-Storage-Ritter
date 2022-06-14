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
    tb = SummaryWriter()

    # GPU is faster for everything but smaller LSTM dims on non embedder tests
    if exp_settings['dim_hidden_lstm'] <= 256 and exp_settings['mem_store'] != 'embedding':
        device = torch.device('cpu')
    else:
        device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

    if not exp_settings['randomize']:
        seed_val = 0
        torch.manual_seed(seed_val)
        np.random.seed(seed_val)
   
    '''init task'''
    # input/output/hidden/memory dim
    num_arms = exp_settings['num_arms']  # LSTM input dim
    barcode_size = exp_settings['barcode_size']   #Possible additional LSTM input dim
    num_barcodes = exp_settings['num_barcodes']   #Different number of contexts

    # Do we need a new set of contexts for each epoch, or do we keep it constant
    reset_barcodes_per_epoch = exp_settings['reset_barcodes_per_epoch']

    # Do we shuffle the arm assignments while keeping the barcodes constant?
    # Obvi don't have this and reset_barcodes true at the same time.
    reset_arms_per_epoch = exp_settings['reset_arms_per_epoch']

    # Arm pulls per single barcode episode
    pulls_per_episode = exp_settings['pulls_per_episode']

    # Arm rewards can be deterministic for debugging
    perfect_info = exp_settings['perfect_info']

    # Make a percent of observed pulls useless for prediction
    ### Noise not implemented yet for Bandit###
    noise_percent = exp_settings['noise_percent']
    noise_observations = int(pulls_per_episode * noise_percent)

    # Task Choice
    if exp_settings['task_version'] == 'bandit':
        # Example: 4 unique barcodes -> 16 total barcodes in epoch, 4 trials of each unique barcode
        episodes_per_epoch = num_barcodes**2

        task = ContextualBandit(
            pulls_per_episode, episodes_per_epoch,
            num_arms, num_barcodes, barcode_size,
            reset_barcodes_per_epoch, reset_arms_per_epoch,
            noise_observations, device, perfect_info)

        # LSTM Chooses which arm to pull
        dim_output_lstm = num_arms
        dict_len = num_barcodes**2
        value_weight = 0.5
        entropy_weight = 0.01

    elif exp_settings['task_version'] == 'original':
        task = ContextualChoice(
            num_arms, num_barcodes, noise_observations)
        episodes_per_epoch = 2*num_barcodes
        epoch_mapping = {}

        # LSTM indicates if this is a 0 or a 1 task
        dim_output_lstm = 2
        dict_len = 50
        value_weight = 1
        entropy_weight = 0

    '''init model'''
    n_epochs = exp_settings['epochs']
    kernel = exp_settings['kernel']
    agent_input = exp_settings['agent_input']
    
    # Input to LSTM is only observation
    if agent_input == 'obs':
        dim_input_lstm = num_arms + 1
    
    # Input is obs/context pair
    else:  # agent_input == 'obs/context'
        dim_input_lstm = num_arms + barcode_size + 1

    dim_hidden_lstm = exp_settings['dim_hidden_lstm']
    learning_rate = exp_settings['lstm_learning_rate']

    # init agent / optimizer
    agent = Agent(dim_input_lstm, dim_hidden_lstm, dim_output_lstm,
                     dict_len, exp_settings, device).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, agent.parameters()), lr=learning_rate)
    # scheduler = LRScheduler.OneCycleLR(
    #     optimizer, max_lr=learning_rate, steps_per_epoch=episodes_per_epoch, epochs=n_epochs)

    '''train'''
    run_time = np.zeros(n_epochs,)
    log_return = np.zeros(n_epochs,)
    agent.dnd.log_embedder_accuracy = np.zeros(n_epochs,)
    log_loss_value = np.zeros(n_epochs,)
    log_loss_policy = np.zeros(n_epochs,)
    log_Y = np.zeros((n_epochs, episodes_per_epoch, pulls_per_episode))
    log_Y_hat = np.zeros((n_epochs, episodes_per_epoch, pulls_per_episode))

    barcode_data = [[[]]]
    print("\n", "-*-_-*-"*3, "\n")
    # loop over epoch
    for i in range(n_epochs):
        time_start = time.time()
        if i < n_epochs - 1:
            barcode_data.append([[]])

        # Barcode to arm mapping can be changed per epoch, or kept constant
        # see exp_settings['reset_barcodes_per_epoch']
        # get data for this epoch
        if exp_settings['task_version'] == 'bandit':
            observations_barcodes, reward_from_obs, epoch_mapping = task.sample()
            agent.dnd.mapping = epoch_mapping
            # print(observations_barcodes)

        elif exp_settings['task_version'] == 'original':
            X, Y, contexts = task.sample(num_barcodes)

        # flush hippocampus
        agent.reset_memory()
        agent.turn_on_retrieval()
        # print(sorted(list(epoch_mapping.items())))

        # loop over the training set
        for m in range(episodes_per_epoch):
            # prealloc
            embedder_accuracy = 0
            cumulative_reward = 0
            probs, rewards, values, entropies = [], [], [], []
            h_t, c_t = agent.get_init_states()
            h_t = h_t.to(device)
            c_t = c_t.to(device)
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

            # loop over time, for one training example
            for t in range(pulls_per_episode):

                # only save memory at the last time point
                agent.turn_off_encoding()
                if t == pulls_per_episode-1 and m < episodes_per_epoch:
                    agent.turn_on_encoding()

                if exp_settings['task_version'] == 'bandit':
                    output_t, _ = agent(observations_barcodes[m][t].view(1, 1, -1), 
                                            reward_from_obs[m][t].view(1,1,-1),
                                            h_t, c_t)
                    a_t, assumed_barcode, prob_a_t, v_t, entropy, h_t, c_t = output_t

                elif exp_settings['task_version'] == 'original':
                    # print(X[m][t])
                    output_t, _ = agent(X[m][t].view(1, 1, -1),
                                            h_t, c_t)
                    a_t, assumed_barcode, prob_a_t, v_t, entropy, h_t, c_t = output_t
                    
                # print("Action:", a_t, "P-BC:", assumed_barcode, "P-BA:", epoch_mapping[assumed_barcode])
                a_t = a_t.to(device)
                prob_a_t = prob_a_t.to(device)
                v_t = v_t.to(device)
                entropy = entropy.to(device)
                h_t = h_t.to(device)
                c_t = c_t.to(device)
            
                if exp_settings['task_version'] == 'bandit':
                    assumed_barcode_string = np.array2string(assumed_barcode.cpu().numpy())[1:-1].replace(" ", "").replace(".", "")

                    # compute immediate reward for actor network
                    r_t = get_reward_from_assumed_barcode(a_t, reward_from_obs[m][t], 
                                                        assumed_barcode_string, epoch_mapping, perfect_info)

                    # Does the embedder predicted context match the actual context?
                    obs_bc = observations_barcodes[m][t].view(1,-1)
                    obs = obs_bc[0][:exp_settings['num_arms']]
                    bc = obs_bc[0][-exp_settings['barcode_size']:]
                    real_bc = np.array2string(bc.cpu().numpy())[
                        1:-1].replace(" ", "").replace(".", "")
                    # print(real_bc, assumed_barcode_string)
                    match = int(real_bc == assumed_barcode_string)
                    embedder_accuracy += match
                    if exp_settings['mem_store'] == 'embedding' and i == n_epochs - 1:
                        barcode_data[i][m].append((real_bc, assumed_barcode_string))
                    # if t == pulls_per_episode - 1:
                    #     print(f"R_t: {cumulative_reward} | Emb_Acc: {embedder_accuracy}")

                elif exp_settings['task_version'] == 'original':
                    r_t = get_reward(a_t, Y[m][t])

                r_t = r_t.to(device)

                # log
                probs.append(prob_a_t)
                rewards.append(r_t)
                values.append(v_t)
                entropies.append(entropy)
                cumulative_reward += r_t
                log_Y_hat[i, m, t] = a_t.item()

            # print("-- end of ep --")
            for name, param in agent.named_parameters():
                    if not param.requires_grad:
                        print(name)

            returns = compute_returns(rewards, device, gamma = 0.0)
            loss_policy, loss_value, entropies_tensor = compute_a2c_loss(probs, values, returns, entropies)
            loss = loss_policy + value_weight*loss_value - entropy_weight*entropies_tensor

            # Testing for gradient leaks between embedder model and lstm model
            # print("B-End_of_Ep:\n", agent.a2c.critic.weight.grad)
            # print("B-End_of_Ep:\n", agent.dnd.embedder.e2c.weight.grad)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1)
            optimizer.step()
            # print("A-End_of_Ep:\n", agent.a2c.critic.weight.grad)
            # print("A-End_of_Ep:\n", agent.dnd.embedder.e2c.weight.grad)

            # log
            # if exp_settings['task_version'] == 'bandit':
            #     log_Y[i] = np.squeeze(reward_from_obs.cpu().numpy())
            # elif exp_settings['task_version'] == 'original':
            #    log_Y[i] += np.squeeze(Y[m][t].cpu().numpy())
            
            # Updating avg return per episode
            log_return[i] += cumulative_reward.item()/(episodes_per_epoch*pulls_per_episode)
           
            # Updating avg accuracy per episode
            agent.dnd.log_embedder_accuracy[i] += embedder_accuracy / (episodes_per_epoch*pulls_per_episode)
            
            # Loss Logging
            log_loss_value[i] += loss_value.item() / episodes_per_epoch
            log_loss_policy[i] += loss_policy.item() / episodes_per_epoch
        
            # # Learning Rate Scheduler
            # scheduler.step()

        # Tensorboard Stuff
        if i%5 == 0:
            # episodes = np.count_nonzero(agent.dnd.log_embedder_accuracy)
            # if  episodes > 10:
            #     avg_acc = agent.dnd.log_embedder_accuracy[episodes-10:episodes].mean()
            # else:
            #     avg_acc = agent.dnd.log_embedder_accuracy[:episodes].mean()
        
            # print("Embedder Accuracy: ", round(avg_acc, 4))

            tb.add_scalar("LSTM Loss_Value", log_loss_value[i], i)
            tb.add_scalar("LSTM Loss_Policy", log_loss_policy[i], i)
            tb.add_scalar("LSTM Total Loss",
                        log_loss_policy[i]+log_loss_value[i], i)
            tb.add_scalar("LSTM Returns", log_return[i], i)
            tb.add_scalar("Barcode Prediction Accuracy",
                          agent.dnd.log_embedder_accuracy[i], i)
            for name, weight in agent.named_parameters():
                tb.add_histogram(name, weight, i)
                try:
                    tb.add_histogram(f'{name}.grad', weight.grad, i)
                    tb.add_histogram(f'{name}_grad_norm', weight.grad.norm(), i)
                except Exception:
                    continue

            if exp_settings['mem_store'] == 'embedding':
                for name, weight in agent.dnd.embedder.named_parameters():
                    # print(name)
                    tb.add_histogram(name, weight, i)
                    try:
                        tb.add_histogram(f'{name}.grad', weight.grad, i)
                        tb.add_histogram(f'{name}_grad_norm', weight.grad.norm(), i)
                    except Exception:
                        continue

        # print out some stuff
        time_end = time.time()
        run_time[i] = time_end - time_start
        # print(sorted(list(epoch_mapping.keys())))
        print(
            'Epoch %3d | avg_return = %.2f | loss: val = %.2f, pol = %.2f | time = %.2f'%
            (i, log_return[i], log_loss_value[i], log_loss_policy[i], run_time[i])
        )
# , log_embedder_accuracy[i]
    # Flatten Returns and Accuracy for Graphing
    # log_return = list(itertools.chain.from_iterable(log_return))
    # log_embedder_accuracy = list(
    #     itertools.chain.from_iterable(log_embedder_accuracy))

    # Additional returns to graph out of this file
    keys, prediction_mapping = agent.get_all_mems_embedder()

    # Embedder avg loss per epoch
    embedder_loss = 0
    if exp_settings['mem_store'] == 'embedding':
        embed_loss = [x.cpu().detach().numpy() for x in agent.dnd.embedder_loss]
        embedder_loss = np.zeros(n_epochs)
        for m in range(n_epochs):
            for i in range(episodes_per_epoch):
                embedder_loss[m] += embed_loss[m*episodes_per_epoch+i]/episodes_per_epoch
            tb.add_scalar("Embedder Loss", embedder_loss[m], m)

    logs = log_return, log_loss_value, log_loss_policy, agent.dnd.log_embedder_accuracy, embedder_loss
    key_data = keys, prediction_mapping, epoch_mapping, barcode_data

    tb.flush()
    tb.close()
    return  logs, key_data

def update_avg_value(current_list, epoch_num, episode_num, episode_sum, episodes_per_epoch, pulls_per_episode):
    i = epoch_num
    m = episode_num
    episode_counter = (m) + (i)*(episodes_per_epoch)
    episode_avg = episode_sum/pulls_per_episode

    if m == 0:
        if i == 0:
            current_list[i][m] = episode_avg
        else:
            current_list[i][m] = (
                current_list[i-1][-1]*(episode_counter) + episode_avg)/(episode_counter+1)
    else:
        current_list[i][m] = (
            current_list[i][m-1]*(episode_counter) + episode_avg)/(episode_counter+1)
    return current_list

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

def plot_tsne_distribution(keys, labels, fig, axes):
    features = np.array([y.cpu().numpy() for y in keys])
    tsne = TSNE(n_components=2).fit_transform(features)
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)
    label_list = [labels[i][0] for i in range(len(labels))]
    start, end = 0,0
    # # for every class, we'll add a scatter plot separately
    for barcode, num, _ in labels:
        start = end
        end = end + num
        # find the samples of the current class in the data
        indices = [i for i in range(start, end)]
        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        # add a scatter plot with the corresponding color and label
        # , label=f"B:{barcode} | Valid:{valid} | Num:{num}"
        axes.scatter(current_tx, current_ty)
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

if __name__  == '__main__':
    """     
    Embedder Best Vals so far for no barcode/no arm switching:
        exp_settings['dim_hidden_lstm'] = 64
        exp_settings['embedding_size'] = 512
        exp_settings['num_arms'] = 5
        exp_settings['barcode_size'] = 5
        exp_settings['num_barcodes'] = 5
        exp_settings['pulls_per_episode'] = 50
        exp_settings['perfect_info'] = False
        exp_settings['reset_barcodes_per_epoch'] = False
        exp_settings['reset_arms_per_epoch'] = False
        exp_settings['lstm_learning_rate'] = 1e-3
        exp_settings['embedder_learning_rate'] = 5e-4

    Ritter Context "Best" Vals So Far, with arm switching
        exp_settings['num_arms'] = 10
        exp_settings['barcode_size'] = 10
        exp_settings['num_barcodes'] = 10
        exp_settings['pulls_per_episode'] = 25
        exp_settings['dim_hidden_lstm'] = 128
        exp_settings['lstm_learning_rate'] = 5e-4
        exp_settings['reset_arms_per_epoch'] = True

        Actor/Critic Network:
            self.dim_hidden = 2*(dim_input + dim_output) - 1
            shared first layer (dim_input, dim_hidden)
            0.5 dropout layer
            actor layer (dim_hidden, num_arms)
            
            and

            shared first layer (dim_input, dim_hidden)
            0.5 dropout layer
            critic layer (dim_hidden, 1)
    """

    # Experimental Parameters
    exp_settings = {}
    exp_settings['randomize'] = False
    exp_settings['epochs'] = 50
    exp_settings['kernel'] = 'cosine'           #cosine, l2
    exp_settings['noise_percent'] = 0.5
    exp_settings['agent_input'] = 'obs/context' #obs, obs/context
    exp_settings['mem_store'] = 'context'   #obs/context, context, embedding, obs, hidden (unsure how to do obs, hidden return calc w/o barcode predictions)
    exp_settings['dim_hidden_lstm'] = 128
    exp_settings['embedding_size'] = 512
    exp_settings['num_arms'] = 10
    exp_settings['barcode_size'] = 10
    exp_settings['num_barcodes'] = 10
    exp_settings['pulls_per_episode'] = 25
    exp_settings['perfect_info'] = False
    exp_settings['reset_barcodes_per_epoch'] = False
    exp_settings['reset_arms_per_epoch'] = True
    exp_settings['lstm_learning_rate'] = 5e-4
    exp_settings['embedder_learning_rate'] = 5e-4
    exp_settings['task_version'] = 'bandit'      #bandit, original

    perfect_ret, random_ret = expected_return(
        exp_settings['num_arms'], exp_settings['perfect_info'])
    f, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Context in memory version
    # tot_returns = []
    # tot_loss = []
    # ppe_l = [10, 20, 30, 50]
    # for ppe in ppe_l:
    #     exp_settings['pulls_per_episode'] = ppe

    # logs, key_data = run_experiment_sl(exp_settings)
    # log_return, log_loss_value, log_loss_policy, log_embedder_accuracy, embedder_loss = logs
    # keys, prediction_mapping, epoch_mapping, barcode_data = key_data 
    # # tot_returns.append(log_return)

    # # # # # Did the embedder graph run out of memory? Copy paste the console to a txt file and salvage some results
    # # # # filename = 'C:\\Users\\joshc\\Google Drive\\CS Research\\Memory-Storage-Ritter\\dnd-lstm-sup-learning\\src\\cont_data.txt'
    # # # # log_return = []
    # # # # log_loss_value = []
    # # # # with open(filename, 'r') as file:
    # # # #     for line in file:
    # # # #         splits = line.split('|')
    # # # #         log_return.append(float(splits[1][-5:-1]))
    # # # #         loss = splits[2].split(',')
    # # # #         loc = loss[0].index('=')+2
    # # # #         loss_val = loss[0][loc:]
    # # # #         log_loss_value.append(float(loss_val))
    # # # # # file.close()
    # total_loss = log_loss_value + log_loss_policy
    # # tot_loss.append(total_loss)
    # axes[0].plot(log_return, label=f'Ritter Returns')
    # axes[1].plot(total_loss, label=f'Ritter Total Loss')
    # axes[0].plot(np.mean(tot_returns, axis = 0), label = 'Averaged Returns')
    # axes[1].plot(np.mean(tot_loss, axis = 0), label = 'Averaged Total Loss')
    # axes[1].plot(log_loss_value,
    #                 label=f'Ritter Value Loss')
    # axes[1].plot(log_loss_policy,
                    # label=f'Ritter Policy Loss')

    # Embedding Version
    exp_settings['mem_store'] = 'embedding'
    logs, key_data = run_experiment_sl(exp_settings)
    log_return, log_loss_value, log_loss_policy, log_embedder_accuracy, embedder_loss = logs
    keys, prediction_mapping, epoch_mapping, barcode_data = key_data

    # # Did the embedder graph run out of memory? Copy paste the console to a txt file and salvage some results
    # filename = 'C:\\Users\\joshc\\Google Drive\\CS Research\\Memory-Storage-Ritter\\dnd-lstm-sup-learning\\src\\emb_data.txt'
    # log_return = []
    # log_embedder_accuracy = []
    # log_loss_value = []
    # with open(filename, 'r') as file:
    #     for line in file:
    #         splits = line.split('|')
    #         log_return.append(float(splits[1][-5:-1]))
    #         log_embedder_accuracy.append(float(line[-4:]))
    #         loss = splits[2].split(',')
    #         loc = loss[0].index('=')+2
    #         loss_val = loss[0][loc:]
    #         log_loss_value.append(float(loss_val))
    # file.close()
    total_loss = log_loss_value + log_loss_policy
    axes[0].plot(log_return, label = f'Embeddings')
    axes[1].plot(total_loss,label=f'Emb A2C Total Loss')
    # axes[1].plot(log_loss_value,
    #                 label=f'Embedding LSTM Value Loss')
    # axes[1].plot(log_loss_policy,
    #                 label=f'Embedding LSTM Policy Loss')
    axes[1].plot(embedder_loss, label = f'Embedder Loss')

    # # Original Task from QiHong
    # exp_settings['task_version'] = 'original'
    # exp_settings['mem_store'] = 'obs/context'
    # exp_settings['epochs'] = 20
    # exp_settings['dim_hidden_lstm'] = 32
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
    LSTM Hidden Dim: {exp_settings['dim_hidden_lstm']} | LSTM Learning Rate: {exp_settings['lstm_learning_rate']} 
    Embedding Dim: {exp_settings['embedding_size']} | Embedder Learning Rate: {exp_settings['embedder_learning_rate']} 
    Epochs: {exp_settings['epochs']} | Unique Barcodes: {exp_settings['num_barcodes']} | Barcode Dim: {exp_settings['barcode_size']}
    Arms: {exp_settings['num_arms']} | Pulls per Trial: {exp_settings['pulls_per_episode']} | Perfect Arms: {exp_settings['perfect_info']}"""

    # Returns
    if exp_settings['task_version'] == 'bandit':
        # axes[0].axhline(y=perfect_ret, color='r', linestyle='dashed', label = 'Perfect Pulls')
        axes[0].axhline(y=random_ret, color='b', linestyle='dashed', label = 'Random Pulls')
    axes[0].set_ylabel('Returns')
    axes[0].set_xlabel('Epoch')
    axes[0].legend(bbox_to_anchor=(0, -0.2, 1, 0), loc="upper left",
            mode="expand", borderaxespad=0, ncol=2)

    # Loss for Embedder and LSTM models
    axes[1].set_ylabel('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(bbox_to_anchor=(0, -0.2, 1, 0), loc="upper left",
            mode="expand", borderaxespad=0, ncol=2)

    if exp_settings['mem_store'] == 'embedding':
        # Embedder Accuracy 
        f1, axs = plt.subplots(1, 2, figsize=(18, 6))
        axs[0].plot(log_embedder_accuracy, label=f"Embeddings")
        axs[0].set_ylabel('Accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].set_title('Embedding Model Barcode Prediction Accuracy')
        axs[0].axhline(y=1/exp_settings['num_barcodes'], color='b', linestyle='dashed', label = 'Random Choice')
        axs[0].legend(bbox_to_anchor=(0, -0.2, 1, 0), loc="upper left",
                mode="expand", borderaxespad=0, ncol=2)

        # Embedder Barcode Confusion Matrix
        axs[1] = make_confusion_matrix(epoch_mapping, barcode_data)
        axs[1].legend(bbox_to_anchor=(0, -0.2, 1, 0), loc="upper left",
                    mode="expand", borderaxespad=0, ncol=2)
        f1.tight_layout()

        # T-SNE Mapping Attempts (from https://learnopencv.com/t-sne-for-feature-visualization/)
        labels = []
        total_keys = exp_settings['pulls_per_episode']*(exp_settings['num_barcodes']**2)
        for mem_id, barcode_keys in enumerate(keys):
            # print(prediction_mapping)
            num_keys = len(barcode_keys)
            # print(mem_id, num_keys)
            if num_keys > 0:
                barcode = get_barcode(mem_id, prediction_mapping)
                labels.append((barcode, num_keys, round(100*num_keys/total_keys, 2)))
        # print("Epoch Mapping:", epoch_mapping.keys())
        # print("Key Info:", labels, total_keys)
        
        flattened_keys = list(itertools.chain.from_iterable(keys))
        # print(len(flattened_keys))
        
        f3, axes3 = plt.subplots(1, 1, figsize=(8, 5))
        f3, axes3 = plot_tsne_distribution(flattened_keys, labels, f3, axes3)
        axes3.xaxis.set_visible(False)
        axes3.yaxis.set_visible(False)
        axes3.set_title("t-SNE on Embeddings from last epoch")
        # axes3.legend(bbox_to_anchor=(0, -0.1, 1, 0), loc="upper left",
        #            mode="expand", borderaxespad=0, ncol=2)
        f3.tight_layout()

    sns.despine()
    f.tight_layout()
    f.subplots_adjust(top=0.8)
    f.suptitle(graph_title)
    plt.show()