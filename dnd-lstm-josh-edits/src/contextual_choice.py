"""demo: train a DND LSTM on a contextual choice task
"""
import time
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from task import ContextualChoice
from model import DNDLSTM as Agent
from utils import compute_stats, to_sqnp
from model.DND import compute_similarities
from model.utils import get_reward, compute_returns, compute_a2c_loss

def run_experiment(exp_settings):
    sns.set(style='white', context='talk', palette='colorblind')
    """
    exp_settings is a dict with parameters as keys:

    randomize: Boolean (for performing multiple trials to average results or not)
    epochs: int (number of times to wipe memory and rerun learning)
    obs_dim: int (input size of observaton)
    ctx_dim: int (input size of context label)
    kernel: string (should be either 'l2' or 'cosine')
    mem_key: string (should be 'obs', 'obs_con', or 'con')
    sim_threshold: float (should be specific to the kerneland the memory key)
    noise_percent: float (between 0 and 1)
    agent_input: string (choose between passing obs/context, or only obs into agent)
    mem_store: string (what gets stored in memory, obs, obs/context, or only context)
    kaiser_key_update: Boolean (do you apply kaiser averaging to a existing key)
    """

    if not exp_settings['randomize']:
        seed_val = 0
        torch.manual_seed(seed_val)
        np.random.seed(seed_val)

    epochs = exp_settings['epochs']
    sim_threshhold = exp_settings['sim_threshhold']
    kernel = exp_settings['kernel']
    
    '''init task'''
    n_unique_example = exp_settings['n_unique_examples']
    n_trials = 2 * n_unique_example

    # n time steps of a trial
    trial_length = 10

    # Percent of original observations to corrupt
    noise_percent = exp_settings['noise_percent']
    # after `tp_corrupt`, turn off the noise
    t_noise_off = int(trial_length * noise_percent)

    # input/output/hidden/memory dim
    obs_dim = exp_settings['obs_dim']
    ctx_dim = exp_settings['ctx_dim']

    # Task Choice
    task = ContextualChoice(
        obs_dim = obs_dim, 
        ctx_dim = ctx_dim,
        trial_length = trial_length,
        t_noise_off = t_noise_off
    )

    agent_input = exp_settings['agent_input']

    """
    exp_settings['randomize']
    exp_settings['epochs']
    exp_settings['sim_threshhold']
    exp_settings['kernel']
    exp_settings['n_unique_examples']
    exp_settings['noise_percent']
    exp_settings['obs_dim']
    exp_settings['ctx_dim']
    exp_settings['agent_input']
    exp_settings['mem_store']
    exp_settings['kaiser_key_update']
    """


    '''init model'''
    # set params
    dim_hidden = 32
    dim_output = 2
    dict_len = 100
    learning_rate = 5e-4
    n_epochs = epochs
    
    # Input is only observation, memory could be obs or context
    if exp_settings['agent_input'] != 'obs/context':
        x_dim = obs_dim
    
    # Input is obs/context pair, network needs to be larger than obs_dim
    # Will need to redefine this based on newer task label conventions
    else:
        x_dim = obs_dim + ctx_dim

    # init agent / optimizer
    agent = Agent(x_dim, dim_hidden, dim_output, dict_len, exp_settings)
    optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)

    '''train'''
    log_sims = np.zeros(n_epochs,)
    valid_pulls = np.zeros(n_epochs,)
    run_time = np.zeros(n_epochs,)
    log_return = np.zeros(n_epochs,)
    log_loss_value = np.zeros(n_epochs,)
    log_loss_policy = np.zeros(n_epochs,)

    log_Y = np.zeros((n_epochs, n_trials, trial_length))
    log_Y_hat = np.zeros((n_epochs, n_trials, trial_length))

    # loop over epoch
    for i in range(n_epochs):
        time_start = time.time()
        # get data for this epoch
        X, Y = task.sample(n_unique_example)
        # flush hippocampus
        agent.reset_memory()
        agent.turn_on_retrieval()

        # loop over the training set
        for m in range(n_trials):
            # prealloc
            cumulative_reward = 0
            probs, rewards, values = [], [], []
            h_t, c_t = agent.get_init_states()

            # Clearing the per trial hidden state buffer
            agent.flush_trial_buffer()

            # loop over time, for one training example
            for t in range(trial_length):
                # only save memory at the last time point
                agent.turn_off_encoding()
                if t == trial_length-1 and m < n_unique_example:
                    agent.turn_on_encoding()
                # print("_-_-"*10)
                # print(X[m][t], Y[m][t])
                # print(X[m][t][:obs_dim], Y[m][t])
                # print(X[m][t][obs_dim:], Y[m][t])
                # print("_-_-"*10)

                # Use memory_storage_type to know what gets passed into the agent
                    # Agent will further split inputs based on what gets stored in memory
                    # X[m][t].view(1, 1, -1) -> Pass in both oberservation and context
                    # X[m][t][:obs_dim].view(1, 1, -1) -> pass in only observation
                    # X[m][t][obs_dim:].view(1, 1, -1) -> pass in only context
                        # This won't happen but is useful to note for when it gets 
                        # split in the agent for mem store tests

                # recurrent computation at time t
                # if agent_input == 'obs':
                #     output_t, _ = agent(X[m][t][:obs_dim].view(1, 1, -1), h_t, c_t)
                #     a_t, prob_a_t, v_t, h_t, c_t = output_t
                # else: #agent_input == 'obs/context'
                
                # Pass in the whole observation/context pair, and split it up in the agent
                output_t, _ = agent(X[m][t].view(1, 1, -1), h_t, c_t)
                a_t, prob_a_t, v_t, h_t, c_t = output_t

                # compute immediate reward
                r_t = get_reward(a_t, Y[m][t])

                # log
                probs.append(prob_a_t)
                rewards.append(r_t)
                values.append(v_t)

                # log
                cumulative_reward += r_t
                log_Y_hat[i, m, t] = a_t.item()

            returns = compute_returns(rewards)
            loss_policy, loss_value = compute_a2c_loss(probs, values, returns)
            loss = loss_policy + loss_value
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log
            log_Y[i] = np.squeeze(Y.numpy())
            log_return[i] += cumulative_reward / n_trials
            log_loss_value[i] += loss_value.item() / n_trials
            log_loss_policy[i] += loss_policy.item() / n_trials

        # Memory retrievals above sim threshold
        good_pull = np.array(agent.dnd.recall_sims) >= sim_threshhold
        # print(len(agent.dnd.recall_sims), (n_trials-1) * trial_length)
        valid_pulls[i] = sum(good_pull)/ ((n_trials-1) * trial_length)

        # Avg Similarity between queries and memory
        log_sims[i] += np.mean(agent.dnd.recall_sims)
        # print(log_sims)

        # print out some stuff
        time_end = time.time()
        run_time[i] = time_end - time_start
        print(
            'Epoch %3d | return = %.2f | loss: val = %.2f, pol = %.2f | time = %.2f | avg sim = %.2f'%
            (i, log_return[i], log_loss_value[i], log_loss_policy[i], run_time[i], log_sims[i])
        )
    avg_valid_pulls = np.mean(valid_pulls)
    avg_time = np.mean(run_time)
    avg_sim = np.mean(log_sims)
    print(f"-*-*- \n\tAvg Time: {avg_time:.2f} | Avg Sim ({kernel}): {avg_sim:.2f} | Valid Pulls: {avg_valid_pulls:.2f}\n-*-*-")

    # Additional returns to graph out of this file
    keys, vals = agent.get_all_mems_josh()

    return log_return, log_loss_value, avg_sim, keys, vals

"""
'''learning curve'''
f, axes = plt.subplots(1, 2, figsize=(8, 3))
axes[0].plot(log_return)
axes[0].set_ylabel('Return')
axes[0].set_xlabel('Epoch')
axes[1].plot(log_loss_value)
axes[1].set_ylabel('Value loss')
axes[1].set_xlabel('Epoch')
sns.despine()
f.tight_layout()

'''show behavior'''
corrects = log_Y_hat[-1] == log_Y[-1]
acc_mu_no_memory, acc_se_no_memory = compute_stats(
    corrects[:n_unique_example])
acc_mu_has_memory, acc_se_has_memory = compute_stats(
    corrects[n_unique_example:])

n_se = 2
f, ax = plt.subplots(1, 1, figsize=(7, 4))
ax.errorbar(range(trial_length), y=acc_mu_no_memory,
            yerr=acc_se_no_memory * n_se, label='w/o memory')
ax.errorbar(range(trial_length), y=acc_mu_has_memory,
            yerr=acc_se_has_memory * n_se, label='w/  memory')
ax.axvline(t_noise_off, label='turn off noise', color='grey', linestyle='--')
ax.set_xlabel('Time')
ax.set_ylabel('Correct rate')
ax.set_title('Choice accuracy by condition')
f.legend(frameon=False, bbox_to_anchor=(1, .6))
sns.despine()
f.tight_layout()
# f.savefig('../figs/correct-rate.png', dpi=100, bbox_inches='tight')

'''visualize keys and values'''
keys, vals = agent.get_all_mems()
n_mems = len(agent.dnd.keys)
dmat_kk, dmat_vv = np.zeros((n_mems, n_mems)), np.zeros((n_mems, n_mems))
for i in range(n_mems):
    dmat_kk[i, :] = to_sqnp(compute_similarities(
        keys[i], keys, agent.dnd.kernel))
    dmat_vv[i, :] = to_sqnp(compute_similarities(
        vals[i], vals, agent.dnd.kernel))

# plot
dmats = {'key': dmat_kk, 'value': dmat_vv}
f, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, (label_i, dmat_i) in enumerate(dmats.items()):
    sns.heatmap(dmat_i, cmap='viridis', square=True, ax=axes[i])
    axes[i].set_xlabel(f'id, {label_i} i')
    axes[i].set_ylabel(f'id, {label_i} j')
    axes[i].set_title(
        f'{label_i}-{label_i} similarity, metric = {agent.dnd.kernel}'
    )
f.tight_layout()

# Something in the dimension of the hidden keys is causing a problem in this graph #

# '''project memory content to low dim space'''
# # convert the values to a np array, #memories x mem_dim
# vals_np = np.vstack([to_sqnp(vals[i]) for i in range(n_mems)])
# # project to PC space
# vals_centered = (vals_np - np.mean(vals_np, axis=0, keepdims=True))
# U, S, _ = np.linalg.svd(vals_centered, full_matrices=False)
# vals_pc = np.dot(U, np.diag(S))

# # pick pcs
# pc_x = 0
# pc_y = 1

# # plot
# f, ax = plt.subplots(1, 1, figsize=(7, 5))
# Y_phase2 = to_sqnp(Y[:n_unique_example, 0])
# for y_val in np.unique(Y_phase2):
#     ax.scatter(
#         vals_pc[Y_phase2 == y_val, pc_x],
#         vals_pc[Y_phase2 == y_val, pc_y],
#         marker='o', alpha=.7,
#     )
# ax.set_title(f'Each point is a memory (i.e. value)')
# ax.set_xlabel(f'PC {pc_x}')
# ax.set_ylabel(f'PC {pc_y}')
# ax.legend(['left trial', 'right trial'], bbox_to_anchor=(.6, .3))
# sns.despine(offset=20)
# f.tight_layout()

# Display all Graphs
plt.show()
# f.savefig('../figs/pc-v.png', dpi=100, bbox_inches='tight')
"""