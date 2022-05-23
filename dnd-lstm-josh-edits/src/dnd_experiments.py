from cProfile import run
from cmath import log
from contextual_choice import run_experiment
from contextual_choice_sl import run_experiment_sl
from utils import compute_stats, to_sqnp
from sl_model.DND import compute_similarities
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def get_sim_threshhold(mem_type, kernel = "cosine", context = "no_context"):
    # Manually found over some data 
    # HIGHLY TWEAKABLE
    if mem_type == "Original":
        if kernel == "l2":
            if context == "full":
                return -9.5
            elif context == 'no_context':
                return -7
        elif kernel == 'cosine':
            if context == "full":
                return 0.05
            elif context == 'no_context':
                return 0.1
    elif mem_type == "Quarter":
        if kernel == "l2":
            if context == "full":
                return -1.2
            elif context == 'no_context':
                return -0.8
        elif kernel == 'cosine':
            if context == "full":
                return 0.4
            elif context == 'no_context':
                return 0.45
    else:
        raise ValueError(f'unrecog mem_type: {mem_type} or unrecog kernel: {kernel}')

def graph_the_things(epochs, mem_type, num_repeats = 1, sim_thresh = False, kernel = [], update = [], change_task = [], noise = []):
    """
    run experiments function is wonky, so how to specialize it to make these changes easier
    """
    log_return = [[]]
    log_sims = [[]]
    k,v = [[]],[[]]
    exp_settings = {}

    """  Possible options for exp_settings
    exp_settings['randomize'] = True/False
    exp_settings['epochs'] = #
    exp_settings['sim_threshhold'] = #
    exp_settings['kernel'] = l2/cosine
    exp_settings['n_unique_examples'] = #
    exp_settings['noise_percent'] = #
    exp_settings['obs_dim'] = #
    exp_settings['ctx_dim'] = #
    exp_settings['agent_input'] = 'obs', 'obs/context'
    exp_settings['mem_store'] = 'obs/context', 'context', 'hidden'
    exp_settings['kaiser_key_update'] = True/False
    exp_settings['hidden_layer_size'] = #
    """

    # Default Settings
    exp_settings['randomize'] = True
    exp_settings['epochs'] = epochs
    exp_settings['sim_threshhold'] = -200
    exp_settings['kernel'] = 'cosine'
    exp_settings['n_unique_examples'] = 5
    exp_settings['noise_percent'] = 0.5
    exp_settings['obs_dim'] = 10
    exp_settings['ctx_dim'] = 10
    exp_settings['agent_input'] = 'obs'
    exp_settings['mem_store'] = 'obs/context'
    exp_settings['kaiser_key_update'] = False
    exp_settings['hidden_layer_size'] = 16


    for num1 in range(len(mem_type)):
        exp_settings['mem_store'] = mem_type[num1]
        len_input2 = max(len(kernel), len(update), len(change_task), len(noise))
        for num2 in range(len_input2):
            runs = []
            sims = []
            change = [f'M:{mem_type[num1]}']
            # context = 'no_context'
            kernel1 = 'cosine'
            
            # Only use memory update on our version
            if not change_task and exp_settings['mem_store'] == 'hidden':
                exp_settings['agent_input'] = 'obs'
                exp_settings['kaiser_key_update'] = True
                change.append('Kaiser_Updates')

            if kernel:
                exp_settings['kernel'] = kernel[num2]
                kernel1 = kernel[num2]
                change.append(kernel1)
            elif update:
                exp_settings['kaiser_key_update'] = update[num2]
                change.append(update[num2])
            elif change_task:
                exp_settings['agent_input'] = change_task[num2]
                context = change_task[num2]
                change.append(f'T:{context}')
            elif noise:
                exp_settings['noise_percent'] = noise[num2]
                change.append(('noise', noise[num2]))

            # Won't need this for supervised learning, but keeping for legacy code versions
            if sim_thresh and exp_settings['mem_store'] == 'hidden':
                exp_settings['sim_threshhold'] = 0.4
                # get_sim_threshhold(mem_type = 'Quarter', kernel = kernel1, context = context)

            # if exp_settings['agent_input'] == 'obs' and exp_settings['mem_store'] == 'obs/context':
            #     continue

            for iter in range(num_repeats):
                print("\n", "- -"*10)
                print('Iteration:', iter, change)
                returns, loss, avg_sim, keys, vals = run_experiment(exp_settings)
                runs.append(returns)
                sims.append(avg_sim)
            avg_returns = np.mean(runs, axis = 0)                                     
            avg_sims = np.mean(sims, axis = 0)                                     
            log_return[num1].append((avg_returns, change))        
            log_sims[num1].append((avg_sims, change))

            # This stores the k,v pairs from memory for every run, but currently the graph only uses the last version present
            k[num1].append(keys)
            v[num1].append(vals)
        k.append([])
        v.append([])
        log_return.append([])
        log_sims.append([])
    return log_return, log_sims, k, v

epochs = 2
num_repeats = 1
mem_type = ['obs/context', 'hidden']
kernel = ['cosine', 'l2']
noise = [0.5]
# , 0.2, 0.5]
# , 0.7]
# , 0.8, 0.9]
update_type = [True, False]
change_task = ['obs/context', 'obs']

log_returns = []
# log_returns, log_sims, k, v = graph_the_things(epochs, mem_type, num_repeats, sim_thresh = True, change_task=change_task)
# log_returns.append(graph_the_things(epochs, mem_type, num_repeats, sim_thresh = False, change_task=change_task))
# log_returns.append(graph_the_things(epochs, mem_type, kernel=kernel))
# log_returns.append(graph_the_things(epochs, mem_type, num_repeats, sim_thresh = True, update=update_type))
# log_returns.append(graph_the_things(epochs, mem_type, num_repeats, sim_thresh = False, update=update_type))
log_returns, log_sims, k, v = graph_the_things(epochs, mem_type, num_repeats, sim_thresh = True, noise=noise)
# print(log_returns)

f, axes = plt.subplots(1, 2, figsize=(16, 10))
dataz = []
labelz= []
for ind_trial in range(len(log_returns)):
    for exp in range(len(log_returns[ind_trial])):
        # Returns
        data = log_returns[ind_trial][exp][0]
        mem_type_name = log_returns[ind_trial][exp][1][0]
        label = (mem_type_name,log_returns[ind_trial][exp][1][-1][1])
        if mem_type_name == 'M:obs/context':
            linestyle = 'dotted'
            marker = '.'
        if mem_type_name == 'M:context':
            linestyle = 'dashed'
            marker = 'o'
        if mem_type_name == 'M:hidden':
            linestyle = 'solid'
            marker = 'x'
        axes[0].plot(data, linestyle=linestyle, marker=marker, label = label)

        # Avg Similarity
        dataz.append(log_sims[ind_trial][exp][0])
        labelz.append((log_sims[ind_trial][exp][1][0],log_sims[ind_trial][exp][1][-1][1]))

axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Return')
axes[0].legend(bbox_to_anchor=(0,-0.3,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2)
axes[1].plot(dataz, 'bo')
axes[1].set_ylabel('Average Similiarity')
plt.subplot(122)
plt.xticks(range(len(labelz)), labelz, rotation = 45)

sns.despine()
f.tight_layout()
f.subplots_adjust(top=0.9)
plt.suptitle("Kernel: Cosine, Dict Len: 100, Input Dim: 128, Trial Length: 10 \n Sim Thresholding, Memory Updates on Quarter Trials Only", y=.98)


# '''visualize keys and values'''
# Need to figure out what i want to display, as of now it shows the memory from the last trial of the manipulated variable for og and hidden
def plot_sim_comparisons(k, v, f, a, mem_type_name, noise):
    for x in range(len(k)):
        # Code from original contextual_choice.py file from QiHong Lu github
        n_mems = len(k[x])
        keys = k[x]
        vals = v[x]
        dmat_kk, dmat_vv = np.zeros((n_mems, n_mems)), np.zeros((n_mems, n_mems))
        for i in range(n_mems):
            dmat_kk[i, :] = to_sqnp(compute_similarities(
                keys[i], keys, 'cosine'))
            dmat_vv[i, :] = to_sqnp(compute_similarities(
                vals[i], vals, 'cosine'))

        # plot
        dmats = {'key': dmat_kk, 'value': dmat_vv}
        for i, (label_i, dmat_i) in enumerate(dmats.items()):
            sns.heatmap(dmat_i, cmap='viridis', square=True, ax=a[x][i])
            a[x][i].set_xlabel(f'id, {label_i} i')
            a[x][i].set_ylabel(f'id, {label_i} j')
            a[x][i].set_title(
                f'{label_i}-{label_i} sim, metric=cos, {mem_type_name}, {noise[x]}'
            )
        f.tight_layout()
    return f, a

# f1, axes1 = plt.subplots(2, 2, figsize=(10, 10))
# f1, axes1 = plot_sim_comparisons(k[0],v[0],f1,axes1,'Original', noise)
# f2, axes2 = plt.subplots(2, 2, figsize=(10, 10))
# f2, axes2 = plot_sim_comparisons(k[1],v[1],f2,axes2,'HidState', noise)


# T-sne Mapping Attempts (from https://learnopencv.com/t-sne-for-feature-visualization/)
from sklearn.manifold import TSNE
def scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range
def plot_tsne_distribution(k,f,a, mem_type_name, noise):
    for i, trial in enumerate(k):
        features = np.array([y.numpy() for y in trial])
        tsne = TSNE(n_components=2).fit_transform(features)
        tx = tsne[:, 0]
        ty = tsne[:, 1]
        tx = scale_to_01_range(tx)
        ty = scale_to_01_range(ty)
        """
        # # for every class, we'll add a scatter plot separately
        for label in colors_per_class:
            # find the samples of the current class in the data
            indices = [i for i, l in enumerate(labels) if l == label]
            # extract the coordinates of the points of this class only
            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)
            # convert the class color to matplotlib format
            color = np.array(colors_per_class[label], dtype=np.float) / 255
            # add a scatter plot with the corresponding color and label
            ax.scatter(current_tx, current_ty, c=color, label=label)
        """
        a[i].scatter(tx, ty)
        a[i].xaxis.set_visible(False)
        a[i].yaxis.set_visible(False)
        a[i].set_title(f"t-SNE on Keys, {mem_type_name}, {noise[i]}")
        f.tight_layout()
    return f, a
# f3, axes3 = plt.subplots(1, len(noise), figsize=(10, 5))
# f3, axes3 = plot_tsne_distribution(k[0],f3,axes3, 'Original', noise)
# f4, axes4 = plt.subplots(1, len(noise), figsize=(10, 5))
# f4, axes4 = plot_tsne_distribution(k[1],f4,axes4, 'HidState', noise)
plt.show()