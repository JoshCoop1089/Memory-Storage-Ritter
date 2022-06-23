from contextual_choice_sl import run_experiment_sl
from bayes_opt import BayesianOptimization
from bayes_opt import SequentialDomainReductionTransformer

from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs


import numpy as np


def avg_returns(dim_hidden_lstm, lstm_learning_rate, dim_hidden_a2c, value_error_coef, entropy_error_coef):

    # Experimental Parameters
    exp_settings = {}
    exp_settings['randomize'] = False
    exp_settings['tensorboard_logging'] = False

    # Task Info
    exp_settings['kernel'] = 'cosine'           #cosine, l2
    exp_settings['agent_input'] = 'obs/context' #obs, obs/context
    exp_settings['mem_store'] = 'context'   #obs/context, context, embedding, obs, hidden (unsure how to do obs, hidden return calc w/o barcode predictions)
    exp_settings['task_version'] = 'bandit'      #bandit, original
    exp_settings['noise_percent'] = 0.5

    # 1000 epochs with 10 barcodes == 100k episodes
    # Ritter returns were averaging ~0.35 at this point in training
    exp_settings['epochs'] = 1000
    exp_settings['num_arms'] = 10
    exp_settings['barcode_size'] = 10
    exp_settings['num_barcodes'] = 10
    exp_settings['pulls_per_episode'] = 10
    exp_settings['perfect_info'] = False
    exp_settings['reset_barcodes_per_epoch'] = False
    exp_settings['reset_arms_per_epoch'] = True

    # HyperParam Searches for BayesOpt #
    # LSTM Model Info
    exp_settings['dim_hidden_lstm'] = int(dim_hidden_lstm)
    exp_settings['value_error_coef'] = value_error_coef
    exp_settings['entropy_error_coef'] = entropy_error_coef

    # Using ints in bayes-opt for better performance there
    exp_settings['lstm_learning_rate'] = 10**lstm_learning_rate

    # A2C Model Info
    exp_settings['dim_hidden_a2c'] = int(dim_hidden_a2c)

    # Embedder Model Info
    exp_settings['embedding_size'] = 512
    exp_settings['embedder_learning_rate'] = 5e-4
    #End HyperParam Searches for BayesOpt#

    # Current function being used as maximization target is just avg of total epoch returns
    logs, key_data = run_experiment_sl(exp_settings)
    log_return, log_loss_value, log_loss_policy, log_embedder_accuracy, embedder_loss = logs
    keys, prediction_mapping, epoch_mapping, barcode_data = key_data

    # Focusing only on last quarter of returns to maximize longer term learning
    final_q = 3*(exp_settings['epochs']//4)
    return np.mean(log_return[final_q:])

# Bounded region of parameter space
pbounds = { 'dim_hidden_a2c': (64, 512),
            'dim_hidden_lstm': (32, 256),
            'entropy_error_coef': (0, 0.25),
            'lstm_learning_rate': (-5, -2), #transformed into 1e-6 -> 1e-2 in function
            'value_error_coef': (0, 1)}

# Best Results So Far 
# (4 arms/barcodes, 10 pulls, 1500 epochs, last quarter returns, bounds_transformer off)
# | Iter      | Avg_Ret     |A2C Dim    |LSTM Dim   |Ent Coef   |LSTM LR    |Value Coef |
# |  7        |  0.5884     |  102.8    |  76.94    |  0.07705  |  0.000587 |  0.6295   |
# |  10       |  0.5753     |  104.1    |  75.49    |  0.08517  |  0.000494 |  0.2141   |
# |  12       |  0.5728     |  274.1    |  40.43    |  0.06511  |  0.000708 |  0.4685   |

# (4 arms/barcodes, 10 pulls, 2000 Epochs, last quarter returns, bounds_transformer on)
# | Iter      | Avg_Ret     |A2C Dim    |LSTM Dim   |Ent Coef   |LSTM LR    |Value Coef |
# |  21       |  0.592      |  404.8    |  124.4    |  0.07925  |  0.002947 |  0.2597   |
# |  24       |  0.5844     |  399.8    |  120.8    |  0.07298  |  0.002679 |  0.2805   |

# Storing runs because these take so fkn long
# (10 arms/barcodes, 10 pulls, 2000 epochs, last quarter returns, bounds_transformer on)
# | Iter      | Avg_Ret     |A2C Dim    |LSTM Dim   |Ent Coef   |LSTM LR    |Value Coef |
# |  1        |  0.1869     |  250.8    |  193.4    |  2.859e-0 |  0.003024 |  0.1468   |
# |  2        |  0.187      |  105.4    |  73.72    |  0.08639  |  0.003968 |  0.5388   |
# |  3        |  0.1784     |  251.8    |  185.5    |  0.05111  |  0.008781 |  0.02739  |
# |  4        |  0.2878     |  364.4    |  125.5    |  0.1397   |  0.001405 |  0.1981   |
# |  5        |  0.184      |  422.7    |  248.9    |  0.07836  |  0.006924 |  0.8764   |
# |  6        |  0.1967     |  363.8    |  124.7    |  0.1183   |  0.003332 |  0.0559   |

# My next long trials:
# | Iter      | Avg_Ret     |A2C Dim    |LSTM Dim   |Ent Coef   |LSTM LR    |Value Coef |
# |           |             | 103       | 75        | 0.08      | 5e-4      |0.4
# |           |             | 400       | 120       | 0.08      | 2e-3      |0.25

# Does this cause it to converge too quickly?
# bounds_transformer = SequentialDomainReductionTransformer()

optimizer = BayesianOptimization(
    f=avg_returns,
    pbounds=pbounds,
    verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
    # bounds_transformer=bounds_transformer
)

# Suspend/Resume Function for longer iterations
logger = JSONLogger(path="./logs.json")
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
# load_logs(optimizer, logs=["./logs.json"])
print("New optimizer is now aware of {} points.".format(len(optimizer.space)))

# Manual Point addition cause i forgot to set up the logger before running a long ass trial =(
# 10 arms/barcodes/pulls over 2k epochs
obs = [
# | Iter      | Avg_Ret     |A2C Dim    |LSTM Dim   |Ent Coef   |LSTM LR    |Value Coef |
"|  1        |  0.1869     |  250.8    |  193.4    |  2.859e-0 |  0.003024 |  0.1468   |",
"|  2        |  0.187      |  105.4    |  73.72    |  0.08639  |  0.003968 |  0.5388   |",
"|  3        |  0.1784     |  251.8    |  185.5    |  0.05111  |  0.008781 |  0.02739  |",
"|  4        |  0.2878     |  364.4    |  125.5    |  0.1397   |  0.001405 |  0.1981   |",
"|  5        |  0.184      |  422.7    |  248.9    |  0.07836  |  0.006924 |  0.8764   |",
"|  6        |  0.1967     |  363.8    |  124.7    |  0.1183   |  0.003332 |  0.0559   |"
]

for val in obs:
    val_split = val.split('|')
    tar = float(val_split[2])
    params = [float(val_split[i]) for i in range(3,len(val_split)-1)]
    # print(tar)
    # print(params) 
    optimizer.register(params, tar)

print("New optimizer is now aware of {} points.".format(len(optimizer.space)))

optimizer.maximize(
    init_points=2,
    n_iter=8,
)

# # for i, res in enumerate(optimizer.res):
# #     print("Iteration {}: \n\t{}".format(i, res))

print(" *-* "*5)    
print(optimizer.max)