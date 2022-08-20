from contextual_choice_sl import run_experiment_sl
from bayes_opt import BayesianOptimization
from bayes_opt import SequentialDomainReductionTransformer

from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

import numpy as np

def avg_returns(dim_hidden_lstm = 0, lstm_learning_rate = 0, dim_hidden_a2c = 0, 
                value_error_coef = 0, entropy_error_coef = 0,
                embedding_size = 0, embedding_learning_rate = 0):
    exp_settings = {}

    ### Experimental Parameters ###
    exp_settings['randomize'] = False
    exp_settings['perfect_info'] = False
    exp_settings['reset_barcodes_per_epoch'] = False
    exp_settings['reset_arms_per_epoch'] = True
    exp_settings['lstm_inputs_looped'] = True       # Use action predictions from lstm as next input, instead of predetermined pulls
    exp_settings['torch_device'] = 'CPU'            # 'CPU' or 'GPU'

    # Task Info
    exp_settings['kernel'] = 'cosine'               # Cosine, l2
    exp_settings['agent_input'] = 'obs/context'     # Obs, obs/context
    exp_settings['mem_store'] = 'embedding'           # Context, embedding, obs/context, obs, hidden (unsure how to do obs, hidden return calc w/o barcode predictions)
    exp_settings['task_version'] = 'bandit'         # Bandit, original

    # Task Complexity
    exp_settings['noise_percent'] = 0.5
    exp_settings['num_arms'] = 5
    exp_settings['barcode_size'] = 5
    exp_settings['num_barcodes'] = 5
    exp_settings['pulls_per_episode'] = 10
    exp_settings['epochs'] = 300

    # Data Logging
    exp_settings['tensorboard_logging'] = False
    exp_settings['timing'] = False
    ### End of Experimental Parameters ###


    # HyperParam Searches for BayesOpt #
    # Using ints in bayes-opt for better performance
    # exp_settings['dim_hidden_lstm'] = int(2**dim_hidden_lstm)
    # exp_settings['value_error_coef'] = value_error_coef
    # exp_settings['entropy_error_coef'] = entropy_error_coef
    # exp_settings['lstm_learning_rate'] = 10**lstm_learning_rate
    # exp_settings['dim_hidden_a2c'] = int(2**dim_hidden_a2c)
    # exp_settings['embedder_learning_rate'] = 10**embedding_learning_rate
    # exp_settings['embedding_size'] = int(2**embedding_size)

    exp_settings['dim_hidden_a2c'] = int(2**6.909)        #120
    exp_settings['dim_hidden_lstm'] = int(2**5.302)       #39
    exp_settings['entropy_error_coef'] = 0.0641
    exp_settings['lstm_learning_rate'] = 10**-2.668       #2.1e-3
    exp_settings['value_error_coef'] = 0.335

    # LSTM/A2C Model Info
    # exp_settings['dim_hidden_a2c'] = 364
    # exp_settings['dim_hidden_lstm'] = 125
    # exp_settings['entropy_error_coef'] = .054388
    # exp_settings['lstm_learning_rate'] = 0.001405
    # exp_settings['value_error_coef'] = .276659

    # Embedder Model Info
    exp_settings['embedder_learning_rate'] = 10**embedding_learning_rate
    exp_settings['embedding_size'] = int(2**embedding_size)
    #End HyperParam Searches for BayesOpt#

    # Print out current hyperparams to console
    print("\nNext Run Commencing with the following params:")
    print(f"A2C_Size: {exp_settings['dim_hidden_a2c']} | LSTM_Size: {exp_settings['dim_hidden_lstm']} | LSTM_LR: {round(exp_settings['lstm_learning_rate'], 5)}")
    print(f"Value_Coef: {round(exp_settings['value_error_coef'], 4)} | Entropy_Coef: {round(exp_settings['entropy_error_coef'], 4)}")
    print(f"Emb_LR: {round(exp_settings['embedder_learning_rate'], 5)} | Emb_Size: {exp_settings['embedding_size']}")
    
    # Current function being used as maximization target is just avg of total epoch returns
    logs, key_data = run_experiment_sl(exp_settings)
    log_return, log_loss_value, log_loss_policy, log_loss_total, log_embedder_accuracy, embedder_loss, log_memory_accuracy = logs
    keys, prediction_mapping, epoch_mapping, barcode_data = key_data

    # Focusing only on last quarter of returns to maximize longer term learning
    final_q = 3*(exp_settings['epochs']//4)
    accuracy_scaler = log_memory_accuracy[-1]/0.8
    target = np.mean(log_return[final_q:])*accuracy_scaler
    print(f"Bayes Target = {round(target, 3)} | Accuracy Scaling = {round(accuracy_scaler, 3)}")
    return target

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

# (4 arms/barcodes, 10 pulls, 1000 Epochs, last quarter returns)
# {"target": 0.6357, "params": {"dim_hidden_a2c": 6.8757, "dim_hidden_lstm": 5.15,
#  "entropy_error_coef": 0.0651, "lstm_learning_rate": -2.883, "value_error_coef": 0.3514},

# (10 arms/barcodes, 10 pulls, 2000 epochs, last quarter returns, bounds_transformer on)
# | Iter      | Avg_Ret     |A2C Dim    |LSTM Dim   |Ent Coef   |LSTM LR    |Value Coef |
# |  1        |  0.1869     |  250.8    |  193.4    |  2.859e-0 |  0.003024 |  0.1468   |
# |  2        |  0.187      |  105.4    |  73.72    |  0.08639  |  0.003968 |  0.5388   |
# |  3        |  0.1784     |  251.8    |  185.5    |  0.05111  |  0.008781 |  0.02739  |
# |  4        |  0.2878     |  364.4    |  125.5    |  0.1397   |  0.001405 |  0.1981   |
# |  5        |  0.184      |  422.7    |  248.9    |  0.07836  |  0.006924 |  0.8764   |
# |  6        |  0.1967     |  363.8    |  124.7    |  0.1183   |  0.003332 |  0.0559   |

# 10abp
# | Iter      | Avg_Ret     |A2C Dim    |LSTM Dim   |Ent Coef   |LSTM LR (10**x)|Value Coef |
# |  ??       |  0.32       |  364.4    |  125.5    |  0.05439  |  -2.852       |  0.276659 |

# Embedder Model 
# Best Result for 4ab/10p    
# {"target": 0.753, "params": {"embedding_learning_rate": -3.0399, "embedding_size": 8.629}
    
# Bounded region of parameter space
pbounds = { 
            # 'dim_hidden_a2c': (5, 8),               #transformed into 2**x in function
            # 'dim_hidden_lstm': (5, 8),              #transformed into 2**x in function
            # 'entropy_error_coef': (0, 0.5),
            # 'lstm_learning_rate': (-5, -2),         #transformed into 10**x in function
            # 'value_error_coef': (0, 0.75),
            'embedding_learning_rate': (-5, -2),    #transformed into 10**x in function
            'embedding_size': (7,11),               #transformed into 2**x in function
            }

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
# logger = JSONLogger(
#     path="./dnd-lstm-sup-learning/src/logs_5_1k_epochs.json", reset=False)

# optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
# print("New optimizer is now aware of {} points.".format(len(optimizer.space)))

optimizer.maximize(
    init_points=4,
    n_iter=12,
)

print(" *-* "*5)    
print(optimizer.max)