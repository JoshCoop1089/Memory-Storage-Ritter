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

    # Experimental Parameters
    exp_settings = {}
    exp_settings['randomize'] = False
    exp_settings['tensorboard_logging'] = False
    exp_settings['timing'] = False
    exp_settings['lstm_inputs_looped'] = True

    # Task Info
    exp_settings['kernel'] = 'cosine'           #cosine, l2
    exp_settings['agent_input'] = 'obs/context' #obs, obs/context
    exp_settings['mem_store'] = 'context'   #obs/context, context, embedding, obs, hidden (unsure how to do obs, hidden return calc w/o barcode predictions)
    exp_settings['task_version'] = 'bandit'      #bandit, original
    exp_settings['noise_percent'] = 0.5

    # 1000 epochs with 10 barcodes == 100k episodes
    # Ritter returns were averaging ~0.35 at this point in training
    exp_settings['epochs'] = 1000
    exp_settings['num_arms'] = 4
    exp_settings['barcode_size'] = 4
    exp_settings['num_barcodes'] = 4
    exp_settings['pulls_per_episode'] = 10
    exp_settings['perfect_info'] = False
    exp_settings['reset_barcodes_per_epoch'] = False
    exp_settings['reset_arms_per_epoch'] = True

    # HyperParam Searches for BayesOpt #
    # Using ints in bayes-opt for better performance
    exp_settings['dim_hidden_lstm'] = int(2**dim_hidden_lstm)
    exp_settings['value_error_coef'] = value_error_coef
    exp_settings['entropy_error_coef'] = entropy_error_coef
    exp_settings['lstm_learning_rate'] = 10**lstm_learning_rate
    exp_settings['dim_hidden_a2c'] = int(2**dim_hidden_a2c)
    # exp_settings['embedder_learning_rate'] = 10**embedding_learning_rate
    # exp_settings['embedding_size'] = int(2**embedding_size)

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
    # print(f"Emb_LR: {exp_settings['embedder_learning_rate']} | Emb_Size: {exp_settings['embedding_size']}")
    
    # Current function being used as maximization target is just avg of total epoch returns
    logs, key_data = run_experiment_sl(exp_settings)
    log_return, log_loss_value, log_loss_policy, log_loss_total, log_embedder_accuracy, embedder_loss = logs
    keys, prediction_mapping, epoch_mapping, barcode_data = key_data

    # Focusing only on last quarter of returns to maximize longer term learning
    final_q = 3*(exp_settings['epochs']//4)
    return np.mean(log_return[final_q:])

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
            'dim_hidden_a2c': (5, 9),               #transformed into 2**x in function
            'dim_hidden_lstm': (5, 9),              #transformed into 2**x in function
            'entropy_error_coef': (0, 0.5),
            'lstm_learning_rate': (-5, -2),         #transformed into 10**x in function
            'value_error_coef': (0, 0.75),
            # 'embedding_learning_rate': (-5, -2),    #transformed into 10**x in function
            # 'embedding_size': (5,10),               #transformed into 2**x in function
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

# 4 arms/barcodes, 10 pulls
# load_logs(optimizer, logs=[
#           "./dnd-lstm-sup-learning/src/logs_4_1k_epochs.json"])
logger = JSONLogger(
    path="./dnd-lstm-sup-learning/src/logs_4_1k_epochs_loop.json", reset=False)

# 10 arms/barcodes/pulls
# load_logs(optimizer, logs=["./dnd-lstm-sup-learning/src/logs_10.json"])
# logger = JSONLogger(path="./dnd-lstm-sup-learning/src/logs_10_loss_coefs.json", reset=False)

optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
print("New optimizer is now aware of {} points.".format(len(optimizer.space)))

# # # Manual Point addition cause i forgot to set up the logger before running a long ass trial =(
# # # 10 arms/barcodes/pulls over 2k epochs
# obs = [
# # | Iter      | Avg_Ret     |A2C Dim    |LSTM Dim   |Ent Coef   |LSTM LR    |Value Coef |
# "|  1        |  0.1869     |  250.8    |  193.4    |  2.859e-0 |  -2.519418213170831 |  0.1468   |",
# "|  2        |  0.187      |  105.4    |  73.72    |  0.08639  |  -2.401428336517859 |  0.5388   |",
# "|  3        |  0.1784     |  251.8    |  185.5    |  0.05111  |  -2.0564560228465454 |  0.02739  |",
# "|  4        |  0.2878     |  364.4    |  125.5    |  0.1397   |  -2.8523236757589014 |  0.1981   |",
# "|  5        |  0.184      |  422.7    |  248.9    |  0.07836  |  -2.159642940796644 |  0.8764   |",
# "|  6        |  0.1967     |  363.8    |  124.7    |  0.1183   |  -2.47729500726525 |  0.0559   |"
# ]

# {"target": 0.1890480000000001, "params": {"dim_hidden_a2c": 360.28050479192956, "dim_hidden_lstm": 144.9514141076652, "entropy_error_coef": 0.06526811936609955, "lstm_learning_rate": -3.1330933661544993, "value_error_coef": 0.40681007049253626}, "datetime": {"datetime": "2022-06-23 07:23:34", "elapsed": 0.0, "delta": 0.0}}
# {"target": 0.1869, "params": {"dim_hidden_a2c": 250.8, "dim_hidden_lstm": 193.4, "entropy_error_coef": 2.859, "lstm_learning_rate": -2.519418213170831, "value_error_coef": 0.1468}, "datetime": {"datetime": "2022-06-24 16:46:43", "elapsed": 0.0, "delta": 0.0}}
# {"target": 0.187, "params": {"dim_hidden_a2c": 105.4, "dim_hidden_lstm": 73.72, "entropy_error_coef": 0.08639, "lstm_learning_rate": -2.401428336517859, "value_error_coef": 0.5388}, "datetime": {"datetime": "2022-06-24 16:46:43", "elapsed": 0.000998, "delta": 0.000998}}
# {"target": 0.1784, "params": {"dim_hidden_a2c": 251.8, "dim_hidden_lstm": 185.5, "entropy_error_coef": 0.05111, "lstm_learning_rate": -2.0564560228465454, "value_error_coef": 0.02739}, "datetime": {"datetime": "2022-06-24 16:46:43", "elapsed": 0.000998, "delta": 0.0}}
# {"target": 0.2878, "params": {"dim_hidden_a2c": 364.4, "dim_hidden_lstm": 125.5, "entropy_error_coef": 0.1397, "lstm_learning_rate": -2.8523236757589014, "value_error_coef": 0.1981}, "datetime": {"datetime": "2022-06-24 16:46:43", "elapsed": 0.001999, "delta": 0.001001}}
# {"target": 0.184, "params": {"dim_hidden_a2c": 422.7, "dim_hidden_lstm": 248.9, "entropy_error_coef": 0.07836, "lstm_learning_rate": -2.159642940796644, "value_error_coef": 0.8764}, "datetime": {"datetime": "2022-06-24 16:46:43", "elapsed": 0.001999, "delta": 0.0}}
# {"target": 0.1967, "params": {"dim_hidden_a2c": 363.8, "dim_hidden_lstm": 124.7, "entropy_error_coef": 0.1183, "lstm_learning_rate": -2.47729500726525, "value_error_coef": 0.0559}, "datetime": {"datetime": "2022-06-24 16:46:43", "elapsed": 0.002999, "delta": 0.001}}
# {"target": 0.1785413333333334, "params": {"dim_hidden_a2c": 250.82585810675315, "dim_hidden_lstm": 193.35268653104342, "entropy_error_coef": 2.859370433622166e-05, "lstm_learning_rate": -4.0930022821044805, "value_error_coef": 0.14675589081711304}, "datetime": {"datetime": "2022-06-24 17:53:59", "elapsed": 4036.367885, "delta": 4036.364886}}
# {"target": 0.1794506666666668, "params": {"dim_hidden_a2c": 105.36769045642141, "dim_hidden_lstm": 73.72228734859829, "entropy_error_coef": 0.08639018176076194, "lstm_learning_rate": -3.80969757730799, "value_error_coef": 0.538816734003357}, "datetime": {"datetime": "2022-06-24 18:53:09", "elapsed": 7586.800449, "delta": 3550.432564}}
# {"target": 0.17880800000000013, "params": {"dim_hidden_a2c": 364.3694445599242, "dim_hidden_lstm": 125.47627573023644, "entropy_error_coef": 0.13967245711143791, "lstm_learning_rate": -4.578839184214298, "value_error_coef": 0.1981014890848788}, "datetime": {"datetime": "2022-06-25 00:58:52", "elapsed": 3690.115813, "delta": 3690.114795}}
# {"target": 0.1900160000000001, "params": {"dim_hidden_a2c": 198.3545875221977, "dim_hidden_lstm": 134.69564927422536, "entropy_error_coef": 0.14653443867158242, "lstm_learning_rate": -2.770736961792505, "value_error_coef": 0.47419758533659195}, "datetime": {"datetime": "2022-06-25 02:00:11", "elapsed": 7368.79629, "delta": 3678.680477}}
# {"target": 0.1881920000000001, "params": {"dim_hidden_a2c": 361.9867714168124, "dim_hidden_lstm": 125.75799619296069, "entropy_error_coef": 0.10399705748487956, "lstm_learning_rate": -2.4802976731528377, "value_error_coef": 0.9003656997697922}, "datetime": {"datetime": "2022-06-25 03:01:29", "elapsed": 11046.485084, "delta": 3677.688794}}
# {"target": 0.1849786666666668, "params": {"dim_hidden_a2c": 238.03793802874452, "dim_hidden_lstm": 239.3684941387513, "entropy_error_coef": 0.21219179505158609, "lstm_learning_rate": -3.2403981549142262, "value_error_coef": 0.2193865132025623}, "datetime": {"datetime": "2022-06-25 04:25:12", "elapsed": 16070.044709, "delta": 5023.559625}}

# for val in obs:
#     val_split = val.split('|')
#     tar = float(val_split[2])
#     params = [float(val_split[i]) for i in range(3,len(val_split)-1)]
#     optimizer.register(params, tar)

# print("New optimizer is now aware of {} points.".format(len(optimizer.space)))

optimizer.maximize(
    init_points=10,
    n_iter=30,
)

# # # for i, res in enumerate(optimizer.res):
# # #     print("Iteration {}: \n\t{}".format(i, res))

print(" *-* "*5)    
print(optimizer.max)