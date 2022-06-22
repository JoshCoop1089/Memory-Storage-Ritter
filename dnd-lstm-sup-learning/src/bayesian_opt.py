from contextual_choice_sl import run_experiment_sl
from bayes_opt import BayesianOptimization
from bayes_opt import SequentialDomainReductionTransformer

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
    exp_settings['epochs'] = 2000
    exp_settings['num_arms'] = 4
    exp_settings['barcode_size'] = 4
    exp_settings['num_barcodes'] = 4
    exp_settings['pulls_per_episode'] = 10
    exp_settings['perfect_info'] = False
    exp_settings['reset_barcodes_per_epoch'] = False
    exp_settings['reset_arms_per_epoch'] = True

    # HyperParam Searches for BayesOpt #
    # LSTM Model Info
    exp_settings['dim_hidden_lstm'] = int(dim_hidden_lstm)
    exp_settings['lstm_learning_rate'] = lstm_learning_rate
    exp_settings['value_error_coef'] = value_error_coef
    exp_settings['entropy_error_coef'] = entropy_error_coef

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
            'lstm_learning_rate': (1e-6, 1e-2),
            'value_error_coef': (0, 1)}

# Best Results So Far (4 arms/barcodes, 10 pulls, 1500 epochs, last quarter returns)
# | Iter      | Avg_Ret     |A2C Dim    |LSTM Dim   |Ent Coef   |LSTM LR    |Value Coef |
# |  7        |  0.5884     |  102.8    |  76.94    |  0.07705  |  0.000587 |  0.6295   |
# |  10       |  0.5753     |  104.1    |  75.49    |  0.08517  |  0.000494 |  0.2141   |
# |  12       |  0.5728     |  274.1    |  40.43    |  0.06511  |  0.000708 |  0.4685   |

# My next long trial:
# |           |             | 103       | 75        | 0.08      | 5e-4      |?? 0.4?

bounds_transformer = SequentialDomainReductionTransformer()

optimizer = BayesianOptimization(
    f=avg_returns,
    pbounds=pbounds,
    verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
    bounds_transformer=bounds_transformer
)


optimizer.maximize(
    init_points=5,
    n_iter=20,
)


# for i, res in enumerate(optimizer.res):
#     print("Iteration {}: \n\t{}".format(i, res))

print(" *-* "*5)    
print(optimizer.max)
