import torch
import numpy as np
from torch.nn.functional import smooth_l1_loss

'''helpers'''
eps = np.finfo(np.float32).eps.item()
def compute_returns(rewards, gamma=0, normalize=False):
    """compute return in the standard policy gradient setting.

    Parameters
    ----------
    rewards : list, 1d array
        immediate reward at time t, for all t
    gamma : float, [0,1]
        temporal discount factor
    normalize : bool
        whether to normalize the return
        - default to false, because we care about absolute scales

    Returns
    -------
    1d torch.tensor
        the sequence of cumulative return

    """
    R = 0
    returns = []
    for r in rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    if normalize:
        returns = (returns - returns.mean()) / (returns.std() + eps)
    return returns

def get_reward(a_t, a_t_targ):
    """define the reward function at time t

    Parameters
    ----------
    a_t : int
        action
    a_t_targ : int
        target action

    Returns
    -------
    torch.FloatTensor, scalar
        immediate reward at time t

    """
    if a_t == a_t_targ:
        r_t = 1
    else:
        r_t = 0
    return torch.tensor(r_t).type(torch.FloatTensor).data

def get_reward_from_assumed_barcode(a_t, reward_from_obs, assumed_barcode, mapping, perfect_info = False):
    """
    Uses the embedding models prediction of a barcode to pull a specific arm.
    The reward from that pull is checked against the reward from the observation.
    Actual LSTM reward is if those two rewards match each other.
    """
    try:
        best_arm = mapping[assumed_barcode]
        if perfect_info == False:
            if a_t == best_arm:
                reward = torch.tensor([int(np.random.random() < 0.9)])
            else:
                reward = torch.tensor([int(np.random.random() < 0.1)])

        # Deterministic Arm Rewards (for debugging purposes)
        # Make sure to change generate_one_episode in ContextBandits.py as well
        else:  # perfect_info == True
            reward = torch.tensor([int(a_t == best_arm)])

    # Penalize a predicted barcode which isn't a part of the mapping for the epoch
    except Exception:
            reward = torch.tensor([-1])

    # print("P-R:", reward, "R-R:", reward_from_obs)
    matched_reward = int(torch.equal(reward, reward_from_obs))
    return torch.tensor(matched_reward).type(torch.FloatTensor).data

def compute_a2c_loss(probs, values, returns):
    """compute the objective node for policy/value networks

    Parameters
    ----------
    probs : list
        action prob at time t
    values : list
        state value at time t
    returns : list
        return at time t

    Returns
    -------
    torch.tensor, torch.tensor
        Description of returned object.

    """
    policy_grads, value_losses = [], []
    for prob_t, v_t, R_t in zip(probs, values, returns):
        A_t = R_t - v_t.item()
        policy_grads.append(-prob_t * A_t)
        value_losses.append(
            smooth_l1_loss(torch.squeeze(v_t), torch.squeeze(R_t))
        )
    loss_policy = torch.stack(policy_grads).sum()
    loss_value = torch.stack(value_losses).sum()
    return loss_policy, loss_value