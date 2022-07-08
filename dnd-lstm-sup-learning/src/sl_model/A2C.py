import torch
import torch.nn as nn
import torch.nn.functional as F

class A2C(nn.Module):
    """a MLP actor-critic network
    process: relu(Wx) -> pi, v

    Parameters
    ----------
    dim_input : int
        dim state space
    dim_hidden : int
        number of hidden units
    dim_output : int
        dim action space

    Attributes
    ----------
    ih : torch.nn.Linear
        input to hidden mapping
    actor : torch.nn.Linear
        the actor network
    critic : torch.nn.Linear
        the critic network
    _init_weights : helper func
        default weight init scheme

    """

    def __init__(self, dim_input, dim_hidden, dim_output, device):
        super(A2C, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_hidden = dim_hidden

        # Shared Feature Extractor
        self.ih = nn.Linear(self.dim_input, self.dim_hidden, device = device)

        # Actor/Policy Network
        self.actor = nn.Linear(self.dim_hidden, self.dim_output, device = device)

        # Critic/Value Network
        self.critic = nn.Linear(self.dim_hidden, 1, device = device)
        # ortho_init(self)

    def forward(self, x, beta=1):
        """compute action distribution and value estimate, pi(a|s), v(s)

        Parameters
        ----------
        x : a vector
            a vector, state representation
        beta : float, >0
            softmax temp, big value -> more "randomness"

        Returns
        -------
        vector, scalar
            pi(a|s), v(s)

        """
        
        # Should the first layer be shared?
        h = F.leaky_relu(self.ih(x))

        # Critic Network
        value_estimate = self.critic(h)

        # Actor Network
        action_distribution = softmax(self.actor(h), beta)

        # Entropy caluclation for exploration
        dist = torch.distributions.Categorical(action_distribution)
        entropy = dist.entropy().mean()

        return action_distribution, value_estimate, entropy

class A2C_linear(nn.Module):
    """a linear actor-critic network
    process: x -> pi, v

    Parameters
    ----------
    dim_input : int
        dim state space
    dim_output : int
        dim action space

    Attributes
    ----------
    actor : torch.nn.Linear
        the actor network
    critic : torch.nn.Linear
        the critic network

    """

    def __init__(self, dim_input, dim_output):
        super(A2C_linear, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.actor = nn.Linear(dim_input, dim_output)
        self.critic = nn.Linear(dim_input, 1)

    def forward(self, x, beta = 1):
        """compute action distribution and value estimate, pi(a|s), v(s)

        Parameters
        ----------
        x : a vector
            a vector, state representation
        beta : float, >0
            softmax temp, big value -> more "randomness"

        Returns
        -------
        vector, scalar
            pi(a|s), v(s)

        """
        action_distribution = softmax(self.actor(x), beta)
        value_estimate = self.critic(x)

        # Entropy caluclation for exploration
        dist = torch.distributions.Categorical(action_distribution)
        entropy = dist.entropy().mean()
        return action_distribution, value_estimate, entropy


def softmax(z, beta):
    """helper function, softmax with beta

    Parameters
    ----------
    z : torch tensor, has 1d underlying structure after torch.squeeze
        the raw logits
    beta : float, >0
        softmax temp, big value -> more "randomness"

    Returns
    -------
    1d torch tensor
        a probability distribution | beta

    """
    assert beta > 0
    return F.softmax(torch.squeeze(z / beta), dim=0)
