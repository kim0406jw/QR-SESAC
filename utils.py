import numpy as np
import random
import torch
import torch.nn as nn

from environment.inverted_pendulum import InvertedPendulumSwing
from environment.triple_inverted_pendulum import TripleInvertedPendulumSwing


def cal_critic_loss(v, target_v, quantile_critic, sum_over_quantiles=True):
    """
    The quantile-regression loss, as described in the QR-DQN and TQC papers.
    Partially taken from https://github.com/bayesgroup/tqc_pytorch.
    :param current_quantiles: current estimate of quantiles, must be either
        (batch_size, n_quantiles) or (batch_size, n_critics, n_quantiles)
    :param target_quantiles: target of quantiles, must be either (batch_size, n_target_quantiles),
        (batch_size, 1, n_target_quantiles), or (batch_size, n_critics, n_target_quantiles)
    :param cum_prob: cumulative probabilities to calculate quantiles (also called midpoints in QR-DQN paper),
        must be either (batch_size, n_quantiles), (batch_size, 1, n_quantiles), or (batch_size, n_critics, n_quantiles).
        (if None, calculating unit quantiles)
    :param sum_over_quantiles: if summing over the quantile dimension or not
    :return: the loss
    """
    if quantile_critic:
        n_quantiles = v.shape[-1]
        current_quantiles = v
        target_quantiles = target_v

        cum_prob = (torch.arange(n_quantiles, device=current_quantiles.device, dtype=torch.float) + 0.5) / n_quantiles
        if current_quantiles.ndim == 2:
            # For QR-DQN, current_quantiles have a shape (batch_size, n_quantiles), and make cum_prob
            # broadcastable to (batch_size, n_quantiles, n_target_quantiles)
            cum_prob = cum_prob.view(1, -1, 1)
        elif current_quantiles.ndim == 3:
            # For TQC, current_quantiles have a shape (batch_size, n_critics, n_quantiles), and make cum_prob
            # broadcastable to (batch_size, n_critics, n_quantiles, n_target_quantiles)
            cum_prob = cum_prob.view(1, 1, -1, 1)

        # QR-DQN
        # target_quantiles: (batch_size, n_target_quantiles) -> (batch_size, 1, n_target_quantiles)
        # current_quantiles: (batch_size, n_quantiles) -> (batch_size, n_quantiles, 1)
        # pairwise_delta: (batch_size, n_target_quantiles, n_quantiles)
        # TQC
        # target_quantiles: (batch_size, 1, n_target_quantiles) -> (batch_size, 1, 1, n_target_quantiles)
        # current_quantiles: (batch_size, n_critics, n_quantiles) -> (batch_size, n_critics, n_quantiles, 1)
        # pairwise_delta: (batch_size, n_critics, n_quantiles, n_target_quantiles)
        # Note: in both cases, the loss has the same shape as pairwise_delta

        pairwise_delta = target_quantiles.unsqueeze(-2) - current_quantiles.unsqueeze(-1)
        abs_pairwise_delta = torch.abs(pairwise_delta)
        huber_loss = torch.where(abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta ** 2 * 0.5)
        loss = torch.abs(cum_prob - (pairwise_delta.detach() < 0).float()) * huber_loss

        if sum_over_quantiles:
            critic_loss = loss.sum(dim=-2).mean()
        else:
            critic_loss = loss.mean()

    else:
        critic_loss = ((v - target_v)**2).mean()

    return critic_loss


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers.
        Reference: https://github.com/MishaLaskin/rad/blob/master/curl_sac.py"""

    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        #m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def hard_update(network, target_network):
    with torch.no_grad():
        for param, target_param in zip(network.parameters(), target_network.parameters()):
            target_param.data.copy_(param.data)


def soft_update(network, target_network, tau):
    with torch.no_grad():
        for param, target_param in zip(network.parameters(), target_network.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def set_seed(random_seed):
    if random_seed <= 0:
        random_seed = np.random.randint(1, 9999)
    else:
        random_seed = random_seed

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    return random_seed


def make_env(env_name, random_seed):
    set_seed(random_seed)

    if env_name == "TripleInvertedPendulumSwing":
        env = TripleInvertedPendulumSwing()
        eval_env = TripleInvertedPendulumSwing()
    elif env_name == "InvertedPendulumSwing":
        env = InvertedPendulumSwing()
        eval_env = InvertedPendulumSwing()
    else:
        raise NotImplementedError

    return env, eval_env


def log_to_txt(args, total_step, result):
    env_name = args.env_name
    seed = '(' + str(args.random_seed) + ')'
    f = open('./log/' + env_name + '_seed' + seed + '.txt', 'a')
    log = str(total_step) + ' ' + str(result) + '\n'
    f.write(log)
    f.close()
