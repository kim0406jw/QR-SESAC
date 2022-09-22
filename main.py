import argparse
import torch
from sac import SAC
from trainer import Trainer
from utils import set_seed, make_env

'''
    env_name: Name of environment.
    random_seed: Seed for random number generators.
    eval_flag: If true, perform evaluation.
    eval_freq: Frequency in performance evaluation.
    eval_episode: Number of episodes to test the deterministic policy at the end of each epoch.
    automating_temperature: If true, adjust value of entropy-temperature parameter automatically.
    temperature: The temperature parameter at constant temperature setting. 
    start_step: Number of steps for uniform-random action selection, before running real policy. 
    max_step: Maximum steps of interaction. 
    update_after: Number of env interactions to collect before starting to do gradient descent updates.
                  (Ensures replay buffer is full enough for useful updates.)
    hidden_dims: Dimension of hidden layers of networks.
    batch_size: Mini-batch size for SGD.
    buffer_size: Maximum replay buffer size.
    update_every: Number of env interactions that should elapse between gradient descent updates.
                  (The ratio of env steps to gradient steps is locked to 1.)
    log_std_bound: Bound of log_std from the GaussianPolicy. <=> [log_std_min, log_std_max]
    gamma: Discount factor.
    actor_lr: Learning rate for policy.
    critic_lr: Learning rate for Q-networks.
    tau: Interpolation factor in polyak averaging for networks and target networks.
    log: If true, log the evaluation result as txt file. (./log/...txt)
    show_loss: If true, show actor ana critic loss in training step

'''


def get_parameters():
    parser = argparse.ArgumentParser()

    parser.add_argument('--env-name', default="TripleInvertedPendulumSwing")
    parser.add_argument('--random-seed', default=-1, type=int)
    parser.add_argument('--eval_flag', default=True, type=bool)
    parser.add_argument('--eval-freq', default=5000, type=int)
    parser.add_argument('--eval-episode', default=5, type=int)

    parser.add_argument('--automating-temperature', default=True, type=bool)
    parser.add_argument('--temperature', default=0.2, type=float)
    parser.add_argument('--start-step', default=10000, type=int)
    parser.add_argument('--max-step', default=1000000, type=int)
    parser.add_argument('--update_after', default=1000, type=int)
    parser.add_argument('--hidden-dims', default=(256, 256))
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--buffer-size', default=1000000, type=int)
    parser.add_argument('--update-every', default=1, type=int)
    parser.add_argument('--log_std_bound', default=[-20, 2])
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--actor-lr', default=3e-4, type=float)  # 3e-4
    parser.add_argument('--critic-lr', default=3e-4, type=float)  # 3e-4
    parser.add_argument('--temperature-lr', default=3e-4, type=float)  # 3e-4
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--log', default=True, type=bool)
    parser.add_argument('--show-loss', default=False, type=bool)
    parser.add_argument('--save-model', default=True, type=bool)

    parser.add_argument('--n-update', default=2, type=int)
    parser.add_argument('--n-supports', default=32, type=int)

    parser.add_argument('--quantile-critic', default=True, type=bool)

    param = parser.parse_args()

    return param


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.random_seed = set_seed(args.random_seed)
    env, eval_env = make_env(args.env_name, args.random_seed)

    print("Device:", device, "\nRandom Seed:", args.random_seed, "\nEnvironment:", args.env_name,
          '\nAdjust Temperature Automatically:', args.automating_temperature, '\nActor_lr:', args.actor_lr,
          '\nCritic_lr:', args.critic_lr, '\nTemperature_lr', args.temperature_lr, '\n')

    state_dim = env.state_dim
    action_dim = env.action_dim
    action_bound = [-1, 1]

    eqi_idx = env.eqi_idx
    reg_idx = env.reg_idx

    agent = SAC(args, state_dim, action_dim, eqi_idx, reg_idx, action_bound, device)

    trainer = Trainer(env, eval_env, agent, args)
    trainer.run()


if __name__ == '__main__':
    args = get_parameters()
    main(args)
