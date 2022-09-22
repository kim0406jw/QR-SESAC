import torch
from main import get_parameters
from sac import SAC
from utils import make_env, set_seed
import matplotlib.pyplot as plt
import numpy as np

# Hyper-parameters setting.
LOAD_BEST_MODEL = False
args = get_parameters()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random_seed = set_seed(args.random_seed)

# Declare the TripleInvertedPendulum environment.
env, _ = make_env(args.env_name, random_seed)
state_dim = env.state_dim
action_dim = env.action_dim
action_bound = [-1, 1]
eqi_idx = env.eqi_idx
reg_idx = env.reg_idx

# Declare the SAC agent.
agent = SAC(args, state_dim, action_dim, eqi_idx, reg_idx, action_bound, device)

# Load the trained networks.
agent.actor.to('cpu')
agent.critic.to('cpu')
agent.target_critic.to('cpu')
if LOAD_BEST_MODEL:
    agent.actor.load_state_dict(torch.load('./save/best_model/actor.pt'))
    agent.critic.load_state_dict(torch.load('./save/best_model/critic.pt'))
    agent.target_critic.load_state_dict(torch.load('./save/best_model/target_critic.pt'))
else:
    agent.actor.load_state_dict(torch.load('./save/model/actor.pt'))
    agent.critic.load_state_dict(torch.load('./save/model/critic.pt'))
    agent.target_critic.load_state_dict(torch.load('./save/model/target_critic.pt'))
agent.actor.to('cuda')
agent.critic.to('cuda')
agent.target_critic.to('cuda')

# Test starts here.
n_eval = 10
for i in range(n_eval):
    state = env.test_reset(idx=i)
    episode_reward = 0
    if i % 2 == 0:
        even_traj = []
        even_act_traj = []
    else:
        odd_traj = []
        odd_act_traj = []
    for step in range(1000):
        action = agent.get_action(state, evaluation=True)
        if i % 2 == 0:
            even_traj.append(state[0])
            even_act_traj.append(action)
        else:
            odd_traj.append(state[0])
            odd_act_traj.append(action)
        next_state, reward, done, true_done = env.step(env.action_max * action)
        episode_reward += reward
        state = next_state
        if done: break
    if i > 0:
        # plt.plot(np.array(odd_traj), c='k')
        plt.plot(np.array(even_traj) + np.array(odd_traj), c='k')
        plt.plot(np.array(even_act_traj) + np.array(odd_act_traj), c='r')
        plt.show()
    print("[EPI %d] %.3f"%(i+1, episode_reward))