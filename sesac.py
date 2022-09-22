import torch
import torch.nn.functional as F
from replay_memory import ReplayMemory
from network import EQI_GaussianPolicy, INV_Twin_Q_net
from utils import hard_update, soft_update, cal_critic_loss


class SESAC:
    def __init__(self, args, state_dim, action_dim, inv_idx, reg_idx, action_bound, device):
        self.args = args

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound

        self.device = device
        self.buffer = ReplayMemory(state_dim, action_dim, device, args.buffer_size)
        self.batch_size = args.batch_size

        self.gamma = args.gamma
        self.tau = args.tau

        self.n_supports = args.n_supports

        self.actor = EQI_GaussianPolicy(args, inv_idx, reg_idx, action_dim, action_bound, args.hidden_dims, F.relu, device).to(device)
        self.critic = INV_Twin_Q_net(args, inv_idx, reg_idx, action_dim, (512, 512, 512), F.relu, device).to(device)
        self.target_critic = INV_Twin_Q_net(args, inv_idx, reg_idx, action_dim,  (512, 512, 512), F.relu, device).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        # Automating Entropy Adjustment for Maximum Entropy RL
        if args.automating_temperature is True:
            self.target_entropy = -torch.prod(torch.Tensor((action_dim, ))).to(device)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.temperature_lr)
        else:
            self.log_alpha = torch.log(torch.tensor(args.temperature, device=device, dtype=torch.float32))

        hard_update(self.critic, self.target_critic)

    def get_action(self, state, evaluation=True):
        with torch.no_grad():
            if evaluation:
                _, _, action = self.actor.sample(state)
            else:
                action, _, _ = self.actor.sample(state)
        return action.cpu().numpy()[0]

    def train_actor(self, states, train_alpha=True):
        self.actor_optimizer.zero_grad()
        actions, log_pis, _ = self.actor.sample(states)
        if self.args.quantile_critic:
            q_values_dist_A, q_values_dist_B = self.critic(states, actions)
            q_values_A = q_values_dist_A.mean(1).unsqueeze(1)
            q_values_B = q_values_dist_B.mean(1).unsqueeze(1)
        else:
            q_values_A, q_values_B = self.critic(states, actions)
        q_values = torch.min(q_values_A, q_values_B)

        actor_loss = (self.log_alpha.exp().detach() * log_pis - q_values).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        if train_alpha:
            self.alpha_optimizer.zero_grad()
            alpha_loss = -(self.log_alpha.exp() * (log_pis + self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        else:
            alpha_loss = torch.tensor(0.)

        return actor_loss.item(), alpha_loss.item()

    def train_critic(self, states, actions, rewards, next_states, dones):
        self.critic_optimizer.zero_grad()
        if self.args.quantile_critic:
            with torch.no_grad():
                next_actions, next_log_pis, _ = self.actor.sample(next_states)
                next_q_values_dist_A, next_q_values_dist_B = self.target_critic(next_states, next_actions)
                next_q_values_A = next_q_values_dist_A.mean(1).unsqueeze(1)
                next_q_values_B = next_q_values_dist_B.mean(1).unsqueeze(1)

                _, idx = torch.min(input=torch.cat([next_q_values_A, next_q_values_B], dim=1), dim=1, keepdim=True)
                mixed_dist = torch.stack([next_q_values_dist_A, next_q_values_dist_B], dim=1)
                idx = idx.repeat(1, self.n_supports)
                idx = idx.view(self.batch_size, 1, self.n_supports)
                next_dist = torch.gather(mixed_dist, dim=1, index=idx).view(next_q_values_dist_A.shape[0], next_q_values_dist_A.shape[1])
                target_q_values = rewards - self.log_alpha.exp() * next_log_pis + (1 - dones) * self.gamma * next_dist

            q_values_dist_A, q_values_dist_B = self.critic(states, actions)
            critic_loss_A = cal_critic_loss(q_values_dist_A, target_q_values, quantile_critic=True)
            critic_loss_B = cal_critic_loss(q_values_dist_B, target_q_values, quantile_critic=True)
            critic_loss = (critic_loss_A + critic_loss_B) / 2
        else:
            with torch.no_grad():
                next_actions, next_log_pis, _ = self.actor.sample(next_states)
                next_q_values_A, next_q_values_B = self.target_critic(next_states, next_actions)
                next_q_values = torch.min(next_q_values_A, next_q_values_B) - self.log_alpha.exp() * next_log_pis
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

            q_values_A, q_values_B = self.critic(states, actions)
            critic_loss_A = cal_critic_loss(q_values_A,  target_q_values, quantile_critic=False)
            critic_loss_B = cal_critic_loss(q_values_B, target_q_values, quantile_critic=False)
            critic_loss = (critic_loss_A + critic_loss_B) / 2

        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item()

    def train(self):
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        critic_loss1 = self.train_critic(states, actions, rewards, next_states, dones)

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        critic_loss2 = self.train_critic(states, actions, rewards, next_states, dones)

        critic_loss = (critic_loss1 + critic_loss2) / 2

        if self.args.automating_temperature is True:
            actor_loss, log_alpha_loss = self.train_actor(states, train_alpha=True)
        else:
            actor_loss, log_alpha_loss = self.train_actor(states, train_alpha=False)

        soft_update(self.critic, self.target_critic, self.tau)

        return critic_loss, actor_loss, log_alpha_loss

