import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from utils import weight_init


class Twin_Q_net(nn.Module):
    def __init__(self, args, state_dim, action_dim, hidden_dims, activation_fc, device):
        super(Twin_Q_net, self).__init__()
        self.device = device
        self.activation_fc = activation_fc

        self.input_layer_A = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.hidden_layers_A = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer_A = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers_A.append(hidden_layer_A)
        if args.quantile_critic:
            self.output_layer_A = nn.Linear(hidden_dims[-1], args.n_supports)
        else:
            self.output_layer_A = nn.Linear(hidden_dims[-1], 1)

        self.input_layer_B = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.hidden_layers_B = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer_B = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers_B.append(hidden_layer_B)

        if args.quantile_critic:
            self.output_layer_B = nn.Linear(hidden_dims[-1], args.n_supports)
        else:
            self.output_layer_B = nn.Linear(hidden_dims[-1], 1)

        self.apply(weight_init)

    def _format(self, state, action):
        s, a = state, action
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float32)
            s = s.unsqueeze(0)

        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, device=self.device, dtype=torch.float32)
            a = a.unsqueeze(0)

        return s, a

    def forward(self, state, action):
        state, action = self._format(state, action)
        input = torch.cat([state, action], dim=1)

        x_A = self.activation_fc(self.input_layer_A(input))
        for i, hidden_layer_A in enumerate(self.hidden_layers_A):
            x_A = self.activation_fc(hidden_layer_A(x_A))
        x_A = self.output_layer_A(x_A)

        x_B = self.activation_fc(self.input_layer_B(input))
        for i, hidden_layer_B in enumerate(self.hidden_layers_B):
            x_B = self.activation_fc(hidden_layer_B(x_B))
        x_B = self.output_layer_A(x_B)

        return x_A, x_B


class GaussianPolicy(nn.Module):
    def __init__(self, args, state_dim, action_dim, action_bound,
                 hidden_dims=(400, 300), activation_fc=F.relu, device='cuda'):
        super(GaussianPolicy, self).__init__()
        self.device = device

        self.log_std_min = args.log_std_bound[0]
        self.log_std_max = args.log_std_bound[1]

        self.activation_fc = activation_fc

        self.input_layer = nn.Linear(state_dim, hidden_dims[0])

        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)

        self.mean_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)

        self.action_rescale = torch.as_tensor((action_bound[1] - action_bound[0]) / 2., dtype=torch.float32)
        self.action_rescale_bias = torch.as_tensor((action_bound[1] + action_bound[0]) / 2., dtype=torch.float32)

        self.apply(weight_init)

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def forward(self, state):
        state = self._format(state)

        x = self.activation_fc(self.input_layer(state))
        for i, hidden_layer in enumerate(self.hidden_layers):
            x = self.activation_fc(hidden_layer(x))
        mean = self.mean_layer(x)

        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        distribution = Normal(mean, log_std.exp())

        unbounded_action = distribution.rsample()
        # [Paper: Appendix C] Enforcing Action Bounds: [a_min, a_max] -> [-1, 1]
        bounded_action = torch.tanh(unbounded_action)
        action = bounded_action * self.action_rescale + self.action_rescale_bias

        # We must recover ture log_prob from true distribution by 'The Change of Variable Formula'.
        log_prob = distribution.log_prob(unbounded_action) - torch.log(self.action_rescale *
                                                                       (1 - bounded_action.pow(2).clamp(0, 1)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        mean = torch.tanh(mean) * self.action_rescale + self.action_rescale_bias

        return action, log_prob, mean

class INV_Twin_Q_net(nn.Module):
    def __init__(self, args, inv_idx, reg_idx, action_dim, hidden_dims, activation_fc, device):
        super(INV_Twin_Q_net, self).__init__()
        self.device = device
        self.activation_fc = activation_fc

        self.inv_input_layer_A = nn.Linear(len(inv_idx) + action_dim, hidden_dims[0], bias=False)
        self.reg_input_layer_A = nn.Linear(len(reg_idx) + action_dim, hidden_dims[0])

        self.inv_idx = torch.LongTensor(inv_idx).to(device)
        self.reg_idx = torch.LongTensor(reg_idx).to(device)

        self.hidden_layers_A = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer_A = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers_A.append(hidden_layer_A)

        if args.quantile_critic:
            self.output_layer_A = nn.Linear(hidden_dims[-1], args.n_supports)
        else:
            self.output_layer_A = nn.Linear(hidden_dims[-1], 1)

        self.inv_input_layer_B = nn.Linear(len(inv_idx) + action_dim, hidden_dims[0], bias=False)
        self.reg_input_layer_B = nn.Linear(len(reg_idx) + action_dim, hidden_dims[0])

        self.hidden_layers_B = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer_B = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers_B.append(hidden_layer_B)

        if args.quantile_critic:
            self.output_layer_B = nn.Linear(hidden_dims[-1], args.n_supports)
        else:
            self.output_layer_B = nn.Linear(hidden_dims[-1], 1)

        self.apply(weight_init)

    def _format(self, state, action):
        s, a = state, action
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float32)
            s = s.unsqueeze(0)

        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, device=self.device, dtype=torch.float32)
            a = a.unsqueeze(0)

        return s, a

    def forward(self, state, action):
        state, action = self._format(state, action)

        batch_size = state.shape[0]
        inv_idx = self.inv_idx.repeat(1, batch_size)
        inv_idx = inv_idx.view(batch_size, -1)

        reg_idx = self.reg_idx.repeat(1, batch_size)
        reg_idx = reg_idx.view(batch_size, -1)

        inv_state = torch.gather(state, 1, inv_idx)
        reg_state = torch.gather(state, 1, reg_idx)

        inv_input = torch.cat([inv_state, action], dim=1)
        reg_input = torch.cat([reg_state, action], dim=1)

        inv_feature_A = self.inv_input_layer_A(inv_input)
        reg_feature_A = self.reg_input_layer_A(reg_input)
        x_A = self.activation_fc(torch.abs(inv_feature_A) + reg_feature_A)
        for i, hidden_layer_A in enumerate(self.hidden_layers_A):
            x_A = self.activation_fc(hidden_layer_A(x_A))
        x_A = self.output_layer_A(x_A)

        inv_feature_B = self.inv_input_layer_B(inv_input)
        reg_feature_B = self.reg_input_layer_B(reg_input)
        x_B = self.activation_fc(torch.abs(inv_feature_B) + reg_feature_B)
        for i, hidden_layer_B in enumerate(self.hidden_layers_B):
            x_B = self.activation_fc(hidden_layer_B(x_B))
        x_B = self.output_layer_A(x_B)

        return x_A, x_B


class EQI_GaussianPolicy(nn.Module):
    def __init__(self, args, eqi_idx, reg_idx, action_dim, action_bound,
                 hidden_dims=(400, 300), activation_fc=F.relu, device='cuda'):
        super(EQI_GaussianPolicy, self).__init__()
        self.device = device

        self.log_std_min = args.log_std_bound[0]
        self.log_std_max = args.log_std_bound[1]

        self.eqi_idx = torch.LongTensor(eqi_idx).to(device)
        self.reg_idx = torch.LongTensor(reg_idx).to(device)

        self.activation_fc = activation_fc

        self.eqi_input_layer = nn.Linear(len(eqi_idx), hidden_dims[0], bias=False)
        self.reg_input_layer = nn.Linear(len(reg_idx), hidden_dims[0])

        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)

        self.mean_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)

        self.action_rescale = torch.as_tensor((action_bound[1] - action_bound[0]) / 2., dtype=torch.float32)
        self.action_rescale_bias = torch.as_tensor((action_bound[1] + action_bound[0]) / 2., dtype=torch.float32)

        self.apply(weight_init)

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def forward(self, state):
        state = self._format(state)

        batch_size = state.shape[0]
        eqi_idx = self.eqi_idx.repeat(1, batch_size)
        eqi_idx = eqi_idx.view(batch_size, -1)

        reg_idx = self.reg_idx.repeat(1, batch_size)
        reg_idx = reg_idx.view(batch_size, -1)

        eqi_state = torch.gather(state, 1, eqi_idx)
        reg_state = torch.gather(state, 1, reg_idx)

        eqi_feature = self.eqi_input_layer(eqi_state)
        reg_feature = self.reg_input_layer(reg_state)

        source = torch.sign(torch.sum(eqi_feature, dim=1))
        multiplier = torch.where(source == 0., torch.ones_like(source), source)
        multiplier = multiplier.view(-1, 1)

        x = self.activation_fc(torch.abs(eqi_feature) + reg_feature)
        for i, hidden_layer in enumerate(self.hidden_layers):
            x = self.activation_fc(hidden_layer(x))
        mean = self.mean_layer(x)

        log_std = self.log_std_layer(x)

        mean *= multiplier
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        distribution = Normal(mean, log_std.exp())

        unbounded_action = distribution.rsample()
        # [Paper: Appendix C] Enforcing Action Bounds: [a_min, a_max] -> [-1, 1]
        bounded_action = torch.tanh(unbounded_action)
        action = bounded_action * self.action_rescale + self.action_rescale_bias

        # We must recover ture log_prob from true distribution by 'The Change of Variable Formula'.
        log_prob = distribution.log_prob(unbounded_action) - torch.log(self.action_rescale *
                                                                       (1 - bounded_action.pow(2).clamp(0, 1)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        mean = torch.tanh(mean) * self.action_rescale + self.action_rescale_bias

        return action, log_prob, mean