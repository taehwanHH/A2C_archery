import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np

class TDActorCritic(nn.Module):

    def __init__(self,
                 policy_net,
                 value_net,
                 beta=1.0,
                 batch_size=256,
                 gamma: float = 1.0,
                 lr: float = 0.0002):
        super(TDActorCritic, self).__init__()
        self.policy_net = policy_net
        self.value_net = value_net
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr

        # use shared optimizer
        total_param = list(policy_net.parameters()) + list(value_net.parameters())
        self.optimizer = torch.optim.Adam(params=total_param, lr=lr)

        self._eps = 1e-25
        self._mse = torch.nn.MSELoss()
        self.beta = beta

    def get_action(self, state):
        with torch.no_grad():
            logits = self.policy_net(state)
            dist = Categorical(logits=logits)
            a = dist.sample()  # sample action from softmax policy
        return a

    def update(self, state, action, reward, next_state, done):
        # compute targets
        with torch.no_grad():
            td_target = reward + self.gamma * self.value_net(next_state) * (1 - done)
            td_error = td_target - self.value_net(state)

        # compute log probabilities
        dist = Categorical(logits=self.policy_net(state))
        entropy = dist.entropy()
        prob = dist.probs.gather(1, action.long())

        # compute the values of current states
        v = self.value_net(state)

        loss = -torch.log(prob + self._eps) * td_error + self._mse(v, td_target) - self.beta*entropy
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
