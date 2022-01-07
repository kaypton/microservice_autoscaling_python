import torch
import torch.nn as nn
import torch.optim as optim
import numpy

import ddpg
from ddpg import DDPGActor as actor
from ddpg import DDPGCritic as critic
from ddpg import DDPGExCache as exCache


class DDPGAgent(object):
    def __init__(self,
                 s_dim: int,
                 a_dim: int,
                 experience_cache_capacity: int,
                 batch_size: int,
                 tau: float,
                 gamma: float,
                 **kwargs):
        """

        :param s_dim: state dimension
        :param a_dim: action dimension
        :param experience_cache_capacity: experience replay cache
        :param batch_size: training batch size
        :param kwargs:
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.experience_cache = exCache.ExCache(experience_cache_capacity,
                                                batch_size)
        self.soft_update_tau = tau
        self.gamma = gamma

        self.batch_size = batch_size

        self.actor = actor.DDPGActor(s_dim, 256, a_dim)
        self.actor_target = actor.DDPGActor(s_dim, 256, a_dim)

        self.critic = critic.DDPGCritic(s_dim + a_dim, 256, a_dim)
        self.critic_target = critic.DDPGCritic(s_dim + a_dim, 256, a_dim)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.__getattribute__("actor_lr"))
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.__getattribute__("critic_lr"))

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.loss_function = nn.MSELoss()

    def act(self, s0) -> numpy.ndarray:
        s0 = torch.tensor(s0, dtype=torch.float).unsqueeze(0)
        a0 = self.actor(s0).squeeze(0).detach().numpy()
        return a0

    def put_ex(self, elem: exCache.ExCacheElem):
        self.experience_cache.put(elem)

    def train_critic(self,
                     s1_tensor: torch.Tensor,
                     s0_tensor: torch.Tensor,
                     a0_tensor: torch.Tensor,
                     r1_tensor: torch.Tensor):

        a1_tensor = self.actor_target(s1_tensor).detach()
        y_true = r1_tensor + self.gamma * self.critic_target(s1_tensor, a1_tensor).detach()

        y_pred = self.critic(s0_tensor, a0_tensor)

        loss = self.loss_function(y_pred, y_true)
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

    def train_actor(self,
                    s0_tensor: torch.Tensor):
        loss = -torch.mean(self.critic(s0_tensor, self.actor(s0_tensor)))
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

    def soft_update(self,
                    net_target: nn.Module,
                    net: nn.Module):
        for target_param, param in zip(net_target.parameters(), net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.soft_update_tau) + param.data * self.soft_update_tau)

    def train(self):
        samples: list[exCache.ExCacheElem] = self.experience_cache.get_random_batch()
        if len(samples) == 0:
            return

        s0: list[numpy.ndarray] = []
        a0: list[numpy.ndarray] = []
        r1: list[float] = []
        s1: list[numpy.ndarray] = []

        for sample in samples:
            s0.append(sample.state0.to_ndarray())
            a0.append(sample.action0.to_ndarray())
            r1.append(sample.reward)
            s1.append(sample.state1.to_ndarray())

        s0_tensor: torch.Tensor = torch.tensor(s0, dtype=torch.float)
        a0_tensor: torch.Tensor = torch.tensor(a0, dtype=torch.float)
        r1_tensor: torch.Tensor = torch.tensor(r1, dtype=torch.float).view(self.batch_size, -1)
        s1_tensor: torch.Tensor = torch.tensor(s1, dtype=torch.float)

        self.train_critic(s1_tensor, s0_tensor, a0_tensor, r1_tensor)
        self.train_actor(s0_tensor)
        self.soft_update(self.critic_target, self.critic)
        self.soft_update(self.actor_target, self.actor)



