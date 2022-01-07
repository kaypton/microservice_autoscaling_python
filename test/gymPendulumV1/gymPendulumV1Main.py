import gym
import numpy
import torch

from ddpg import DDPGAgent as agent
from ddpg import DDPGExCache as ex_cache
from ddpg import DDPGState
from ddpg import DDPGAction


class State(DDPGState):

    def __init__(self, cos: float, sin: float, thetadot: float):
        super(State, self).__init__()
        self.cos = cos
        self.sin = sin
        self.thetadot = thetadot

    def to_ndarray(self) -> numpy.ndarray:
        return numpy.array([self.cos, self.sin, self.thetadot], dtype=numpy.float)

    def to_torch_tensor(self) -> torch.Tensor:
        return torch.tensor([self.cos, self.sin, self.thetadot], dtype=torch.float)


class Action(DDPGAction):

    def __init__(self, joint_effort: float):
        super(Action, self).__init__()
        self.joint_effort = joint_effort

    def to_ndarray(self) -> numpy.ndarray:
        return numpy.array([self.joint_effort], dtype=numpy.float)

    def to_torch_tensor(self) -> torch.Tensor:
        return torch.tensor([self.joint_effort], dtype=torch.float)


def main():
    env = gym.make('Pendulum-v1')
    env.reset()
    env.render()

    params = {
        'actor_lr': 0.001,
        'critic_lr': 0.001,
    }

    ddpg_agent = agent.DDPGAgent(3, 1, 10000, 50, 0.02, 0.99, **params)

    for episode in range(10000000):
        s0: numpy.ndarray = env.reset()
        episode_reward = 0
        env.render()

        for step in range(500):
            env.render()
            a0 = ddpg_agent.act(s0)
            s1, r1, done, _ = env.step(a0)

            print("================================")
            print("episode %d step %d reward: %d" % (episode, step, r1))
            print("episode %d step %d action: " % (episode, step) + str(a0))
            print("================================")

            ex_elem: ex_cache.ExCacheElem= ex_cache.ExCacheElem(
                State(s0[0], s0[1], s0[2]),
                Action(a0[0]),
                r1.item(),
                State(s1[0], s1[1], s1[2])
            )

            ddpg_agent.put_ex(ex_elem)
            episode_reward += r1
            s0 = s1

            ddpg_agent.train()

        print(episode, ': ', episode_reward)

