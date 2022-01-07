import random

import ddpg


class ExCacheElem(object):
    def __init__(self,
                 state0: ddpg.DDPGState,
                 action0: ddpg.DDPGAction,
                 reward: float,
                 state1: ddpg.DDPGState):
        self.state0 = state0
        self.state1 = state1
        self.reward = reward
        self.action0 = action0


class ExCache(object):
    def __init__(self,
                 cache_capacity: int,
                 batch_size: int):
        self.cache_capacity: int = cache_capacity
        self.batch_size: int = batch_size

        self.cache: list[ExCacheElem] = []

    def put(self, elem: ExCacheElem):
        if len(self.cache) == self.cache_capacity:
            self.cache.pop(0)
        self.cache.append(elem)

    def get_random_batch(self) -> list[ExCacheElem]:
        if len(self.cache) > self.batch_size:
            return random.sample(self.cache, self.batch_size)
        else:
            return []

