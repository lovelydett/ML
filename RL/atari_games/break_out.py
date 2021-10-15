'''
Author: tt
Date: 2021.10.15
Description: DQN to play BreakOut in Atari
'''

import torch
import numpy as np
import gym

# QNet for DQN
class QNet(torch.nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        pass
    def forward(self, input):
        pass

class DQN:
    def __init__(self):
        pass
    def evaluate(self, state, action):
        return -1
    def store_memory(self, state, action, reward):
        pass
    def train_with_memory(self):
        pass

# Observation space: (210, 160, 3)
# Action space: Discrete(4)
class Env:
    def __init__(self):
        self.game = gym.make("Breakout-v0")
        self.dqn = DQN()

    def start_play(self, EPISODES=100, e_greedy=0.8, gamma=0.8, memory_size=20, is_train=True):
        def choose_action(observation):
            pass
        def get_reward():
            return 0
        for ep in range(EPISODES):
            ep += 1
            is_done = False
            observation = self.game.reset() # Acquire initial observation
            while not is_done:
                pass
                # 1. Evaluate all possible actions and choose one: argmax Q(s, a)
                # 2. Get reward for this action: r
                # 3. Figure out next observation: pi(s, a) -> s'
                # 4. Get max expected value for next non-terminal state: max Q(s', a')
                # 5. Record memory: [s, a, r + gamma * max Q(s', a')]
                # 6. If step is enough, do train once with current memory


def check_env(game_name=""):
    from gym import envs
    env_specs = envs.registry.all()
    env_ids = [s.id for s in env_specs]
    print(f"All envs: {env_ids}")
    if len(game_name) == 0:
        return
    print(f"Contains {game_name}: {game_name in env_ids}")
    if game_name not in env_ids:
        return
    env = gym.make(game_name)
    print(f"Observation space: {env.observation_space}") # (210, 160, 3)
    print(f"Observation ub: {env.observation_space.high}")
    print(f"Observation lb: {env.observation_space.low}")
    print(f"Action space: {env.action_space}") # Discrete(4)
    print(f"Num of actions: {env.action_space.n}")
    env.close()

if __name__ == "__main__":
    # check_env("Breakout-v0")
    env = Env()
    env.start_play()



