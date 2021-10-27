'''
Author: tt
Date: 2021.10.27
Description: Dueling-DDQN to play Cartpole game with experience replay, implemented with pytorch, considering fixed target net.
'''

import torch
import numpy as np
import gym
import random

from collections import namedtuple

# A namedtuple to represent a transition
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

class MemoryPool:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.index = 0

    def __len__(self):
        return len(self.memory)

    def push(self, state, action, next_state, reward):
        action = action.cuda()
        reward = reward.cuda()
        state = state.cuda()
        next_state = None if next_state is None else next_state.cuda()
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.index] = Transition(state, action, next_state, reward)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

class QNet(torch.nn.Module):
    def __init__(self, num_states, num_actions):
        super().__init__()
        self.dense1 = torch.nn.Linear(in_features=num_states, out_features=32).cuda()
        self.dense2 = torch.nn.Linear(in_features=32, out_features=32).cuda()
        self.dense3 = torch.nn.Linear(in_features=32, out_features=32).cuda()
        self.dense_V = torch.nn.Linear(in_features=32, out_features=1).cuda() # The value determined only by current state
        self.dense_A = torch.nn.Linear(in_features=32, out_features=num_actions).cuda() # The Advantage value

        self.dense1.weight.data.normal_(0, 0.5)
        self.dense2.weight.data.normal_(0, 0.5)
        self.dense3.weight.data.normal_(0, 0.5)
        self.dense_V.weight.data.normal_(0, 0.5)
        self.dense_A.weight.data.normal_(0, 0.5)

        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()

    def forward(self, input):
        input = input.cuda()
        x = self.dense1(input)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        # x = self.dense3(x)
        # x = self.relu(x)

        # No need to relu, cuz we allow negative values
        V = self.dense_V(x)
        A = self.dense_A(x)

        # Q(s, a) = V(s) + A(s, a)
        V = V.expand(-1, A.shape[1])
        Q = V + A
        Q -= torch.mean(A) # Advantage is defined as THE ADVANTAGE each action has towards the average of all!

        return Q

class DDQN:
    def __init__(self, num_states, num_actions, lr=1e-4):
        self.num_actions = num_actions
        self.memory_pool = MemoryPool(capacity=1000)

        # In DDQN, we have two nets: main and target to cut-off relationship
        self.main_net = QNet(num_states=num_states, num_actions=num_actions)
        self.target_net = QNet(num_states=num_states, num_actions=num_actions)

        self.optimizer = torch.optim.Adam(self.main_net.parameters(), lr=lr)
        self.loss_func = torch.nn.SmoothL1Loss() # SmoothL1Loss is HuberLoss

    def experience_replay(self, batch_size=20, gamma=0.9):
        '''
        Learn with current memory pool
        '''
        if len(self.memory_pool) <  batch_size:
            return

        transitions = self.memory_pool.sample(batch_size) # Get BATCH_SIZE * (state, ...)
        batch = Transition(*zip(*transitions)) # We have to make it (BATCH_SIZE * state) (BATCH_SIZE * action) ...
        states = torch.cat(batch.state) # From BATCH_SIZE * 1 * 4 to BATCH_SIZE * 4
        actions = torch.cat(batch.action).unsqueeze_(1) # From (BATCH_SIZE, ) TO (BATCH_SIZE, 1)
        rewards = torch.cat(batch.reward)
        none_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_action_values = self.main_net.forward(states).gather(1, actions)
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
        next_state_values = torch.zeros(batch_size).cuda()

        # Use main net to decide action for next state
        next_state_max_actions = self.main_net.forward(none_final_next_states).max(1)[1].unsqueeze(1)
        # Then use target net to evaluate those new actions in next state
        next_state_values[non_final_mask] = self.target_net.forward(none_final_next_states).gather(0, next_state_max_actions).squeeze(1).detach()

        expected_state_action_values = rewards.cuda() + gamma * next_state_values.cuda()

        # Calculate loss
        loss = self.loss_func(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def choose_action(self, state, episode):
        e = 0.9 * (1 / (episode + 1))
        if np.random.uniform(0, 1) > e:
            action = self.main_net.forward(state).max(1)[1]
        else:
            action = torch.IntTensor([np.random.randint(0, self.num_actions)])
        return action

    def update_target_net(self):
        self.target_net.load_state_dict(self.main_net.state_dict())


class Agent:
    '''
    This is the agent in environment, can be seemed as a player
    '''
    def __init__(self, num_states, num_actions):
        self.brain = DDQN(num_states=num_states, num_actions=num_actions) # This is his brain

    def learn(self):
        self.brain.experience_replay() # He learns from his memory

    def get_action(self, state, episode):
        return self.brain.choose_action(state, episode)

    def memorize(self, state, action, next_state, reward):
        self.brain.memory_pool.push(state, action, next_state, reward)

    def update_target_net(self):
        self.brain.update_target_net()

class Environment:
    '''
    This is the cartpole environment
    '''
    def __init__(self):
        self.env = gym.make("CartPole-v0")
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        self.agent = Agent(self.num_states, self.num_actions)

    def run(self, max_episode=1000, max_steps=200, ddqn_interval=4):
        for ep in range(max_episode):
            observation = self.env.reset() # Get initial observation
            state = observation # Use observation directly as state
            state = torch.FloatTensor([state]) # Make it (1, 4) rather then(4, )
            for step in range(max_steps):
                action = self.agent.get_action(state, episode=ep) # Get an action through DQN
                next_observation, _, done, _ = self.env.step(int(action.item())) # Execute this action
                if done:
                    reward = torch.Tensor([-1.0])
                    next_state = None
                else:
                    reward = torch.Tensor([0.0])
                    next_state = torch.FloatTensor([next_observation])

                self.agent.memorize(state, action, next_state, reward)
                self.agent.learn()
                state = next_state
                if done or step == max_steps:
                    print(f"Episode{ep + 1}: played {step} times, {'Win' if not done else 'Lose'}")
                    break
                if step % ddqn_interval == 0: # Every several steps, update target net to match main net
                    self.agent.update_target_net()


if __name__ == "__main__":
    env = Environment()
    env.run()





