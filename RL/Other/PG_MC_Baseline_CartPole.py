'''
Author: tt
Date: 2021.11.2
Description: Policy Gradient to play CartPole, Monte Carlo with V(s) as baseline to train. On-policy
'''

import torch
import numpy as np
import gym

class Net(torch.nn.Module):
    def __init__(self, num_states, num_actions):
        super(Net, self).__init__()

        # Feature extraction layers
        self.dense1 = torch.nn.Linear(in_features=num_states, out_features=32)
        self.dense2 = torch.nn.Linear(in_features=32, out_features=32)
        self.dense3 = torch.nn.Linear(in_features=32, out_features=32)

        # Policy output
        self.dense4 = torch.nn.Linear(in_features=32, out_features=num_actions)

        # State value output
        self.dense5 = torch.nn.Linear(in_features=32, out_features=16)
        self.dense6 = torch.nn.Linear(in_features=16, out_features=8)
        self.dense7 = torch.nn.Linear(in_features=8, out_features=1)

        self.dense1.weight.data.normal_(0, 0.5)
        self.dense2.weight.data.normal_(0, 0.5)
        self.dense3.weight.data.normal_(0, 0.5)
        self.dense4.weight.data.normal_(0, 0.5)
        self.dense5.weight.data.normal_(0, 0.5)
        self.dense6.weight.data.normal_(0, 0.5)
        self.dense7.weight.data.normal_(0, 0.5)

        self.dense1 = self.dense1.cuda()
        self.dense2 = self.dense2.cuda()
        self.dense3 = self.dense3.cuda()
        self.dense4 = self.dense4.cuda()
        self.dense5 = self.dense5.cuda()
        self.dense6 = self.dense6.cuda()
        self.dense7 = self.dense7.cuda()

        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()

    def forward(self, input):
        '''
        :param input:
        :return: The probability to take each action, and the state value
        '''
        input = input.cuda()
        x = self.dense1(input)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)
        x = self.relu(x)

        y1 = self.dense4(x)
        y1 = self.softmax(y1)

        y2 = self.dense5(x)
        y2 = self.relu(y2)
        y2 = self.dense6(y2)
        y2 = self.relu(y2)
        y2 = self.dense7(y2)
        # y2 = self.relu(y2)

        return y1, y2

class Brain:
    def __init__(self, num_states, num_actions, lr=1e-4):
        self.num_actions = num_actions
        self.net = Net(num_states=num_states, num_actions=num_actions)
        self.memory_action = []
        self.memory_value = []
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def choose_action(self, state, e):
        '''
        :param state: current state
        :param e: e-greedy param
        :return: return BOTH output of the net and the chosen index!
        '''

        # Whether e-greedy or not, we have to do forward once and get an output.
        output, state_value = self.net.forward(torch.Tensor([state]))
        chosen = np.random.randint(0, self.num_actions)
        if np.random.uniform(0, 1) > e:
            chosen = torch.argmax(output).item()
        return output, chosen, state_value

    def memorize(self, action, value):
        self.memory_action.append(action)
        self.memory_value.append(value)

    def learn(self, G=0, gamma=0.9):
        steps = len(self.memory_action)
        rewards = torch.Tensor([G * (gamma ** (steps - i - 1)) for i in range(steps)]).cuda().detach()
        actions = torch.cat(self.memory_action)
        values = torch.cat(self.memory_value)
        delta = rewards - values
        delta = delta.detach()

        # Train value part
        loss1 = -1 * delta * values
        loss2 = -1 * delta * actions.log()
        loss = loss1 + loss2
        self.optimizer.zero_grad()
        loss.backward(torch.ones_like(loss))
        self.optimizer.step()

        self.memory_action = []
        self.memory_value = []

        return -1 * loss.sum()

class Env:
    def __init__(self):
        self.env = gym.make("CartPole-v0")
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        self.agent = Brain(self.num_states, self.num_actions)

    def run(self, episodes=1000, max_steps=200):
        for ep in range(episodes):
            state = self.env.reset()
            step = 0
            G = 1
            while True:
                step += 1
                output, chosen, value = self.agent.choose_action(state, 0.9 / (ep + 1))
                self.agent.memorize(output[0][chosen : chosen + 1], value[0][:]) # [x : x + 1] is to keep last 1 in shape(a, b, c, ..., 1) here
                next_state, _, done, _ = self.env.step(chosen)
                if done:
                    G = -1 + step / max_steps
                    break
                if step == max_steps:
                    G = 1
                    break
                state = next_state

            loss = self.agent.learn(G)
            print(f"Episode{ep + 1}, played {step} steps, final loss = {loss}")



if __name__ == "__main__":
    game = Env()
    game.run()



