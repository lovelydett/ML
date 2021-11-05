'''
Author: tt
Date: 2021.11.4
Description: Simple Actor-Critic to play CartPole, TD to train. On-policy
'''

import torch
import numpy as np
import gym

class PiNet(torch.nn.Module):
    def __init__(self, num_states, num_actions):
        super(PiNet, self).__init__()

        # Feature extraction layers
        self.dense1 = torch.nn.Linear(in_features=num_states, out_features=32)
        self.dense2 = torch.nn.Linear(in_features=32, out_features=32)
        self.dense3 = torch.nn.Linear(in_features=32, out_features=32)
        self.dense4 = torch.nn.Linear(in_features=32, out_features=num_actions)

        self.dense1.weight.data.normal_(0, 0.5)
        self.dense2.weight.data.normal_(0, 0.5)
        self.dense3.weight.data.normal_(0, 0.5)
        self.dense4.weight.data.normal_(0, 0.5)

        self.dense1 = self.dense1.cuda()
        self.dense2 = self.dense2.cuda()
        self.dense3 = self.dense3.cuda()
        self.dense4 = self.dense4.cuda()

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
        x = self.dense4(x)
        x = self.softmax(x)

        return x

class VNet(torch.nn.Module):
    def __init__(self, num_states):
        super(VNet, self).__init__()

        self.dense1 = torch.nn.Linear(in_features=num_states, out_features=32)
        self.dense2 = torch.nn.Linear(in_features=32, out_features=32)
        self.dense3 = torch.nn.Linear(in_features=32, out_features=32)
        self.dense4 = torch.nn.Linear(in_features=32, out_features=16)
        self.dense5 = torch.nn.Linear(in_features=16, out_features=8)
        self.dense6 = torch.nn.Linear(in_features=8, out_features=1)

        self.dense1.weight.data.normal_(0, 0.5)
        self.dense2.weight.data.normal_(0, 0.5)
        self.dense3.weight.data.normal_(0, 0.5)
        self.dense4.weight.data.normal_(0, 0.5)
        self.dense5.weight.data.normal_(0, 0.5)
        self.dense6.weight.data.normal_(0, 0.5)

        self.dense1 = self.dense1.cuda()
        self.dense2 = self.dense2.cuda()
        self.dense3 = self.dense3.cuda()
        self.dense4 = self.dense4.cuda()
        self.dense5 = self.dense5.cuda()
        self.dense6 = self.dense6.cuda()

        self.relu = torch.nn.ReLU()

    def forward(self, input):
        input = input.cuda()
        x = self.dense1(input)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)
        x = self.relu(x)
        x = self.dense4(x)
        x = self.relu(x)
        x = self.dense5(x)
        x = self.relu(x)
        x = self.dense6(x)
        x = self.relu(x)

        return x

class Brain:
    def __init__(self, num_states, num_actions, lr=1e-5):
        self.num_actions = num_actions
        self.pi_net = PiNet(num_states=num_states, num_actions=num_actions)
        self.v_net = VNet(num_states=num_states)
        self.optimizer_pi = torch.optim.Adam(self.pi_net.parameters(), lr=lr)
        self.optimizer_v = torch.optim.Adam(self.v_net.parameters(), lr=lr)

    def choose_action(self, state, e):
        '''
        :param state: current state
        :param e: e-greedy param
        :return: return BOTH output of the net and the chosen index!
        '''
        # Whether e-greedy or not, we have to do forward once and get an output.
        output, state_value = self.pi_net.forward(torch.Tensor([state])), self.v_net.forward(torch.Tensor([state]))
        chosen = np.random.randint(0, self.num_actions)
        if np.random.uniform(0, 1) > e:
            chosen = torch.argmax(output).item()
        return output, chosen, state_value

    def learn(self, action, next_state, reward, value, done, gamma=0.9):
        next_vs = torch.Tensor([[0]]).cuda()
        if not done:
            next_vs = self.v_net.forward(torch.Tensor([next_state]))
        next_vs = next_vs.detach()

        td_error = reward + gamma * next_vs - value



        # The loss for Critic
        loss_v = td_error.square()

        # The loss for Actor
        loss_pi = -1 * td_error.detach() * action.log()

        self.optimizer_v.zero_grad()
        loss_v.backward()
        self.optimizer_v.step()

        self.optimizer_pi.zero_grad()
        loss_pi.backward()
        self.optimizer_pi.step()

        return loss_v, loss_pi

class Env:
    def __init__(self):
        self.env = gym.make("CartPole-v0")
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        self.agent = Brain(self.num_states, self.num_actions)

    def run(self, episodes=1000, max_steps=200):
        for ep in range(episodes):
            step = 0
            state = self.env.reset()
            e = 0.9
            while True:
                step += 1
                output, chosen, value = self.agent.choose_action(state, e / (ep + 1))
                next_state, _, done, _ = self.env.step(chosen)
                reward = 0
                if done:
                    reward = -1 + step / max_steps
                if step == max_steps:
                    reward = 1
                reward = torch.Tensor([[reward]]).cuda()
                loss1, loss2 = self.agent.learn(output[0][chosen:chosen + 1], next_state, reward, value, done)
                if done or step == max_steps:
                    print(f"Episode{ep}: {step} steps, loss1 = {loss1}, loss2 = {loss2}")
                    break


if __name__ == "__main__":
    game = Env()
    game.run()



