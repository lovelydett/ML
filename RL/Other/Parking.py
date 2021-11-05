import gym
import highway_env

import torch
import numpy as np

def examine_env():
    env = gym.make("parking-v0")
    print(env.action_space)
    print(env.observation_space)

    ob = env.reset()
    for k, v in ob.items():
        print(k, v)

    print(env.action_space.high)
    print(env.action_space.low)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.num_state = 2 * 6
        self.num_action = 2

        self.dense1 = torch.nn.Linear(in_features=self.num_state, out_features=32).cuda()
        self.dense2 = torch.nn.Linear(in_features=32, out_features=32).cuda()
        self.dense3 = torch.nn.Linear(in_features=32, out_features=16).cuda()
        self.dense4 = torch.nn.Linear(in_features=16, out_features=8).cuda()
        self.dense5 = torch.nn.Linear(in_features=8, out_features=self.num_action).cuda()

        self.dense1.weight.data.normal_(0, 0.5)
        self.dense2.weight.data.normal_(0, 0.5)
        self.dense3.weight.data.normal_(0, 0.5)
        self.dense4.weight.data.normal_(0, 0.5)
        self.dense5.weight.data.normal_(0, 0.5)

        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

    def forward(self, input):
        input = input.cuda()
        assert input.shape[1] == self.num_state

        x = self.dense1(input)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)
        x = self.relu(x)
        x = self.dense4(x)
        x = self.relu(x)
        x = self.dense5(x)

        for i in range(self.num_action):
            x[:, i] = self.tanh(x[:, i])

        return x

class Actor():
    def __init__(self):
        self.net = Net()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-5)

    def choose_action(self, state, e):
        action = self.net.forward(state)
        if np.random.random() > e:




if __name__ == "__main__":
    a = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    a[:, 2] *= -1
    print(a)