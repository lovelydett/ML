# tt
# 2021.10.7
# DQN model for CartPole game

import torch
from torch import nn
import torch.nn.functional as F

class QNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.dense1 = nn.Linear(in_features=5, out_features=10).cuda() # input is state and action
        self.dense2 = nn.Linear(in_features=10, out_features=16).cuda()
        self.dense3 = nn.Linear(in_features=16, out_features=32).cuda()
        self.dense4 = nn.Linear(in_features=32, out_features=64).cuda()
        self.dense5 = nn.Linear(in_features=64, out_features=64).cuda()
        self.dense6 = nn.Linear(in_features=64, out_features=64).cuda()
        self.dense7 = nn.Linear(in_features=64, out_features=64).cuda()
        self.dense8 = nn.Linear(in_features=64, out_features=64).cuda()
        self.dense9 = nn.Linear(in_features=64, out_features=64).cuda()
        self.dense10 = nn.Linear(in_features=64, out_features=32).cuda()
        self.dense11 = nn.Linear(in_features=32, out_features=16).cuda()
        self.dense12 = nn.Linear(in_features=16, out_features=8).cuda()
        self.dense13 = nn.Linear(in_features=8, out_features=1).cuda() # output is Q value

        self.dense1.weight.data.normal_(0, 0.1)
        self.dense2.weight.data.normal_(0, 0.1)
        self.dense3.weight.data.normal_(0, 0.1)
        self.dense4.weight.data.normal_(0, 0.1)
        self.dense5.weight.data.normal_(0, 0.1)
        self.dense6.weight.data.normal_(0, 0.1)
        self.dense7.weight.data.normal_(0, 0.1)
        self.dense8.weight.data.normal_(0, 0.1)
        self.dense9.weight.data.normal_(0, 0.1)
        self.dense10.weight.data.normal_(0, 0.1)
        self.dense11.weight.data.normal_(0, 0.1)
        self.dense12.weight.data.normal_(0, 0.1)
        self.dense13.weight.data.normal_(0, 0.1)

    def forward(self, input: torch.Tensor):
        input = input.cuda()
        x = self.dense1(input)
        x = F.relu(x)
        x = self.dense2(x)
        x = F.relu(x)
        # x = self.dense3(x)
        # x = F.relu(x)
        # x = self.dense4(x)
        # x = F.relu(x)
        # x = self.dense5(x)
        # x = F.relu(x)
        # x = self.dense6(x)
        # x = F.relu(x)
        # x = self.dense7(x)
        # x = F.relu(x)
        # x = self.dense8(x)
        # x = F.relu(x)
        # x = self.dense9(x)
        # x = F.relu(x)
        # x = self.dense10(x)
        # x = F.relu(x)
        # x = self.dense11(x)
        # x = F.relu(x)
        x = self.dense12(x)
        x = F.relu(x)
        x = self.dense13(x)
        x = F.relu(x)
        return x

class DQN:
    def __init__(self, lr=1e-3):
        self.learn_net = QNet() # Used to learn from the calculated loss
        self.pred_net = QNet() # Used to predict next action

        self.optimizer = torch.optim.Adam(self.learn_net.parameters(), lr=lr)
        self.loss_func = torch.nn.MSELoss()

    def learn(self, pred_y, y):
        y = y.detach() # dont calculate his loss
        loss = self.loss_func(pred_y, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def store_memory(self):
        torch.save(self.learn_net.state_dict(), "./model.pth")
        self.pred_net.load_state_dict(torch.load("./model.pth"))

    def get_predict(self, input):
        return self.learn_net.forward(input.cuda())

    def get_next_value(self, input):
        return self.pred_net.forward(input.cuda())




