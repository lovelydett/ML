# tt
# 2021.7.30
# Pytorch neural network modules
# Implement LeNet

import torch as t
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

class Net(nn.Module):
    def __init__(self):
        # sons of nn.Module must implement his father's ctr!
        super(Net, self).__init__() # this equals to nn.Module.__init__(self)

        # Conv layers
        # 1 = input image channel, 6 = output channels (filters), 5 = 5*5 kernel sizes
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # FC layers
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # we'd like to do 10-classification

    # once forward implemented, backward auto implemented by auto grad
    def forward(self, input):
        # Describes the forward process
        x = F.relu(self.conv1(input))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size()[0], -1) # view = reshape
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1) # dim=1 means apply "softmax: sum equals 1" to the second dimension of x
        return x

def basic_test():
    net = Net()
    print(f"net:\n{net}")
    for name, param in net.named_parameters():
        print(f"{name}, {param.size()}")
    print("\n")

    # Forward process
    input = Variable(t.randn(1, 1, 32, 32)) # randomly generate an image
    out = net(input)
    print(f"out.size():{out.size()}")
    target = Variable(t.zeros(1, 10)) # mock a classify target with size (1, 10)
    target[0][3] = 1
    print(f"mock a classify target: {target}")
    loss_func = nn.MSELoss() # loss function
    loss = loss_func(out, target) # compute loss (essentially diff between output and target)
    print(f"loss:{loss}")

    # Here is the most IMPORTANT part!
    # loss is a Variable object which contains the overall computation graph information:
    # input -> Conv2d -> Relu -> ... -> out -> MSE_loss -> loss
    # So when loss.backward() is called, it computes ALL grads on the graph, saved in xxx.grad!
    loss.backward()

    # Back propagation to update params
    # Choose an optimizer and pass the params to be updated and learning rate
    optimizer = opt.SGD(net.parameters(), lr=0.001)
    optimizer.step() # do update!

    # Dont forget to clear all grads
    optimizer.zero_grad()



if __name__ == "__main__":
    basic_test()



