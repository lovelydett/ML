# tt
# 2021.8.2
# Implement LeNet in pytorch for CIFAR-10

import torch as t
import torchvision as tv
import torchvision.transforms as tf
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

import timeit

class LeNet(nn.Module):
    def __init__(self):
        # sons of nn.Module must implement his father's ctr!
        super(LeNet, self).__init__() # this equals to nn.Module.__init__(self)

        # Conv layers
        # 3 = input image channel, 6 = output channels (filters), 5 = 5*5 kernel sizes
        self.conv1 = nn.Conv2d(3, 6, 5)
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

def load_data():
    show = tf.ToPILImage() # convert Tensor to PIL.Image for visualization

    # transform process for input images
    transform = tf.Compose([
        tf.ToTensor(), # convert to Tensor
        tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))], # (mean1, mean2 ... meann) and (stddiv1, stddiv2 ... stddivn) for n channels
    )

    # set up metadata for CIFAR10 data set
    train_set = tv.datasets.CIFAR10(
        root="/home/tt/Datasets/",
        train=True,
        download=False,
        transform=transform,
    )

    # load data for train set, data_loader is an iterable object which supports multi-threading and shuffle
    train_loader = t.utils.data.DataLoader(
        train_set,
        batch_size=4,
        shuffle=True,
        num_workers=2,
    )

    # set uo testing set metas
    test_set = tv.datasets.CIFAR10(
        root="/home/tt/Datasets/",
        train=False,
        download=False,
        transform=transform,
    )

    # loads data for testing set
    test_loader = t.utils.data.DataLoader(
        test_set,
        batch_size=4,
        shuffle=False,
        num_workers=2,
    )

    return train_loader, test_loader

def train(train_loader, net: LeNet):
    loss_func = nn.CrossEntropyLoss()
    optimizer = opt.SGD(net.parameters(), lr = 1e-3, momentum=1e-4)

    # train the model:
    start = timeit.default_timer()
    for epoch in range(5):
        running_loss = 0.0
        for i, data in enumerate(train_loader, start=0):
            # load next batch of inputs and move them into GPU
            inputs, labels = data
            if t.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            # clear grads for the whole graph for last time
            optimizer.zero_grad()

            # do FP and BP
            outputs = net(Variable(inputs)) # FP for one batch
            loss = loss_func(outputs, Variable(labels)) # compute loss for this batch
            loss.backward() # calculate the grads
            optimizer.step() # finish update params with their grads

            # print training datas
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f"round: {epoch+1},{i+1}, loss: {running_loss}, time elapsed: {(timeit.default_timer()-start)} secs")
                running_loss = 0.0
    print(f"FInish training in {(timeit.default_timer()-start)} secs")


def test(test_loader, net:LeNet):
    correct, total = 0, 0
    for data in test_loader:
        inputs, labels = data
        if t.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
        outputs = net(Variable(inputs))
        _, pred = t.max(outputs.data, 1) # find max on the 2nd dim(1) since the first dimension is batch_size
        total += labels.shape[0]
        correct += (pred == labels).sum()
    print(f"{total} images tested, acc = {correct/total}")

if __name__ == "__main__":
    train_loader, test_loader = load_data()
    net = LeNet()
    if t.cuda.is_available():
        net = net.cuda()
    train(train_loader, net)
    test(test_loader, net)
