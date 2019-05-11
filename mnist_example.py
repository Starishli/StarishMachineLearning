import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as func
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import logging

from source import TRAIN_LOGGER


BATCH_SIZE = 128
NUM_EPOCHS = 10

# preprocessing
normalize = transforms.Normalize(mean=[.5], std=[.5])
transform = transforms.Compose([transforms.ToTensor(), normalize])

# download and load the data
train_dataset = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=transform, download=False)

# encapsulate them into dataloader form
train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)


class SimpleBPNet(nn.Module):
    def __init__(self):
        super(SimpleBPNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = func.max_pool2d(x, kernel_size=2, stride=2)
        x = func.relu(self.conv2(x))
        x = func.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 4 * 4 * 50)
        x = func.relu(self.fc1(x))
        x = self.fc2(x)
        x = func.log_softmax(x, dim=1)
        return x


model = SimpleBPNet()

momentum = 0.5
learning_rate = 0.01

criterion = func.nll_loss
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


logging.basicConfig(format='%(asctime)s %(message)s',
                    filename=TRAIN_LOGGER, level=logging.INFO)

device = torch.device('cpu')


def test(data_loader, model_, device_):
    accuracy = 0
    with torch.no_grad():
        for images_, labels_ in tqdm(data_loader):
            images_, labels_ = images_.to(device_), labels_.to(device_)
            output_ = model_(images_)
            pred_ = output_.argmax(dim=1, keepdim=True)
            accuracy += pred_.eq(labels_.view_as(pred_)).sum().item()

    accuracy = 100. * accuracy / len(data_loader.dataset)
    return accuracy


train_accuracy_list = []
test_accuracy_list = []

for epoch in range(NUM_EPOCHS):
    # train process
    model.train()
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

    model.eval()

    train_accuracy = test(train_loader, model, device)
    test_accuracy = test(test_loader, model, device)

    train_accuracy_list.append(train_accuracy)
    test_accuracy_list.append(test_accuracy)

    # evaluate

    logging.info('''\n
    ============================
    Epoch {}
    train accuracy {:.2f}%
    test accuracy {:.2f}%
    ============================'''.format(
        epoch + 1, train_accuracy, test_accuracy
    ))

logging.info('''\n
    ============================
    Average train accuracy {:.2f}% 
    Average test accuracy {:.2f}%
    ============================'''.format(
    np.mean(train_accuracy_list), np.mean(test_accuracy_list)
))

