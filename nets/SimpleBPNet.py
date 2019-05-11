import torch.nn as nn
import torch.nn.functional as func


class SimpleBPNet(nn.Module):
    def __init__(self):
        super(SimpleBPNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)

        self.fc1 = nn.Linear(4 * 4 * 50, 200)
        self.fc2 = nn.Linear(200, 500)
        self.fc3 = nn.Linear(500, 10)

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

