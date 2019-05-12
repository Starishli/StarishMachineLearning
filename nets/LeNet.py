import torch as tc
import torch.nn as nn
import torch.nn.functional as func


A = 1.7159


class RawLeNet(nn.Module):
    def __init__(self):
        super(RawLeNet, self).__init__()

        self.model_name = "RawLeNet"

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = func.avg_pool2d(x, kernel_size=2, stride=2)
        x = func.sigmoid(x)
        x = self.conv2(x)
        x = func.avg_pool2d(x, kernel_size=2, stride=2)
        x = func.sigmoid(x)
        x = x.view(-1, 16 * 5 * 5)
        x = func.tanh(self.fc1(x)) * A
        x = func.tanh(self.fc2(x)) * A
        x = self.fc3(x)
        return x


class ImprovedLeNet_0(nn.Module):
    def __init__(self):
        super(ImprovedLeNet_0, self).__init__()

        self.model_name = "ImprovedLeNet_0"

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
            
    def forward(self, x):
        x = self.conv1(x)
        x = func.max_pool2d(x, kernel_size=2, stride=2)
        x = func.relu(x)
        x = self.conv2(x)
        x = func.max_pool2d(x, kernel_size=2, stride=2)
        x = func.relu(x)
        x = x.view(-1, 16 * 5 * 5)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 略为增加全连接层神经元数量
class ImprovedLeNet_1(nn.Module):

    def __init__(self):
        super(ImprovedLeNet_1, self).__init__()

        self.model_name = "ImprovedLeNet_1"

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = func.max_pool2d(x, kernel_size=2, stride=2)
        x = func.relu(x)
        x = self.conv2(x)
        x = func.max_pool2d(x, kernel_size=2, stride=2)
        x = func.relu(x)
        x = x.view(-1, 16 * 5 * 5)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 略微增加卷积层滤波器数量
class ImprovedLeNet_2(nn.Module):
    def __init__(self):
        super(ImprovedLeNet_2, self).__init__()

        self.model_name = "ImprovedLeNet_2"

        self.conv1 = nn.Conv2d(1, 12, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(12, 30, kernel_size=5)

        self.fc1 = nn.Linear(30 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = func.max_pool2d(x, kernel_size=2, stride=2)
        x = func.relu(x)
        x = self.conv2(x)
        x = func.max_pool2d(x, kernel_size=2, stride=2)
        x = func.relu(x)
        x = x.view(-1, 30 * 5 * 5)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 同时增加
class ImprovedLeNet_3(nn.Module):
    def __init__(self):
        super(ImprovedLeNet_3, self).__init__()

        self.model_name = "ImprovedLeNet_3"

        self.conv1 = nn.Conv2d(1, 12, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(12, 30, kernel_size=5)

        self.fc1 = nn.Linear(30 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = func.max_pool2d(x, kernel_size=2, stride=2)
        x = func.relu(x)
        x = self.conv2(x)
        x = func.max_pool2d(x, kernel_size=2, stride=2)
        x = func.relu(x)
        x = x.view(-1, 30 * 5 * 5)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 大幅增加全连接层
class ImprovedLeNet_4(nn.Module):
    def __init__(self):
        super(ImprovedLeNet_4, self).__init__()

        self.model_name = "ImprovedLeNet_4"

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        self.fc1 = nn.Linear(5 * 5 * 16, 200)
        self.fc2 = nn.Linear(200, 500)
        self.fc3 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = func.max_pool2d(x, kernel_size=2, stride=2)
        x = func.relu(x)
        x = self.conv2(x)
        x = func.max_pool2d(x, kernel_size=2, stride=2)
        x = func.relu(x)
        x = x.view(-1, 16 * 5 * 5)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 大幅增加卷积层
class ImprovedLeNet_5(nn.Module):
    def __init__(self):
        super(ImprovedLeNet_5, self).__init__()

        self.model_name = "ImprovedLeNet_5"

        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)

        self.fc1 = nn.Linear(5 * 5 * 50, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = func.max_pool2d(x, kernel_size=2, stride=2)
        x = func.relu(x)
        x = self.conv2(x)
        x = func.max_pool2d(x, kernel_size=2, stride=2)
        x = func.relu(x)
        x = x.view(-1, 50 * 5 * 5)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 大幅同时增加
class ImprovedLeNet_6(nn.Module):
    def __init__(self):
        super(ImprovedLeNet_6, self).__init__()

        self.model_name = "ImprovedLeNet_6"

        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)

        self.fc1 = nn.Linear(5 * 5 * 50, 200)
        self.fc2 = nn.Linear(200, 500)
        self.fc3 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = func.max_pool2d(x, kernel_size=2, stride=2)
        x = func.relu(x)
        x = self.conv2(x)
        x = func.max_pool2d(x, kernel_size=2, stride=2)
        x = func.relu(x)
        x = x.view(-1, 50 * 5 * 5)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 更大幅同时增加
class ImprovedLeNet_7(nn.Module):
    def __init__(self):
        super(ImprovedLeNet_7, self).__init__()

        self.model_name = "ImprovedLeNet_7"

        self.conv1 = nn.Conv2d(1, 50, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(50, 100, kernel_size=5)

        self.fc1 = nn.Linear(5 * 5 * 100, 500)
        self.fc2 = nn.Linear(500, 1000)
        self.fc3 = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = func.max_pool2d(x, kernel_size=2, stride=2)
        x = func.relu(x)
        x = self.conv2(x)
        x = func.max_pool2d(x, kernel_size=2, stride=2)
        x = func.relu(x)
        x = x.view(-1, 100 * 5 * 5)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x

