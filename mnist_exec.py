import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import logging

from source import TRAIN_LOGGER, CACHE_DIR
from source.helpers import cache_write, cache_load
from nets.LeNet import ImprovedLeNet_0, ImprovedLeNet_1, ImprovedLeNet_2, ImprovedLeNet_3, \
    ImprovedLeNet_4, ImprovedLeNet_5, ImprovedLeNet_6, ImprovedLeNet_7, RawLeNet


class MnistTraining(object):
    def __init__(self, nn_model):
        self.batch_size = 128
        self.num_epochs = 10

        self.momentum = 0.5
        self.learning_rate = 0.01

        self.nn_model = nn_model
        self.device = torch.device('cpu')

        self.train_loader = None
        self.test_loader = None

        self.criterion = None
        self.optimizer = None

        logging.basicConfig(format='%(asctime)s %(message)s',
                            filename=TRAIN_LOGGER, level=logging.INFO)

    def get_data(self):
        # 预处理
        normalize = transforms.Normalize(mean=[.5], std=[.5])
        transform = transforms.Compose([transforms.ToTensor(), normalize])

        # 获取训练集和测试集
        train_dataset = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=transform,
                                                   download=True)
        test_dataset = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=transform,
                                                  download=False)

        self.train_loader = data.DataLoader(train_dataset, batch_size=self.batch_size,
                                            shuffle=False, drop_last=True)
        self.test_loader = data.DataLoader(test_dataset, batch_size=self.batch_size,
                                           shuffle=False, drop_last=True)

    def optimizer_init(self):
        # 初始化优化器
        self.criterion = nn.CrossEntropyLoss(size_average=False)
        self.optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=1e-3, betas=(0.9, 0.99))

    def test(self, data_loader):
        accuracy = 0

        with torch.no_grad():
            for images_, labels_ in tqdm(data_loader):
                images_, labels_ = images_.to(self.device), labels_.to(self.device)
                output_ = self.nn_model(images_)
                pred_ = output_.argmax(dim=1, keepdim=True)
                accuracy += pred_.eq(labels_.view_as(pred_)).sum().item()

        accuracy = 100. * accuracy / len(data_loader.dataset)
        return accuracy

    def exec(self):
        self.get_data()
        self.optimizer_init()

        train_accuracy_list = []
        test_accuracy_list = []

        logging.info(self.nn_model.model_name)

        for epoch in range(self.num_epochs):
            # train process
            self.nn_model.train()
            for images, labels in tqdm(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                output = self.nn_model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()

            self.nn_model.eval()

            train_accuracy = self.test(self.train_loader)
            test_accuracy = self.test(self.test_loader)

            train_accuracy_list.append(train_accuracy)
            test_accuracy_list.append(test_accuracy)

            # evaluate

            logging.info('''\n
============================
Epoch {}
train accuracy {:.2f}%
test accuracy {:.2f}%
============================'''.format(epoch + 1, train_accuracy, test_accuracy))

        logging.info('''\n
============================
Average train accuracy {:.2f}%
Average test accuracy {:.2f}%
============================'''.format(np.mean(train_accuracy_list), np.mean(test_accuracy_list)))

        cache_write(os.path.join(CACHE_DIR, "{}_result.dat".format(self.nn_model.model_name)), (train_accuracy_list, test_accuracy_list))


if __name__ == "__main__":
    # nn_model_ = LeNet()
    #
    # m_t = MnistTraining(nn_model_)
    # m_t.exec()
    #
    # nn_model_ = ImprovedLeNet_1()
    #
    # m_t = MnistTraining(nn_model_)
    # m_t.exec()
    #
    # nn_model_ = ImprovedLeNet_2()
    #
    # m_t = MnistTraining(nn_model_)
    # m_t.exec()
    #
    # nn_model_ = ImprovedLeNet_3()
    #
    # m_t = MnistTraining(nn_model_)
    # m_t.exec()
    #
    # nn_model_ = ImprovedLeNet_4()
    #
    # m_t = MnistTraining(nn_model_)
    # m_t.exec()
    #
    # nn_model_ = ImprovedLeNet_5()
    #
    # m_t = MnistTraining(nn_model_)
    # m_t.exec()
    #
    # nn_model_ = ImprovedLeNet_6()
    #
    # m_t = MnistTraining(nn_model_)
    # m_t.exec()
    #
    # nn_model_ = ImprovedLeNet_7()
    #
    # m_t = MnistTraining(nn_model_)
    # m_t.exec()

    nn_model_ = RawLeNet()

    m_t = MnistTraining(nn_model_)
    m_t.exec()

    # nn_model_ = ImprovedLeNet_0()
    #
    # m_t = MnistTraining(nn_model_)
    # m_t.exec()
