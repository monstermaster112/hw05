import torch
import torch.nn as nn

class LeNet5(nn.Module):
    """
    LeNet-5模型，用于MNIST手写数字识别。
    结构：Conv1 -> Pool1 -> Conv2 -> Pool2 -> Conv3 -> FC1 -> FC2 -> Output
    """
    def __init__(self):
        super(LeNet5, self).__init__()
        # C1: 6@28x28
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        # S2: 6@14x14
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # C3: 16@10x10
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        # S4: 16@5x5
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # C5: 120@1x1
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1)
        # F6: 84
        self.fc1 = nn.Linear(120, 84)
        # Output: 10
        self.fc2 = nn.Linear(84, 10)
        # 激活函数
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.tanh(self.conv1(x))
        x = self.pool1(x)
        x = self.tanh(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = x.view(-1, 120)
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)
        return x