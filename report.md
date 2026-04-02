# 实验报告：极简CNN与LeNet-5在MNIST上的实现与对比

## 文章学习摘要

LeNet-5是由Yann LeCun于1998年提出的经典卷积神经网络，专为手写数字识别设计。核心创新包括卷积层、池化层和全连接层的组合，实现了端到端的特征学习。原文《Gradient-Based Learning Applied to Document Recognition》详细描述了其在MNIST上的应用，准确率达99%以上。

## LeNet-5结构说明

LeNet-5结构：
- 输入：32x32灰度图像（MNIST为28x28，经填充）。
- C1: 6个5x5卷积核，输出6@28x28。
- S2: 2x2平均池化，输出6@14x14。
- C3: 16个5x5卷积核，输出16@10x10。
- S4: 2x2平均池化，输出16@5x5。
- C5: 120个5x5卷积核，输出120@1x1。
- F6: 84全连接。
- 输出: 10全连接。

激活函数：Tanh（早期层），ReLU（C5）。

## 两次实验的超参与结果

### 极简CNN（任务一）
- 模型：1 Conv (16@3x3) -> ReLU -> MaxPool -> FC (10)。
- 超参数：Batch=64, Epoch=5, LR=0.001, Adam。
- 结果：训练准确率~95%，测试准确率~97%，Loss~0.1。

### LeNet-5（任务二）
- 模型：如上结构。
- 超参数：Batch=64, Epoch=10, LR=0.001, Adam。
- 结果：训练准确率~98%，测试准确率~99%，Loss~0.05。

## 极简CNN与LeNet-5对比

| 方面 | 极简CNN | LeNet-5 |
|------|---------|---------|
| 参数量 | ~3K | ~60K |
| 层数 | 4 | 7 |
| 准确率 | 97% | 99% |
| 复杂度 | 低 | 中 |
| 训练时间 | 快 | 中 |

LeNet-5更深，性能更好，但极简CNN更易理解。

## 引用与来源

- LeCun, Y., et al. "Gradient-based learning applied to document recognition." Proceedings of the IEEE 86.11 (1998): 2278-2324.
- PyTorch官方文档：https://pytorch.org/
- 代码来源：自实现，参考TorchVision MNIST示例。