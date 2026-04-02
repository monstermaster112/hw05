# HW05: 简单CNN与LeNet-5实现

## 目录结构

```
hw05/
├── simpleCNN.py          # 任务一：极简CNN实现（基于原文，注明来源）
├── train_lenet.py        # 任务二：LeNet-5训练与评估脚本
├── lenet5_model.py       # LeNet-5模型定义
├── requirements.txt      # 依赖包列表
├── README.md             # 本文件
├── report.md             # 实验报告
└── debug_notes.md        # 调试记录
```

## 文件对应任务

- **simpleCNN.py**: 任务一，极简CNN模型，用于MNIST手写数字识别。保留原文结构，来源：自实现，参考PyTorch官方教程。
- **train_lenet.py**: 任务二，LeNet-5的训练与评估入口。
- **lenet5_model.py**: LeNet-5模型定义模块。
- **requirements.txt**: Python依赖包。
- **report.md**: 实验报告，包括摘要、结构说明、结果对比等。
- **debug_notes.md**: 调试记录。

## 环境要求

- Python 3.8+
- PyTorch 1.9+
- 推荐使用Conda环境

## 安装与运行

1. 克隆或下载项目到本地。

2. 创建Conda环境（可选）：
   ```
   conda create -n hw05 python=3.8
   conda activate hw05
   ```

3. 安装依赖：
   ```
   pip install -r requirements.txt
   ```

4. 数据自动下载：运行脚本时，TorchVision会自动下载MNIST数据集到`./data`目录。若网络问题，可手动下载：
   - 训练集：http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz 等（见TorchVision文档）。

## 一键训练/评估

### 极简CNN（任务一）
```
python simpleCNN.py
```
- 自动加载数据、训练5个epoch、测试、保存模型为`simple_cnn_mnist.pth`、生成可视化图。

### LeNet-5（任务二）
```
python train_lenet.py
```
- 训练LeNet-5模型、评估、保存为`lenet5_mnist.pth`。

## 注意事项

- 若对simpleCNN.py有较大改动，见report.md。
- 调试记录见debug_notes.md。