import numpy as np

import tools.load_MovingMnist
import NNs.ConvLstm
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # data_shape is (20, 10000, 64, 64) -> (10000, 20, 64, 64)
    # 训练参数设置
    path = "..\\Datasets\\MovingMinst\\mnist_train.npy"
    sequence_length = 20
    image_height = 64
    image_width = 64

    input_sequence_length = 10
    batch_size = 10
    print_step = 50
    num_prediction_steps = 20

    input_dim = 1
    hidden_dim = 1
    kernel_size = (3, 3)
    num_layers = 10

    lrG = 0.01
    epochs = 4000
    # 读取文件

    mnistdata = tools.load_MovingMnist.MovingMNISTdataset(path)
    train = DataLoader(mnistdata, batch_size=batch_size)
    # 建立模型
    net = NNs.ConvLstm.ConvLSTM(input_dim, hidden_dim, kernel_size, num_layers)
    optimizerG = optim.Adam(net.parameters(), lr=lrG)
    criterionLoss = nn.MSELoss()

    for epoch in range(epochs):
        Loss_acc = 0
        # 加载数据
        for i, data in enumerate(train):
            inputs = data[:, 0:19, ...]
            targets = data[:, 19:20, ...]
            # torch.Size([10, 19, 1, 64, 64])
            # torch.Size([10, 1, 1, 64, 64])
            outputs = net(inputs)[-1:, ...].permute(1, 0, 2, 3, 4)
            loss = criterionLoss(outputs, targets)
            optimizerG.zero_grad()
            loss.backward()
            optimizerG.step()
            Loss_acc += loss.item()
            print("\r" + str(i) + '/8000', end="", flush=True)
        print(Loss_acc)
