import torch
import tools.draw as dw
import NNs.RBF as RBF
import tools.simple_data as datas
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def use_rbf(X, Z):
    # net make
    net = RBF.RBF_Cell_torch(50, 1)
    Z = torch.from_numpy(Z.astype(np.float32))
    X = torch.from_numpy(X.astype(np.float32))

    criterion = nn.MSELoss()
    optimizerG = optim.Adam(net.parameters(), lr=0.1)

    for i in range(1000):
        Y = net(X)
        loss = criterion(Y, Z)
        optimizerG.zero_grad()
        loss.backward()
        optimizerG.step()
        print(i, '   ', loss.item())
    Y = net(X)
    dw.draw([Y, Z])


if __name__ == "__main__":
    inputs, targets = datas.example1()
    # shape

    use_rbf(inputs, targets)


