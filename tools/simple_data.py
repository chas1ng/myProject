import numpy as np
import NNs.RBF as RBF


def example1():
    # 拟合Hermit多项式
    X = np.linspace(-5, 5, 500)[:, np.newaxis]
    target = np.multiply(1.1 * (1 - X + 2 * X ** 2), np.exp(-0.5 * X ** 2))
    net = RBF.RBF_Cell_torch(50, 1)
    Y = net(X)

