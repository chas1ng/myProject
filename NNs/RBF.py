import torch
import torch.nn as nn
import numpy as np


class RBF_Cell_torch(nn.Module):
    def __int__(self, hidden_nums, features):
        super(self, RBF_Cell_torch).__init__()
        self.n_samples = 0  # 训练集样本数量, 每次训练时重新修正
        self.n_features = features  # 训练集特征数量, input_size
        self.h = hidden_nums  # 隐含层神经元个数
        self.w = nn.Parameter(torch.randn(self.h + 1, 1), requires_grad=True)  # 线性权值
        self.c = nn.Parameter(torch.randn(self.h, self.n_features), requires_grad=True)  # 神经元中心点
        self.sigma = nn.Parameter(torch.randn(self.h, 1), requires_grad=True)  # 高斯核宽
        self.base = nn.Parameter(torch.randn(1), requires_grad=True)

    # 计算径向基距离函数,单个核（hidden_size）
    def guass(self, sigma, X, ci):
        return torch.exp(-torch.norm((X - ci), dim=1) ** 2 / (2 * sigma ** 2))

    # 将原数据高斯转化成新数据, 遍历每个核
    def change(self, sigma, X, c):
        newX = torch.zeros((self.n_samples, len(c)))
        for i in range(len(c)):
            newX[:, i] = self.guass(sigma[i], X, c[i])
        return newX

    # 初始化参数
    def init(self):
        sigma = np.random.random((self.h, 1))  # (h,1)
        c = np.random.random((self.h, self.n_features))  # (h,n)
        w = np.random.random((self.h + 1, 1))  # (h+1,1)
        return sigma, c, w

    # 给输出层的输入加一列截距项
    def addIntercept(self, X):
        return torch.cat((X, torch.ones((self.n_samples, 1))), dim=1)

    def forward(self, inputs):
        self.n_samples, _ = inputs.shape
        hi_output = self.change(self.sigma, inputs, self.c)
        yi_output = self.addIntercept(hi_output)
        yi_output = torch.mm(yi_output, self.w)
        return yi_output
