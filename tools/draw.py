import matplotlib.pyplot as plt
import numpy as np
import torch


def draw(inputs):
    for i in range(len(inputs)):

        lens = len(inputs[i])
        x1 = np.linspace(1, lens, lens)[:, np.newaxis]
        if torch.is_tensor(inputs[i]) is True:
            inputs[i] = inputs[i].detach().numpy()
        plt.plot(x1, inputs[i])

    plt.show()


def picture(inputs):
    plt.imshow(inputs)
    plt.show()

