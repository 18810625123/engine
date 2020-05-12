#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

# 画线 y = x**2
def y_x2(plt):
    step = 1
    x1 = -10
    for i in range(1, 20):
        x1 = x2
        x2 = x1 + step
        y1 = x1 ** 2
        y2 = x2 ** 2
        plt.plot([x1, x2], [y1, y2])
    plt.show()