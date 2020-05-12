#!/usr/bin/python3

import numpy as np
import random

# 生成随机数
print(random.randint(0, 9))

# 生成数组 起始 隔1
x = np.arange(1,10)
print(x)
# 生成数组 起始 总数，自动隔x
x = np.linspace(1,10,num=73)
print(x)