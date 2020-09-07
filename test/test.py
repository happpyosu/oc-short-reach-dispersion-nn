import sys

sys.path.append('../utils')
import tensorflow as tf
import numpy as np
import random
from utils.plotutils import PlotUtils

a = [-1, -1, 3, 3]
b = [-1, 3, 1, 1]

c = [a[i] == b[i] for i in range(len(a))]

print(c)
print(sum(c))

print(1 / 2)

