import matplotlib.pyplot as plt
import numpy as np
import random

res = [3938, 229, 285, 55, 1, 274, 1221, 1, 737, 3640, 1, 1, 503, 1, 1, 1, 1, 3835, 1349, 3991, 1, 4857, 1, 1, 81, 1, 1525, 1, 1, 1, 1, 2179, 3332, 1, 1, 1, 1168, 1, 1, 1, 1, 1, 281, 1, 501, 1, 1, 34, 5906, 1, 1, 7, 1, 167, 1, 1, 850, 1, 262, 1162, 1, 244, 1, 1, 9779, 1, 1, 1, 1, 104, 1, 248, 845, 4549, 1, 1, 5315, 3503, 1, 1, 1, 1, 1997, 1455, 4356, 252, 1, 1, 758, 3643, 1, 1, 1, 2628, 1, 1, 1, 3530, 1, 1]


avg = sum(res) / len(res)
std = np.std(res)

print(avg, std)


