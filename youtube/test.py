# coding: utf-8


# data = [0.1, 0.2, 0.2, 0.2, 0.4, 0.4, 0.5, 0.8, 0.8, 0.8]  # [0.1, 0.3, 0.2, 0.1, 0.3]
# from scipy.stats import norm
#
# out = norm.cdf(data)
#
# print(out)

import numpy as np

# 定义数据
data = np.array([0.1, 0.21, 0.22, 0.23, 0.41, 0.42, 0.5, 0.81, 0.82, 0.83])

# 对数据排序
sorted_data = np.sort(data)

# 计算每个数据点的ECDF值
ecdf_values = np.arange(1, len(sorted_data)+1) / len(sorted_data)

# 将数据点和它们的ECDF值关联起来
ecdf = list(zip(sorted_data, ecdf_values))

print(ecdf)