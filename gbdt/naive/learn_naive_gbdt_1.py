# coding: utf-8
from sklearn.tree import DecisionTreeRegressor
import numpy as np

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        # 初始化 y_pred，将其设为样本均值
        y_pred = np.mean(y) * np.ones(len(y))
        for i in range(self.n_estimators):
            # 计算残差
            residual = y - y_pred
            # 使用 CART 决策树拟合残差
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residual)
            # 计算当前树的预测值
            tree_pred = tree.predict(X)
            # 根据学习率更新 y_pred
            y_pred += self.learning_rate * tree_pred
            # 将当前树加入到列表中
            self.trees.append(tree)

    def predict(self, X):
        # 初始化 y_pred，将其设为样本均值
        y_pred = np.mean(y) * np.ones(len(X))
        for tree in self.trees:
            # 树的预测结果累加
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred