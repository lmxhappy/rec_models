# coding: utf-8
from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=0)
iris = load_iris()
# cross_val_score(clf, iris.data, iris.target, cv=10)
x = iris.data
one_feat = x[:, 1:2]
clf.fit(one_feat, iris.target)
print(one_feat.min(), one_feat.max())
print(clf.get_n_leaves())

# 0到5对应0；6、7对应1；8对应2
for i in range(0, 99):
    i /= 10
    input = np.array([i])
    input = np.reshape(input, [-1,1])
    # print(input, input.shape)
    # print(i, clf.predict(input)  )
    leaf_indices = clf.apply(input)
    print(i, leaf_indices)

# import matplotlib.pyplot as plt
# # plt.plot(one_feat, iris.target)
# plt.scatter(one_feat, iris.target)
# plt.show()



