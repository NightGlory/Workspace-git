import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.datasets import load_digits

# `X` 包含长度为 $64$ 的数组，它们代表了简单的 `8x8` 的平面图像。
dataset = load_digits()
X = dataset['data']
y = dataset['target']
# plt.imshow(X[4].reshape(8,8))

# 决策树的深度变化提高分类准确率
score = []
for depth in [1,2,10] : 
    reg_ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth))
    scores_ada = cross_val_score(reg_ada, X, y, cv=6)
    score.append(scores_ada.mean())
    
print(score)

