import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 预处理
iris = load_iris()
train_data, test_data, train_target, test_target = train_test_split(iris.data, iris.target, test_size=0.2, random_state=1)

# 建模
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion="entropy")
clf.fit(train_data, train_target)
y_pred = clf.predict(test_data)

# 验证
from sklearn import metrics
print(metrics.accuracy_score(y_true=test_target, y_pred=y_pred))
# 混淆矩阵验证
print(metrics.confusion_matrix(y_true=test_target, y_pred=y_pred))

# 输出文件:决策树结构
with open("./scikit-learnDemo/tree.dot", "w") as fw:
    tree.export_graphviz(clf, out_file=fw)

