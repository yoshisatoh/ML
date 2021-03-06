########## 決定木分析


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

#データの読み込み
cancer = pd.read_csv("decision.csv",index_col=0)
cancer

#pd.DataFrame(cancer.data, columns = cancer.feature_names)　
#変数の選定
Y = cancer['regular_customer'].values
X = cancer.drop(['regular_customer'], axis=1).values

from sklearn.model_selection import train_test_split

#学習用データとテスト用データに分割
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,  test_size = 0.3, random_state =0)

from sklearn.tree import DecisionTreeClassifier

#決定木分析
tree = DecisionTreeClassifier(max_depth=3, random_state=1)
tree.fit(X_train, Y_train)

print("学習用データの精度：{:.3f}".format(tree.score(X_train, Y_train)))
print("テスト用データの精度：{:.3f}".format(tree.score(X_test, Y_test)))


##### 決定木の可視化

from sklearn.tree import export_graphviz
import graphviz

#install graphviz if you have an error here.
#
#pip install graphviz --upgrade
#pip install graphviz --U
#pip install --upgrade graphviz --trusted-host pypi.org --trusted-host files.pythonhosted.org


'''
tree_data=export_graphviz(tree, out_file =None, class_names =["なる", "ならない"],
                 feature_names =cancer.columns[1:], impurity =False, filled =True)
'''
tree_data=export_graphviz(tree, out_file =None, class_names =["Yes", "No"],
                 feature_names =cancer.columns[1:], impurity =False, filled =True)
graphviz.Source(tree_data)

graph = graphviz.Source(tree_data)
graph
#graph.render('decision_tree')
print('If you try to draw a graph here, you might see the following error.')
print("graphviz.backend.ExecutableNotFound: failed to execute 'dot', make sure the Graphviz executables are on your systems' PATH")


# You can use sklearn instead of graphviz
from sklearn.tree import plot_tree
plt.figure(figsize=(16, 10))
plot_tree(tree, feature_names=cancer.columns[1:], class_names =["Yes", "No"], filled=True)
plt.savefig('Fig_1.png')
plt.show()



########## ランダムフォレスト

from sklearn.ensemble import RandomForestClassifier

#ランダムフォレスト
clf = RandomForestClassifier(random_state=0)
clf = clf.fit(X_train, Y_train)

from sklearn.metrics import accuracy_score

pred = clf.predict(X_test)

#正解率
accuracy_score(pred, Y_test)

from sklearn.metrics import confusion_matrix

#混合行列で精度を確認する
mat = confusion_matrix(Y_test,pred)

#class_names = ["なる", "ならない"]
class_names = ["Yes", "No"]

df = pd.DataFrame(mat,index=class_names,columns=class_names)
df

features = cancer.columns[1:]

#変数の重要度を読み込む
importances = clf.feature_importances_

#重要度を降順にする
indices = np.argsort(importances)

plt.figure(figsize=(10, 10))
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), features[indices])

plt.savefig('Fig_2.png')

plt.show()











