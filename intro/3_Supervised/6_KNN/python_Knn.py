import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

#データの読み込み
anime = pd.read_csv("knn.csv",index_col=0)
anime

X = anime.drop(['Group'], axis=1).values   # 説明変数
Y = anime['Group'].values   # 目的変数

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, Y_train)

# 予測実行
y_pred = knn.predict(X_test)
y_pred

from sklearn import metrics

metrics.accuracy_score(Y_test, y_pred)

k_range = []
accuracy = []

for k in range(1, 50):
    knn = KNeighborsClassifier(n_neighbors=k) # インスタンス生成
    knn.fit(X_train, Y_train)                 # モデル作成実行
    y_pred = knn.predict(X_test)              # 予測実行
    accuracy.append(metrics.accuracy_score(Y_test, y_pred)) # 精度格納
    k_range.append(k)

plt.plot(k_range, accuracy)
plt.show()

test_df = pd.DataFrame(
    X_test,
   # columns = anime
)

test_df["Group"] = Y_test
test_df["pred_Group"] = y_pred

test_df.head()

test_df.plot(kind="scatter", x=0, y=1, c="Group", cmap="winter")
plt.title("true Group")
plt.show()

test_df.plot(kind="scatter", x=0, y=1, c="pred_Group", cmap="winter")
plt.title("prediction Group")
plt.show()




