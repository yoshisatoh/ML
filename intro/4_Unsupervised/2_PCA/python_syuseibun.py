import numpy as np 
import pandas as pd 

# 図やグラフを図示するためのライブラリインポート
# 日本語表示用ライブラリのインポート
#!pip install japanize-matplotlib
import matplotlib.pyplot as plt

import japanize_matplotlib
#install mlxtend if you have an error here.
#
#pip install japanize_matplotlib --upgrade
#pip install japanize_matplotlib --U
#pip install --upgrade japanize_matplotlib --trusted-host pypi.org --trusted-host files.pythonhosted.org

#%matplotlib inline

import sklearn #機械学習のライブラリインポート
from sklearn.decomposition import PCA #主成分分析器

df_pca = pd.read_csv(("syuseibun.csv"),encoding = "shift-jis")

df_pca.head()

from pandas import plotting 
# plotting.scatter_matrix(df_pca.iloc[:, 1:], figsize=(8, 8), c=list(df.iloc[:, 0]), alpha=0.5)
from pandas import plotting 
plotting.scatter_matrix(df_pca.iloc[:, 1:], figsize=(8, 8), alpha=0.5)
plt.show()

# 行列の標準化
df_pca_stand = df_pca.iloc[:, 1:].apply(lambda x: (x-x.mean())/x.std(), axis=0)
df_pca_stand.head(3)

#主成分分析の実行
pca = PCA()
pca.fit(df_pca_stand)
# データを主成分空間に写像
feature = pca.transform(df_pca_stand)

# 主成分得点
pd.DataFrame(feature, columns=["PC{}".format(x + 1) for x in range(len(df_pca_stand.columns))]).head()

# 第一主成分と第二主成分でプロットする
plt.figure(figsize=(6, 6))
plt.scatter(feature[:, 0], feature[:, 1], alpha=0.8)
plt.grid()
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# 寄与率
pd.DataFrame(pca.explained_variance_ratio_, index=["PC{}".format(x + 1) for x in range(len(df_pca_stand.columns))])

# 累積寄与率を図示する
import matplotlib.ticker as ticker
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
plt.plot([0] + list( np.cumsum(pca.explained_variance_ratio_)), "-o")
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative contribution rate")
plt.grid()
plt.show()

# PCA の固有値
pd.DataFrame(pca.explained_variance_, index=["PC{}".format(x + 1) for x in range(len(df_pca_stand.columns))])

# PCA の固有ベクトル
pd.DataFrame(pca.components_, columns=df_pca.columns[1:], index=["PC{}".format(x + 1) for x in range(len(df_pca_stand.columns))])

# 第一主成分と第二主成分における観測変数の寄与度をプロット
plt.figure(figsize=(6, 6))
for x, y, name in zip(pca.components_[0], pca.components_[1], df_pca.columns[1:]):
    plt.text(x, y, name)
plt.scatter(pca.components_[0], pca.components_[1], alpha=0.8)
plt.grid()
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

import seaborn as sns
import matplotlib.patches as patches

# **************************
# 主成分得点プロット *******
# **************************
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.scatter(feature[:,0],feature[:,1])
ax.set_xlabel('Principal Component 1', size=14)
ax.set_ylabel('Principal Component 2', size=14)
ax.set_title('YouTube Tag PCA', size=16)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=14)

# **************************
# 固有ベクトルを図示 *******
# **************************

pc1 = pca.components_[0]
pc2 = pca.components_[1]
arrow_magnification = 2 # 矢印の倍率
feature_names = list(df_pca.columns[1:])

# ガイド円
patch_circle = patches.Circle(
    xy=(0, 0), radius=1 * arrow_magnification,
    fc='white',
    ec='darkred',
    fill=False)

ax.add_patch(patch_circle)

# 矢印と変数ラベル
for i in range(len(feature_names)):
  # 矢印
  ax.arrow(0, 0,
           pc1[i] * arrow_magnification,
           pc2[i] * arrow_magnification,
           head_width=0.1,
           head_length=0.1,
          color='darkred')
  # 変数ラベル
  ax.text(pc1[i] * arrow_magnification * 1.2,
          pc2[i] * arrow_magnification * 1.2,
          feature_names[i])
plt.axis('scaled')
ax.set_aspect('equal')

plt.show()














