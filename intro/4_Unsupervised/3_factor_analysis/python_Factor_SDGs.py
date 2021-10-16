# 数値計算ライブラリ
import numpy as np
import pandas as pd
# 可視化ライブラリ
import matplotlib.pyplot as plt
#%matplotlib inline

df = pd.read_csv("Factor_SDGs.csv",encoding = "shift-jis")
df.head()

df2 = df.iloc[:, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]

df=df2
df.head()

# データの標準化
# sklearnの標準化モジュールをインポート
from sklearn.preprocessing import StandardScaler

# データを変換する計算式を生成
sc = StandardScaler()
sc.fit(df)

# 実際にデータを変換
z = sc.transform(df)

# sklearnのPCA(主成分分析)クラスをインポート
from sklearn.decomposition import PCA

# 主成分分析のモデルを生成
pca = PCA() # インスタンスを生成･定義
pca.fit(z)  # 標準化得点データにもとづいてモデルを生成

# 寄与率の取得
evr = pca.explained_variance_ratio_
pd.DataFrame(evr, 
             index=["PC{}".format(x + 1) for x in range(len(df.columns))], 
             columns=["寄与率"])

#　図の枠組みを作る
fig = plt.figure(figsize=(12,5))

plt.plot(["{}".format(x + 1) for x in range(len(df.columns))], evr, 'ro-', linewidth=2, color="#57C4CC")

# 図のタイトル
plt.title('Scree plot')

# 図のx軸、y軸ラベル
plt.xlabel('Principle Component')
plt.ylabel('Eigenvalues')

# レジェンドを追加
leg = plt.legend(['Eigenvalues from PCA'], loc='best', borderpad=0.3, 
                 shadow=False,
                 markerscale=0.4)
leg.get_frame().set_alpha(0.4)
plt.show()




# sklearnのFactorAnalysis(因子分析)クラスをインポート
from sklearn.decomposition import FactorAnalysis as FA

# 因子数を指定
n_components=5

# 因子分析の実行
fa = FA(n_components, max_iter=5000) # モデルを定義
fitted = fa.fit_transform(z) # fitとtransformを一括処理

print(fitted)
print(fitted.shape)

# 因子の解釈
# 変数Factor_loading_matrixに格納
Factor_loading_matrix = fa.components_.T

# データフレームに変換
pd.DataFrame(Factor_loading_matrix, 
             columns=["第1因子", "第2因子", "第3因子", "第4因子", "第5因子"], 
             index=[df.columns])

fitted

# big five 理論の５要素：
factors = ['外向性', '調和性', '誠実性', '神経症的傾向', '経験への開放性']

#　第一因子：　外交的、競争、好奇心、努力家、楽天的
#　第二因子：　几帳面、陽気、物怖じしない、誰とでも仲良く、先延ばししない、我が道行かない　（調和性＋誠実性）
#　第三因子: 心配性
#　第四因子：　親切でない、先延ばししない、常に流される
#　第五因子：　好奇心がない、空想ふけたくない

df_sdgs = df.iloc[:,30:]
#df_sdgs_renamed = df_sdgs.rename(index={0:SDG1, 1:})
df_fa = pd.DataFrame(fitted, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
df_scatter_plot_matrix = pd.concat([df_fa, df_sdgs], axis=1)
df_scatter_plot_matrix

pd.plotting.scatter_matrix(df_scatter_plot_matrix, figsize=(30,30))

plt.show()
















