# ライブラリのインポート
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df_original = pd.read_csv("clst.csv",encoding = "shift-jis",index_col=0)
df_original.head()

df=df_original[['forest', 'highest_temp', 'lowest_temp', 'rainfall', 'latitude',
       'longitude', 'altitude', 'moisture']]

from scipy.cluster.hierarchy import linkage, dendrogram

df_hclust = linkage(df, method="ward")
plt.figure(figsize=(12,8))
dendrogram(df_hclust)
plt.show()

cust_array = np.array([df["forest"].tolist(),df["highest_temp"].tolist(),df["lowest_temp"].tolist(),
                      df["rainfall"].tolist(),df["latitude"].tolist(),df["longitude"].tolist(),
                      df["altitude"].tolist(),df["moisture"].tolist()],np.int32)

cust_array = cust_array.T

# クラスタ分析を実行 (クラスタ数=3)
pred = KMeans(n_clusters=3).fit_predict(cust_array)
pred

# クラスタ分析を実行 (クラスタ数=4)
pred2 = KMeans(n_clusters=4).fit_predict(cust_array)
pred2

df_original['cluster_id_3groups'] = pred
df_original['cluster_id_4groups'] = pred2
df_original

df_original.to_csv('after_clst.csv',encoding='cp932')
df_original.columns

from folium import Map, Marker, CustomIcon, LayerControl
from folium.plugins import HeatMap

#If you see an error here, then downgrade your tensorflow to 2.5.
#
#pip install folium --upgrade
#pip install folium  --U
#pip install --upgrade folium --trusted-host pypi.org --trusted-host files.pythonhosted.org

# いらすとやから各主食のイラストをアイコン用に拝借
okome = "https://4.bp.blogspot.com/-q86BWyQStFw/WWNAvW36HNI/AAAAAAABFWc/hzm92utRk6AJzd0z2sRYc_WYJ-n3jQKNQCLcBGAs/s800/kome_hakumai.png"
tomorokoshi = "https://4.bp.blogspot.com/-dicRIara6vg/V6iIdBskkiI/AAAAAAAA9CI/xIHaA-P0oPA4BeAdK99x3HNgd2seFyZrwCLcB/s800/vegetable_corn.png"
mugi = "https://1.bp.blogspot.com/-o7yRDDeXgi0/VGLLONLJqzI/AAAAAAAAovs/ztPMBRtnKQ8/s800/mugi.png"

# foliumのMapオブジェクトを作成。初期位置は、データに含まれる緯度経度を利用
m = Map(location=[df_original['latitude'].mean(), df['longitude'].mean()], zoom_start=2)

# Mapオブジェクトに、緯度経度ベースのヒートマップを追加。
# なお、各国のクラスタリングスコアによって色分けをしている(スコアが高いと色が濃くなる)
HeatMap(df_original[['latitude', 'longitude','cluster_id_3groups']].values.tolist(),name="ヒートマップ").add_to(m)
for row in df_original.itertuples():

  if row.staplefood == "コメ":
    # 一店舗ずつマーカーをMapオブジェクトに追加していく。
    Marker(location=(row.latitude, row.longitude),
                 # ポップアップに表示する項目をhtmlタグで設定
                 popup=row.country,
                 # CustomIconを使うことで、任意の画像をマーカーのアイコンに設定可能
                 icon=CustomIcon(okome,
                                 icon_size=(20, 20),
                                 popup_anchor=(0, 0)),
                 ).add_to(m)
  elif row.staplefood == "トウモロコシ":
    # 一店舗ずつマーカーをMapオブジェクトに追加していく。
    Marker(location=(row.latitude, row.longitude),
                 # ポップアップに表示する項目をhtmlタグで設定
                 popup=row.country,
                 # CustomIconを使うことで、任意の画像をマーカーのアイコンに設定可能
                 icon=CustomIcon(tomorokoshi,
                                 icon_size=(20, 20),
                                 popup_anchor=(0, 0)),
                 ).add_to(m)
  else :
    # 一店舗ずつマーカーをMapオブジェクトに追加していく。
    Marker(location=(row.latitude, row.longitude),
                 # ポップアップに表示する項目をhtmlタグで設定
                 popup=row.country,
                 # CustomIconを使うことで、任意の画像をマーカーのアイコンに設定可能
                 icon=CustomIcon(mugi,
                                 icon_size=(20, 20),
                                 popup_anchor=(0, 0)),
                 ).add_to(m)
                 
# 描画したレイヤーをコントロールするパネルを追加。作成されたファイルの右上に追加できる。
LayerControl().add_to(m)
# 作成したMapオブジェクトをhtmlとして保存
m.save("heatmap.html")







