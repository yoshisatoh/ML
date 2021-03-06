import pandas as pd
import numpy as np

df = pd.read_csv("association.csv", index_col=0)
df.head()

# Customer CDとItemをキーに商品個数を集計する
w1 = df.groupby(['CustomerCD', 'Item'])['Number'].sum()

w1.head()

# 商品番号を列に移動 (unstack関数の利用)
w2 = w1.unstack().reset_index().fillna(0).set_index('CustomerCD')

w2.head()

# 集計結果が正の場合True、0の場合Falseとする
basket_df = w2.apply(lambda x: x>0)

basket_df.head()

# ライブラリの読み込み
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# アプリオリによる分析
freq_items1 = apriori(basket_df, min_support = 0.06, use_colnames = True)

# Supportが高い順に表示
freq_items1.sort_values('support', ascending = False).head()

# アソシエーションルールの抽出
a_rules1 = association_rules(freq_items1, metric = "lift", min_threshold = 1)

# リフト値が高い順にソート
a_rules1 = a_rules1.sort_values('lift', ascending = False).reset_index(drop=True)

a_rules1.head()


import networkx as nx
#install mlxtend if you have an error here.
#
#pip install networkx --upgrade
#pip install networkx --U
#pip install --upgrade networkx --trusted-host pypi.org --trusted-host files.pythonhosted.org


#抽出したリストからantecedentsとconsequentsのみをピックアップ
edges = a_rules1[['antecedents', 'consequents']].values
edges

import networkx as nx
import matplotlib.pyplot as plt

#抽出したリストからantecedentsとconsequentsのみをピックアップ
edges = a_rules1[['antecedents', 'consequents']].values

G = nx.from_edgelist(edges)
 
plt.figure(figsize=(10, 10))
 
pos = nx.spring_layout(G, k=0.7)
 
#PageRankの追加
pr = nx.pagerank(G)
 
nx.draw_networkx_edges(G, pos, edge_color='y')
 
#node_sizeにPageRankの値を組み込む
nx.draw_networkx_nodes(G, pos, node_color='r', alpha=0.5, node_size=[5000*v for v in pr.values()])
nx.draw_networkx_labels(G, pos, font_size=8)
 
plt.axis('off')
plt.show()













