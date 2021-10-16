import numpy as np
import pandas as pd

df = pd.read_csv("SVM.csv",index_col=0)
df

'''
ID：ID months：使用月数
falls：直近1か月で落下させた回数
sex：利用者の性別（1=男性／2=女性）
age：利用者の年代
past_falls：過去の故障歴の有無
crack：ヒビの有無
width：機種サイズ
height：機種サイズ
thick：機種サイズ
cover：ケースにカバーがついているか
ring：ケースにリングがついているか
'''

X = df.drop(['past_falls','sex','age','ring'], axis=1).values
#print(X)
#exit()

# 目的変数
Y = df['past_falls'].values

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#学習データとテストデータに分割
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# データの標準化処理
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

'''
PCAを使用して、次元を削除します。
多次元データを2次元データに減らしています。
'''

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train_std2 = pca.fit_transform(X_train_std)
X_test_std2 = pca.fit_transform(X_test_std)

X_train_std.shape    #(27, 7)
X_train_std2.shape    #(27, 2)




########## モデルの構築

from sklearn.svm import SVC

#モデルの構築（線形SVM，RBFカーネル）
model1 = SVC(kernel='linear')
model2 = SVC(kernel='rbf')

model1.fit(X_train_std2, Y_train)
model2.fit(X_train_std2, Y_train)




########## 分類結果の図示

#from matplotlib import pyplot
import matplotlib.pyplot as plt

#print(Y_train)
#exit()

#pyplot.plot(X_train_std2[Y_train == 1,0], X_train_std2[Y_train == 1,1], 'ro', label='past_falls: 1')
#pyplot.plot(X_train_std2[Y_train == 0,0], X_train_std2[Y_train == 0,1], 'bo', label='past_falls: 0')

plt.plot(X_train_std2[Y_train == 1,0], X_train_std2[Y_train == 1,1], 'ro', label='past_falls: 1')
plt.plot(X_train_std2[Y_train == 0,0], X_train_std2[Y_train == 0,1], 'bo', label='past_falls: 0')

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.savefig('Fig_1.png')

#pyplot.show()
plt.show()




#分類結果を図示する
#import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

#install mlxtend if you have an error here.
#
#pip install mlxtend --upgrade
#pip install nltk --U
#pip install --upgrade mlxtend --trusted-host pypi.org --trusted-host files.pythonhosted.org

fig = plt.figure(figsize=(8,5))
plot_decision_regions(X_train_std2, Y_train, clf=model1)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.savefig('Fig_2.png')

plt.show()




#分類結果を図示する
#import matplotlib.pyplot as plt
#from mlxtend.plotting import plot_decision_regions

fig = plt.figure(figsize=(8,5))
plot_decision_regions(X_train_std2, Y_train, clf=model2)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.savefig('Fig_3.png')

plt.show()




########## モデルの精度を確認

from sklearn.metrics import accuracy_score

# 学習データに対する精度
pred_train = model2.predict(X_train_std2)
accuracy_train = accuracy_score(Y_train, pred_train)
print('学習データに対する正解率： %.2f' % accuracy_train)

# テストデータに対する精度
pred_test = model2.predict(X_test_std2)
accuracy_test = accuracy_score(Y_test, pred_test)
print('テストデータに対する正解率： %.2f' % accuracy_test)

#分類結果を図示する
#import matplotlib.pyplot as plt
#from mlxtend.plotting import plot_decision_regions



fig = plt.figure(figsize=(8,5))
plot_decision_regions(X_test_std2, Y_test, clf=model2)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.savefig('Fig_4.png')

plt.show()




# model_no --> 4
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
#混合行列で精度を確認する
mat = confusion_matrix(Y_test,pred_test)

#pandasで表の形に
class_names = ["0", "1"]
data = pd.DataFrame(mat,index=class_names,columns=class_names)
print("RBFカーネルSVM")
data
print(data)
print("F1score")
print(f1_score(Y_test, pred_test, average="macro"))








