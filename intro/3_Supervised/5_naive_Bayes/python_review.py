#必要なライブラリのインポートを行う TF-IDFを用いたテキストの特徴量化を用いて教師あり学習を実行する
import seaborn as sns
import pandas as pd
import os as os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# Colab内にディレクトリを作成
DIR = "naive_baize/"
# DIRの名称のディレクトリがなければ作成
if not os.path.exists(DIR):
    os.makedirs(DIR)

#csvデータの読み込み
#df = pd.read_csv("/content/naive_baize/review.csv", encoding="cp932")
df = pd.read_csv("review.csv", encoding="cp932")

#元データにカラム名が無いため、カラム名を追加します
df.columns = ["text", "sports"]

df.to_csv('df.csv', sep=',', header=True, index=False)


#上記データフレームをトレーニングデータセットとテストデータセットに分割
X = df["text"]
Y = df["sports"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=None)

#文字データのベクトル化(数値化)
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

#学習の実行
model.fit(X_train, Y_train)

#評価
print('Train accuracy = %.3f' % model.score(X_train, Y_train))
print(' Test accuracy = %.3f' % model.score(X_test, Y_test))




########## prediction by using an arbitrary test

arb_df = pd.read_csv("arb.csv", sep=',', header=None, encoding="utf-8")

print(arb_df)

#元データにカラム名が無いため、カラム名を追加します
arb_df.columns = ["text", "sports"]

print(arb_df)
arb_df.to_csv('arb_df.csv', sep=',', header=True, index=False)

arb_X = arb_df["text"]

print(arb_X)

print(model.predict(arb_X))


with open('arb_X.txt', mode='w', encoding='utf-8') as f:
    f.write(str(model.predict(arb_X)))
#野球、という結果を返すので間違い    
