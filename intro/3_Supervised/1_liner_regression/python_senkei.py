import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline    # You only need this line if you run this Python script on Jupyter Notebook. On the contrary, if you run this script on Terminal of MacOS (Commpand Prompt of Windows), then you do not need this line.

df = pd.read_csv("senkei.csv", index_col=0)

'''
ID：商品ID (or 期間ID)
sales：売上個数
insta_post：Instagram投稿数
insta_good：Instagramいいね！数
flyer：チラシ配布枚数
event：イベントあり＝１、イベントなし＝０
new_item：新作発売日＝１
holiday：土日祝＝１
'''

#df.head()
print(df.head())
'''
    sales  insta_post  insta_good  flyer  event  new_item  holiday
ID
1      62           0          20      0      0         0        0
2      60           0          24      0      0         0        0
3     104           0          26      0      0         0        0
4     102           0          22      0      0         0        0
5     178           0          39      0      0         0        1
'''




#################### 1. Sigle Linear Regression 単回帰分析 ####################


# 説明変数insta_good
X = df.iloc[:, 2].values
X = X.reshape(-1,1)
 
# 目的変数sales
Y = df.iloc[:, 0].values

# sklearn.linear_model.LinearRegression クラスを読み込み
from sklearn import linear_model
clf = linear_model.LinearRegression()

# 予測モデルを作成
clf.fit(X, Y)
 
# 回帰係数
print(clf.coef_)    #[2.05852128]
 
# 切片
print(clf.intercept_)    #101.71515440413765
 
# 決定係数
print(clf.score(X, Y))    #0.5218980221448157




######### Output
#
#
##### Data file: results_1_df.csv
#
Y_pred = (clf.coef_) * X + (clf.intercept_)
#
print(type(X))    #<class 'numpy.ndarray'>
print(type(Y))    #<class 'numpy.ndarray'>
print(type(Y_pred))    #<class 'numpy.ndarray'>
#
df_X      = pd.DataFrame(data=X, columns=['X'])
df_Y      = pd.DataFrame(data=Y, columns=['Y'])
df_Y_pred = pd.DataFrame(data=Y_pred, columns=['Y_pred'])
#
#Indices (on the left hand side) are common for the following pandas data frame, such as,0, 1, 2, 3, 4, ...
print(df_X.head())
print(df_Y.head())
print(df_Y_pred.head())
#
#print(pd.merge(df_X, df_Y, left_index=True, right_index=True))
df_X_Y = pd.merge(df_X, df_Y, left_index=True, right_index=True)
#
results_1_df = pd.merge(df_X_Y, df_Y_pred, left_index=True, right_index=True)
print(results_1_df.head())
#
results_1_df.to_csv('results_1_df.csv', header=True, index=False)
#
#
##### Parameters: results_1_param.csv
# Y_pred = (coefficient a) * X + (intercept b)
with open('results_1_param.csv', mode='w') as f:
    f.write('coefficient, ' + str(clf.coef_[0]) + '\n')    # '\n' (LF) or '\r\n' (CR+LF) is a line feed code
    f.write('intercept, ' + str(clf.intercept_) + '\n')
    f.write('R2, ' + str(clf.score(X, Y)) + '\n')
print(f.closed)
#
#
##### Graph: results_1_df_X_Y_Y_pred.png
plt.figure(1, figsize=(8,8))
#
plt.scatter(df_X, df_Y, s=20, linewidths=2, marker="o", alpha=0.5, c="blue", edgecolors="blue", label='Y')
plt.scatter(df_X, df_Y_pred, s=20, linewidths=2, marker="o", alpha=0.5, c="red", edgecolors="red", label='Y_pred')
#
plt.title("Single Linear Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.legend()
#
#print(df_X['X'].max())
#print(df_X['X'].min())
#print(df_Y['Y'].max())
#print(df_Y['Y'].min())
#print(df_Y_pred['Y_pred'].max())
#print(df_Y_pred['Y_pred'].min())
X_pos = (df_X['X'].max() - df_X['X'].min())/2
Y_max = max(df_Y['Y'].max(), df_Y_pred['Y_pred'].max())
Y_min = min(df_Y['Y'].min(), df_Y_pred['Y_pred'].min())
Y_pos = (Y_max - Y_min)/2
#
plt.text(X_pos, (Y_min + Y_max * 0.10), 'Y_pred = ' + str(clf.coef_[0]) + ' * X + ' + str(clf.intercept_), fontsize=10, c='red')
#
plt.savefig('results_1_df_X_Y_Y_pred.png')
plt.show()
plt.close()
#exit()




#################### 2. Multiple Linear Regression 重回帰分析 ####################

########## A. Raw Data

# 説明変数
X = df.iloc[:, 1:7].values

# 目的変数
Y = df.iloc[:, 0].values

# 予測モデルを作成
clf.fit(X, Y)

# 偏回帰係数
df_except_sales = df.iloc[:, 1:7]
print(pd.DataFrame({"Name":df_except_sales.columns,
                    "Coefficients":np.abs(clf.coef_)}).sort_values(by='Coefficients'))
 
# 切片
print(clf.intercept_)


##### Parameters: results_2_A_raw_param.csv
results_2_A_raw_param = pd.DataFrame({"Name":df_except_sales.columns,
                    "Coefficients":np.abs(clf.coef_)}).sort_values(by='Coefficients')

print(results_2_A_raw_param.columns)
results_2_A_raw_param = results_2_A_raw_param.append({'Name': 'intercept', 'Coefficients': clf.intercept_}, ignore_index=True)

results_2_A_raw_param.to_csv('results_2_A_raw_param.csv', header=True, index=False)

#exit()


########## B. Standardized Data

# データフレームの各列を正規化
df2 = df.apply(lambda x: (x - np.mean(x)) / np.std(x))
#df2.head()
print(df2.head())

# 説明変数
X = df2.iloc[:, 1:7].values
print(df2.iloc[:, 1:7].columns.values)
X_col_names = df2.iloc[:, 1:7].columns.values

# 目的変数
Y = df2.iloc[:, 0].values
print(df2.iloc[:, 0].head())
Y_col_name = df2.iloc[:, 0].name
print(Y_col_name)    #sales



# 予測モデルを作成
clf.fit(X, Y)

# 偏回帰係数
df2_except_sales = df2.iloc[:, 1:7]
print(pd.DataFrame({"Name":df2_except_sales.columns,
                    "Coefficients":np.abs(clf.coef_)}).sort_values(by='Coefficients'))

# 切片
print(clf.intercept_)




######## Output
#
#
##### Data file: results_2_A_standardized_df.csv
#
print(clf.coef_)
print(X)
print(clf.intercept_)
#
#Y_pred = (clf.coef_) * X + (clf.intercept_)
print(np.multiply([[1, 2], [3, 4]], [10, 20]))
print(np.sum(np.multiply([[1, 2], [3, 4]], [10, 20]), axis=1))
#
#Y_pred = np.multiply(clf.coef_, X) + (clf.intercept_)
Y_pred = np.sum(np.multiply(clf.coef_, X), axis=1) + (clf.intercept_)
print(Y_pred)
#exit()
#
#print(type(X))    #<class 'numpy.ndarray'>
#print(type(Y))    #<class 'numpy.ndarray'>
#print(type(Y_pred))    #<class 'numpy.ndarray'>
#
#print(X)
#print(Y)
print(Y_pred)
#
#
print(X_col_names)
#df_X      = pd.DataFrame(data=X, columns=['X'])
#df_X      = pd.DataFrame(data=X, columns=[X_col_names])
df_X      = pd.DataFrame(data=X, columns=X_col_names)
#print(df_X)
#
#df_Y      = pd.DataFrame(data=Y, columns=['Y'])
df_Y      = pd.DataFrame(data=Y, columns=[Y_col_name])
#print(df_Y)
#
#df_Y_pred = pd.DataFrame(data=Y_pred, columns=['Y_pred'])
df_Y_pred = pd.DataFrame(data=Y_pred, columns=[str(Y_col_name) + '_pred'])
print(df_Y_pred)
#
#
#
#Indices (on the left hand side) are common for the following pandas data frame, such as,0, 1, 2, 3, 4, ...
print(df_X.head())
print(df_Y.head())
print(df_Y_pred.head())
#
print(len(df_X))
print(len(df_Y))
print(len(df_Y_pred))
#
#
#print(pd.merge(df_X, df_Y, left_index=True, right_index=True))
df_X_Y = pd.merge(df_X, df_Y, left_index=True, right_index=True)
#
results_2_B_standardized_df = pd.merge(df_X_Y, df_Y_pred, left_index=True, right_index=True)
print(results_2_B_standardized_df.head())
#
results_2_B_standardized_df.to_csv('results_2_B_standardized_df.csv', header=True, index=False)
#
#
##### Parameters: results_2_A_standardized_param.csv
results_2_B_standardized_param = pd.DataFrame({"Name":df_except_sales.columns,
                    "Coefficients":np.abs(clf.coef_)}).sort_values(by='Coefficients')
#
print(results_2_B_standardized_param.columns)
results_2_B_standardized_param = results_2_B_standardized_param.append({'Name': 'intercept', 'Coefficients': clf.intercept_}, ignore_index=True)
#
results_2_B_standardized_param.to_csv('results_2_B_standardized_param.csv', header=True, index=False)
#
#
##### Graph: results_2_B_df_X_Y_Y_pred.png

plt.figure(2, figsize=(8,8))
#
plt.scatter(df_X['flyer'], df_Y, s=20, linewidths=2, marker="o", alpha=0.5, c="blue", edgecolors="blue", label='Y')
plt.scatter(df_X['flyer'], df_Y_pred, s=20, linewidths=2, marker="o", alpha=0.5, c="red", edgecolors="red", label='Y_pred')
#
plt.title("Multiple Linear Regression")
plt.xlabel("X: flyer")
plt.ylabel("Y: sales")
plt.grid(True)
plt.legend()
#
#print(df_X['X'].max())
#print(df_X['X'].min())
#print(df_Y['Y'].max())
#print(df_Y['Y'].min())
#print(df_Y_pred['Y_pred'].max())
#print(df_Y_pred['Y_pred'].min())
#
#X_pos = (df_X['flyer'].max() - df_X['flyer'].min())/2
X_pos = 0
#
Y_max = max(df_Y['sales'].max(), df_Y_pred['sales_pred'].max())
Y_min = min(df_Y['sales'].min(), df_Y_pred['sales_pred'].min())
Y_pos = (Y_max - Y_min)/2
#
plt.text(X_pos, (Y_min + Y_max * 0.10), 'Y_pred = ' + str(clf.coef_[0]) + ' * X + ' + str(clf.intercept_), fontsize=10, c='red')
#
plt.savefig('results_2_B_df_X_Y_Y_pred.png')
plt.show()
plt.close()




#################### 3. Multiple Linear Regression (Supervised Learning) 重回帰分析 ####################


# 説明変数
X = df2.iloc[:, 1:7].values
X_col_names = df2.iloc[:, 1:7].columns.values

# 目的変数
Y = df2.iloc[:, 0].values

df_except_sales = df.iloc[:, 1:7]

#学習用データとテスト用データを7:3に分割する
from sklearn.model_selection import train_test_split
np.random.seed(0)    # fix the results of train_test_split by setting n = 0 of np.random.seed(n)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# 予測モデルを作成
model = linear_model.LinearRegression()
model.fit(X_train, Y_train)

# 偏回帰係数
df2_except_sales = df2.iloc[:, 1:7]
print(pd.DataFrame({"Name":df2_except_sales.columns,
                    "Coefficients":model.coef_}).sort_values(by='Coefficients') )
 
# 切片
print(model.intercept_)

pred_train = model.predict(X_train)
pred_test = model.predict(X_test)



######## Output
#
#
##### Data file: results_3_A_df_train.csv
#
print(model.coef_)
print(X_train)
print(model.intercept_)
#
#Y_pred = (clf.coef_) * X + (clf.intercept_)
#print(np.multiply([[1, 2], [3, 4]], [10, 20]))
#print(np.sum(np.multiply([[1, 2], [3, 4]], [10, 20]), axis=1))
#
#Y_pred = np.multiply(clf.coef_, X) + (clf.intercept_)
Y_train_pred = np.sum(np.multiply(model.coef_, X_train), axis=1) + (model.intercept_)
#print(Y_train_pred)
#exit()
#
#print(type(X))    #<class 'numpy.ndarray'>
#print(type(Y))    #<class 'numpy.ndarray'>
#print(type(Y_pred))    #<class 'numpy.ndarray'>
#
#print(X)
#print(Y)
print(Y_train_pred)
#
#
print(X_col_names)
#df_X      = pd.DataFrame(data=X, columns=['X'])
#df_X      = pd.DataFrame(data=X, columns=[X_col_names])
df_X_train      = pd.DataFrame(data=X_train, columns=X_col_names)
#print(df_X)
#
#df_Y      = pd.DataFrame(data=Y, columns=['Y'])
df_Y_train      = pd.DataFrame(data=Y_train, columns=[Y_col_name])
#print(df_Y)
#
#df_Y_pred = pd.DataFrame(data=Y_pred, columns=['Y_pred'])
df_Y_train_pred = pd.DataFrame(data=Y_train_pred, columns=[str(Y_col_name) + '_pred'])
print(df_Y_train_pred)
#
#
#
#Indices (on the left hand side) are common for the following pandas data frame, such as,0, 1, 2, 3, 4, ...
print(df_X_train.head())
print(df_Y_train.head())
print(df_Y_train_pred.head())
#
print(len(df_X_train))
print(len(df_Y_train))
print(len(df_Y_train_pred))
#
#
#print(pd.merge(df_X, df_Y, left_index=True, right_index=True))
df_X_Y_train = pd.merge(df_X_train, df_Y_train, left_index=True, right_index=True)
#
results_3_A_df_train = pd.merge(df_X_Y_train, df_Y_train_pred, left_index=True, right_index=True)
print(results_3_A_df_train.head())
#
results_3_A_df_train.to_csv('results_3_A_df_train.csv', header=True, index=False)
#
#
##### Parameters: results_3_A_param.csv
results_3_A_param = pd.DataFrame({"Name":df_except_sales.columns,
                    "Coefficients":np.abs(model.coef_)}).sort_values(by='Coefficients')
#
print(results_3_A_param.columns)
results_3_A_param = results_3_A_param.append({'Name': 'intercept', 'Coefficients': model.intercept_}, ignore_index=True)
#
results_3_A_param.to_csv('results_3_A_param.csv', header=True, index=False)
#
#
##### Graph: results_3_A_df_train.png

plt.figure(3, figsize=(8,8))
#
plt.scatter(df_X_train['flyer'], df_Y_train, s=20, linewidths=2, marker="o", alpha=0.5, c="blue", edgecolors="blue", label='Y')
plt.scatter(df_X_train['flyer'], df_Y_train_pred, s=20, linewidths=2, marker="o", alpha=0.5, c="red", edgecolors="red", label='Y_pred')
#
plt.title("Multiple Linear Regression (training data)")
plt.xlabel("X: flyer")
plt.ylabel("Y: sales")
plt.grid(True)
plt.legend()
#
#print(df_X['X'].max())
#print(df_X['X'].min())
#print(df_Y['Y'].max())
#print(df_Y['Y'].min())
#print(df_Y_pred['Y_pred'].max())
#print(df_Y_pred['Y_pred'].min())
#
#X_pos = (df_X['flyer'].max() - df_X['flyer'].min())/2
X_pos = 0
#
Y_max = max(df_Y_train['sales'].max(), df_Y_train_pred['sales_pred'].max())
Y_min = min(df_Y_train['sales'].min(), df_Y_train_pred['sales_pred'].min())
Y_pos = (Y_max - Y_min)/2
#
plt.text(X_pos, (Y_min + Y_max * 0.10), 'Y_pred = ' + str(model.coef_[0]) + ' * X + ' + str(model.intercept_), fontsize=10, c='red')
#
plt.savefig('results_3_A_df_train.png')
plt.show()
plt.close()




##### Data file: results_3_B_df_test.csv
#
#print(model.coef_)
#print(X_train)
#print(model.intercept_)
#
#Y_pred = (clf.coef_) * X + (clf.intercept_)
#print(np.multiply([[1, 2], [3, 4]], [10, 20]))
#print(np.sum(np.multiply([[1, 2], [3, 4]], [10, 20]), axis=1))
#
#Y_pred = np.multiply(clf.coef_, X) + (clf.intercept_)
#Y_train_pred = np.sum(np.multiply(model.coef_, X_train), axis=1) + (model.intercept_)
Y_test_pred = np.sum(np.multiply(model.coef_, X_test), axis=1) + (model.intercept_)
#print(Y_train_pred)
#exit()
#
#print(type(X))    #<class 'numpy.ndarray'>
#print(type(Y))    #<class 'numpy.ndarray'>
#print(type(Y_pred))    #<class 'numpy.ndarray'>
#
#print(X)
#print(Y)
print(Y_test_pred)
#
#
print(X_col_names)
#df_X      = pd.DataFrame(data=X, columns=['X'])
#df_X      = pd.DataFrame(data=X, columns=[X_col_names])
df_X_test      = pd.DataFrame(data=X_test, columns=X_col_names)
#print(df_X)
#
#df_Y      = pd.DataFrame(data=Y, columns=['Y'])
df_Y_test      = pd.DataFrame(data=Y_test, columns=[Y_col_name])
#print(df_Y)
#
#df_Y_pred = pd.DataFrame(data=Y_pred, columns=['Y_pred'])
df_Y_test_pred = pd.DataFrame(data=Y_test_pred, columns=[str(Y_col_name) + '_pred'])
print(df_Y_test_pred)
#
#
#
#Indices (on the left hand side) are common for the following pandas data frame, such as,0, 1, 2, 3, 4, ...
print(df_X_test.head())
print(df_Y_test.head())
print(df_Y_test_pred.head())
#
print(len(df_X_test))
print(len(df_Y_test))
print(len(df_Y_test_pred))
#
#
#print(pd.merge(df_X, df_Y, left_index=True, right_index=True))
df_X_Y_test = pd.merge(df_X_test, df_Y_test, left_index=True, right_index=True)
#
results_3_B_df_test = pd.merge(df_X_Y_test, df_Y_test_pred, left_index=True, right_index=True)
print(results_3_B_df_test.head())
#
results_3_B_df_test.to_csv('results_3_B_df_test.csv', header=True, index=False)




##### Graph: results_3_B_df_test.png

plt.figure(4, figsize=(8,8))
#
plt.scatter(df_X_test['flyer'], df_Y_test, s=20, linewidths=2, marker="o", alpha=0.5, c="blue", edgecolors="blue", label='Y')
plt.scatter(df_X_test['flyer'], df_Y_test_pred, s=20, linewidths=2, marker="o", alpha=0.5, c="red", edgecolors="red", label='Y_pred')
#
plt.title("Multiple Linear Regression (test data)")
plt.xlabel("X: flyer")
plt.ylabel("Y: sales")
plt.grid(True)
plt.legend()
#
#print(df_X['X'].max())
#print(df_X['X'].min())
#print(df_Y['Y'].max())
#print(df_Y['Y'].min())
#print(df_Y_pred['Y_pred'].max())
#print(df_Y_pred['Y_pred'].min())
#
#X_pos = (df_X['flyer'].max() - df_X['flyer'].min())/2
X_pos = 0
#
Y_max = max(df_Y_test['sales'].max(), df_Y_test_pred['sales_pred'].max())
Y_min = min(df_Y_test['sales'].min(), df_Y_test_pred['sales_pred'].min())
Y_pos = (Y_max - Y_min)/2
#
plt.text(X_pos, (Y_min + Y_max * 0.10), 'Y_pred = ' + str(clf.coef_[0]) + ' * X + ' + str(clf.intercept_), fontsize=10, c='red')
#
plt.savefig('results_3_A_df_train.png')
plt.show()
plt.close()














##### results_3_C.png

print('X_trainを使ったモデルの平均二乗誤差＝{:0.2f}'.format(np.mean((np.array(Y_train, dtype = int) - pred_train) ** 2)))
print('X_testを使ったモデルの平均二乗誤差＝{:0.2f}'.format(np.mean((np.array(Y_test, dtype = int) - pred_test) ** 2)))


plt.figure(5, figsize=(8,8))

# 学習用データの残差プロット
train = plt.scatter(pred_train,(pred_train-Y_train),c='b',alpha=0.5)

# テスト用データの残差プロット
test = plt.scatter(pred_test,(pred_test-Y_test),c='r',alpha=0.5)

# y=0の水平線
plt.hlines(y=0,xmin=-8,xmax=8)

plt.legend((train,test),('Training','Test'),loc='lower left')
plt.title('Residual Plots')

plt.savefig('results_3_C.png')
plt.show()

