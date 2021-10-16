#from keras.models import Sequential
#from keras.layers import Activation, Dense
#from keras.optimizers import Adam

#use keras under tensorflow instead instead of keras itself
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

#import keras
import tensorflow.keras

import tensorflow as tf




########## 学習用ディレクトリの作成
'''
# Colab内にディレクトリを作成

#ラベルUSAのディレクトリを作成
DIR = "train/usa/"
# DIRの名称のディレクトリがなければ作成
if not os.path.exists(DIR):
    os.makedirs(DIR)

#ラベルItalyのディレクトリを作成
DIR = "train/Italy/"
# DIRの名称のディレクトリがなければ作成
if not os.path.exists(DIR):
    os.makedirs(DIR)

#ラベルAustraliaのディレクトリを作成
DIR = "train/Australia/"
# DIRの名称のディレクトリがなければ作成
if not os.path.exists(DIR):
    os.makedirs(DIR)
'''




########## 学習用のデータ作成

# 学習用のデータを作る.
image_list = []
label_list = []


# ./img/train 以下のディレクトリの画像を読み込む。
#for dir in os.listdir("/content/train/"):
for dir in os.listdir("train/"):
    
    #dataset_path = "/content/train/" + dir 
    dataset_path = "train/" + dir 
    label = -1

    # USAはラベル0
    if dir == "USA":
      label = 0
    # Italyはラベル1
    elif dir == "Italy":
      label = 1
    # Australiaはラベル2
    elif dir == "Australia":
      label = 2
    

    for file in os.listdir(dataset_path):
      # 配列label_listに正解ラベルを追加(USA:0 Italy:1 Australia:2)
      label_list.append(label)
      filepath = dataset_path + "/" + file
      print(filepath)
      image = Image.open(filepath)
      image = image.resize((500, 500), Image.BICUBIC)
      image = np.asarray(image)
      image_list.append(image / 255.)

# kerasに渡すためにnumpy配列に変換。
train_data = np.array(image_list)

train_label = np.array(label_list)

#%matplotlib inline

import matplotlib.pyplot as plt

plt.imshow(train_data[3])

train_data.shape
train_label.shape




########## モデルの作成

#train.data.shapeがinput_shape
#model = Sequential([
#model = tf.keras.Sequential([
model = tf.keras.models.Sequential([
                             tf.keras.layers.Conv2D(16, (3, 3),
                                                    input_shape=(500, 500, 3), activation="relu"),
                             tf.keras.layers.MaxPool2D((2, 2)),
                             tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
                             tf.keras.layers.MaxPool2D((2, 2)),
                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(512),
                             tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#モデルの学習
np.random.seed(1)
tf.random.set_seed(2)
model.fit(train_data, train_label, epochs=20)

#学習したモデルをファイルに保存
model.save("world_heritage.h5")

#保存したファイルの読み込み
model_loaded = tf.keras.models.load_model("world_heritage.h5")




##### テスト用ディレクトリの作成

# Colab内にディレクトリを作成
DIR = "test/Australia"
# DIRの名称のディレクトリがなければ作成
if not os.path.exists(DIR):
    os.makedirs(DIR)




########## テスト用データの作成

# テスト用のデータを作る。
image_list = []
label_list = []    #added to fix a bug

# ./data/test 以下のディレクトリの画像を読み込む。
#for dir in os.listdir("/content/test/"):
for dir in os.listdir("test/"):
    #
    #dataset_path = "/content/test/" + dir 
    dataset_path = "test/" + dir 
    label = -1

    # USAはラベル0
    if dir == "USA":
      label = 0
    
    # Italyはラベル1
    elif dir == "Italy":
      label = 1
    
    # Australiaはラベル2
    elif dir == "Australia":
      label = 2
    
    for file in os.listdir(dataset_path):
      # 配列label_listに正解ラベルを追加(USA:0 Italy:1 Australia:2)
      label_list.append(label)
      filepath = dataset_path + "/" + file
      print(filepath)
      image = Image.open(filepath)
      image = image.resize((500, 500), Image.BICUBIC)
      image = np.asarray(image)
      image_list.append(image / 255.)

# kerasに渡すためにnumpy配列に変換。
test_data = np.array(image_list)
test_label = np.array(label_list)

#正解率
pred = model_loaded.predict_classes(test_data)
#This function was removed in TensorFlow version 2.6.
#If you see an error here, then downgrade your tensorflow to 2.5.
#
#pip install tensorflow==2.5.0 --upgrade
#pip install tensorflow==2.5.0  --U
#pip install --upgrade tensorflow==2.5.0 --trusted-host pypi.org --trusted-host files.pythonhosted.org
#
#Ignore the following.
#pred_prob = model.predict(test_data)
#pred = np.round(pred_prob).astype(int)    # predicted classes (0, 1, or 2)
#
print(pred)
print(test_label)
#
(pred == test_label).sum() / len(test_label)
print((pred == test_label).sum() / len(test_label))




########## 実行結果

label_names = ["USA","Italy","Australia"]

correct_data = test_data[pred == test_label]
correct_label = pred[pred == test_label]
count_correct=len(correct_label)
print(count_correct)

#正解した画像を表示
flg, axes = plt.subplots(2, 3, figsize=(10, 5))

for i in range(count_correct):
  ax = axes[i // 3][i % 3]
  ax.set_title(label_names[correct_label[i]])
  ax.axis("off")
  ax.imshow(correct_data[i])

plt.imshow(correct_data[0])




########## prediction when using an arbitrary image

arb_image_list = []

arb_filepath = 'arb/Australia__Sydney_Opera_House.jpg'

arb_image = Image.open(arb_filepath)
arb_image = arb_image.resize((500, 500), Image.BICUBIC)
arb_image = np.asarray(arb_image)
arb_image_list.append(arb_image / 255.)

# kerasに渡すためにnumpy配列に変換。
arb_test_data = np.array(arb_image_list)

# prediction when using an arbitrary image
arb_pred = model_loaded.predict_classes(arb_test_data)

print(arb_pred)    #[2]    Australia - This prediction is a correct answer.