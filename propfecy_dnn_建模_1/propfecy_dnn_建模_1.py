
# -*- coding: utf-8 -*-


import pandas as pd
 
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from sklearn import preprocessing


df = pd.read_excel("propfecy_dnn_建模_1\DNN_6m_m1.xlsx") #  資料時間:20190601~20210601


# 加上年、月、日、星期
df["Date"] = pd.to_datetime(df["timecurrent"])
df["year"] = df["Date"].dt.year
df["month"] = df["Date"].dt.month
df["date"] = df["Date"].dt.day
df["day"] = df["Date"].dt.dayofweek+1 



df

# 抓取全部的資料，且只要Open~day
# 若是要抓取特定資料(3筆)，那就:3
x = df.loc[:, 'high':'day'] 

# 刪掉x不需要的
x = x.drop('3d', axis=1)
x = x.drop('5d', axis=1)
x = x.drop('10d', axis=1)
x = x.drop('Date', axis=1)






print(x)


#用pandas轉成numpy數組
x = x.values

y = df.loc[:, '3d']



from imblearn.over_sampling import SMOTE
# 定义SMOTE模型，random_state相当于随机数种子的作用
# 平衡樣本數
smo = SMOTE(random_state=42)
x = x.astype('float64')
x, y = smo.fit_resample(x, y)


# One-Hot編碼(順便轉成np)
encoder = preprocessing.LabelEncoder()
y = encoder.fit_transform(y)
y = np_utils.to_categorical(y)



X_train = x
Y_train = y

import numpy as np
"""数据归一化（最大最小方法）"""
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)   #训练

# 歸一化之前
print(X_train[0])

X_train = scaler.transform(X_train)     #此时输出的X_train就是数组了 

# 歸一化之後
print(X_train[0])

from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

"""顺序模型：类似于搭积木一样一层、一层放上去"""
# 用Sequential建立模型
model = tf.keras.Sequential()


"""添加层:其实就是 wx+b"""
model.add(tf.keras.layers.Dense(units=300, activation='relu', input_dim=30))
model.add(tf.keras.layers.Dense(units=150, activation='relu'))
model.add(tf.keras.layers.Dense(units=75, activation='relu'))
model.add(tf.keras.layers.Dense(units=50, activation='relu'))
model.add(tf.keras.layers.Dense(units=25, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation='relu'))
model.add(tf.keras.layers.Dense(units=2, activation='softmax'))

#編譯模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# 顯示模型
# tf.keras.utils.plot_model(model,show_shapes=True)

#===================================================================

"""编译，配置"""
# model每層定義好後需要經過compile
# optimizer最佳化工具為adam
# 方法為keras.losses.mean_squared_error

model.compile(optimizer= 'adam',
               loss=['categorical_crossentropy'],
               metrics=['accuracy']
                  )

"""训练数据"""
# 早停
callback = EarlyStopping(monitor="val_accuracy", patience=10, verbose=1, mode="max")


history = model.fit(X_train, Y_train,validation_split=0.33,batch_size=200,epochs = 200) #, callbacks=[callback]

# 評估模型
loss, accuracy = model.evaluate(X_train, Y_train)
print("訓練資料集的準確度 = {:.2f}".format(accuracy))

# 顯示訓練和驗證損失圖表
import matplotlib.pyplot as plt

# 绘制训练 & 验证的准确率值
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()



# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# 傳入你想要建置的模型名稱
model_name = "DNN_model"


import pathlib
# 使用 pathlib.Path 类构造路径, 可以免受平台差异性的困扰
model_path = pathlib.Path(model_name+'.h5') 
model = model.save(model_path)