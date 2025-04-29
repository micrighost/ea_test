# -*- coding: utf-8 -*-


import pandas as pd
 
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from sklearn import preprocessing
import numpy as np
"""数据归一化（最大最小方法）"""
from sklearn.preprocessing import MinMaxScaler


"""載入資料"""
# 模擬攝氏Tester地址
# df_prediction = pd.read_csv(r"C:\Users\D\AppData\Roaming\MetaQuotes\Tester\D0E8209F77C8CF37AD8BF550E51FF075\Agent-127.0.0.1-3000\MQL5\Files\data.csv",sep = ",", encoding='UTF-16') #  資料時間:20190601~20210601

# 實際測試Terminal地址
# df_prediction = pd.read_csv(r"C:\Users\D\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Files\data.csv",sep = ",", encoding='UTF-16') #  資料時間:20190601~20210601

# DNNAutomatedTradingProject資料夾excel地址
df_prediction = pd.read_excel("propfecy_dnn_建模_1\DNN_6m28-1d_m1.xlsx")




# 加上月份，星期，日
df_prediction["Date"] = pd.to_datetime(df_prediction["timecurrent"])
df_prediction["year"] = df_prediction["Date"].dt.year
df_prediction["month"] = df_prediction["Date"].dt.month
df_prediction["date"] = df_prediction["Date"].dt.day
df_prediction["day"] = df_prediction["Date"].dt.dayofweek+1



# 取出x
x_prediction = df_prediction.loc[:, 'high':'day'] 
# x_prediction = df_prediction.loc[:, '3bopen':'Date']  # simple


# 刪掉x不需要的
x_prediction = x_prediction.drop('3d', axis=1)
x_prediction = x_prediction.drop('5d', axis=1)
x_prediction = x_prediction.drop('10d', axis=1)
x_prediction = x_prediction.drop('Date', axis=1)


# 用pandas轉成numpy數組
x_prediction = x_prediction.values

# 取出y
y_prediction = df_prediction.loc[:, '3d']




#   ===========================================取得訓練模型的資料，使要預測的資料獲得相同的標準差

# 如果訓練集跟測試集的標準化不同，那訓練集的成效不可能等於測試集，所以下三行程式碼不可用，紀錄一下這個坑
# scaler_prediction = MinMaxScaler()
# scaler_prediction.fit(x_prediction)   #训练
# x_prediction = scaler_prediction.transform(x_prediction)     #此时输出的x_prediction就是数组了 



df = pd.read_excel("propfecy_dnn_建模_1\DNN_6m_m1.xlsx") #  資料時間:20190601~20210601
# 加上月份，星期，日
df["Date"] = pd.to_datetime(df["timecurrent"])
df["year"] = df["Date"].dt.year
df["month"] = df["Date"].dt.month
df["date"] = df["Date"].dt.day
df["day"] = df["Date"].dt.dayofweek+1

x = df.loc[:, 'high':'day'] 
# x = df.loc[:, '3bopen':'Date']  # simple




# 刪掉x不需要的
x = x.drop('3d', axis=1)
x = x.drop('5d', axis=1)
x = x.drop('10d', axis=1)
x = x.drop('Date', axis=1)

scaler = MinMaxScaler()
# 用訓練模型的資料來计算待标准化数据的均值和方差等参数。
scaler.fit(x)   #训练

x_prediction = scaler.transform(x_prediction)     #此时输出的x_prediction就是数组且標準化

np.set_printoptions(threshold=np.inf,suppress=True,precision= 4)
print("壓縮數據:")
print(x_prediction)
print("原始數據:")
print(scaler.inverse_transform(x_prediction))

#   ===========================================以上為取得訓練模型的資料，使要預測的資料獲得相同的標準差


# 傳入你想要載入的模型名稱
model_name = "DNN_model"

import tensorflow as tf
import pathlib
# 导入模型
model_path = pathlib.Path(model_name+'.h5') 
new_model = tf.keras.models.load_model(model_path)

# 用載入的模型預測
predics = new_model.predict(x_prediction)



# 顯示shape
print(predics.shape)
print(y_prediction.shape)

# 引數說明:
# infstr='inf'>全部顯示
# suppress 設定是否科學記數法顯示 （預設值：False）
# precision 設定浮點數的精度 （預設值：8）
np.set_printoptions(threshold=np.inf,suppress=True,precision= 4)

for i in range(len(df_prediction["timecurrent"])):

    # 時間
    print("時間:"+df_prediction["timecurrent"][i])

    # 預測
    print("預測:"+str(predics[i]))

    print(predics[i][0])
    print(predics[i][1])


    from pandas import Series,DataFrame
    if predics[i][0]>predics[i][1]:
        data = {'sell':['0']}
        df = DataFrame(data)

        # header=0 >>這個為標頭
        # 存檔，但不保留索引
        df.to_csv(r"C:\Users\D\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Files\data.csv",index=0,encoding='UTF-16')
        print("0")
    else:    
        data = {'buy':['1']}
        df = DataFrame(data)

        # 存檔，但不保留索引
        df.to_csv(r"C:\Users\D\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Files\data.csv",index=0,encoding='UTF-16')
        
        # 測試時使用
        # df.to_csv(r"C:\Users\D\AppData\Roaming\MetaQuotes\Tester\D0E8209F77C8CF37AD8BF550E51FF075\Agent-127.0.0.1-3000\MQL5\Files\data.csv",header=0,index=0)
        print("1")


import time
time.sleep(15)
  


    
    
