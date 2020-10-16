import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

np.random.seed(10)  # 指定亂數種子
# 載入資料集
df = pd.read_csv(r"C:\Users\w2bc8\Desktop\keras test\wdbc.csv")
target_mapping = {"M": 0,
                  "B": 1}
df["diagnosis"] = df["diagnosis"].map(target_mapping)
dataset = df.values

np.random.shuffle(dataset)  # 使用亂數打亂資料
# 分割成特徵資料和標籤資料
X = dataset[:,2:32].astype(float)
Y = to_categorical(dataset[:,1])
# 特徵標準化
X -= X.mean(axis=0)
X /= X.std(axis=0)
# 分割成訓練和測試資料集
X_train, Y_train = X[:500], Y[:500]     # 訓練資料前120筆
X_test, Y_test = X[500:], Y[500:]       # 測試資料後30筆
# 建立Keras的Sequential模型
model = Sequential()
model.add(Dense(10, input_shape=(30,), activation="relu"))
model.add(Dense(20, activation="relu"))
model.add(Dense(2, activation="sigmoid"))
model.summary()   # 顯示模型摘要資訊
# 編譯模型
model.compile(loss="binary_crossentropy", optimizer="adam",
              metrics=["accuracy"])

# 訓練模型
print("Training ...")
model.fit(X_train, Y_train, epochs=150, batch_size=10)
# 評估模型
print("\nTesting ...")
loss, accuracy = model.evaluate(X_test, Y_test)
print("準確度 = {:.2f}".format(accuracy))

# 儲存Keras模型
print("Saving Model: iris.h5 ...")
model.save(r"C:\Users\w2bc8\Desktop\keras test\model_.h5")
