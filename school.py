import pandas as pd

data = pd.read_csv('gpascore.csv')
print(data)

# 데이터 전처리를 위해 빈 부분을 찾는다.
print(data.isnull().sum())
# data= data.fillna(100)  100 으로 데이터를 채운다.
data = data.dropna()  # 빈칸이 있는 행을 삭제한다.

print(data['gpa'])
print(data['gpa'].min())
print(data['gpa'].count())

y데이터 = data['admit'].values
print(y데이터)

x데이터 = []

for i, rows in data.iterrows():
    x데이터.append([(rows['gre']), rows['gpa'], rows['rank']])

print(x데이터)

import tensorflow as tf
import numpy as np

# 모델을 만든다.
model = tf.keras.models.Sequential([
    # 히든레이어 설정
    tf.keras.layers.Dense(64, activation='sigmoid'),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(256, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # sigmoid는 0~1 사이에 숫자를 내보낸다. 
    # 마지막 레이어는 한개의 결과만 나온다.
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(np.array(x데이터), np.array(y데이터), epochs=2000)
# 데이터를 먼저 넣고 그 다음에 결과를 넣는다. 

예측값 = model.predict([[760, 3.7, 2], [400, 2.8, 1]])
print(예측값)