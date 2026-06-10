import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

house = pd.read_csv("/home/sm/tf_env/20260610/BostonHousing.csv")

x = house.iloc[:,0:13]
y = house.iloc[:,13]

train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=42)#, test_size=0.3)

# 딥러닝 선형회귀 모델 설계
model = Sequential()

scaler = StandardScaler()

train_scaled = scaler.fit_transform(train_x)
test_scaled = scaler.transform(test_x)

# 모델 사용 준비과정 ==> 환경설정과정
model.add(Dense(units=30, input_dim=13, activation="leaky_relu"))
model.add(Dense(units=6, activation="leaky_relu"))
model.add(Dense(units=1, activation="linear"))

# 모델 summary()
model.summary()

# 모델 compile()
model.compile(loss="mse", optimizer="adam", metrics=["mse"])

# 모델 학습
model.fit(train_scaled, train_y, batch_size=10, epochs=200, verbose=1)

# 결과
pre = model.predict(test_scaled).flatten()

for i in range(10):
    print('실제가격 : {:.3f}, 예상가격 : {:.3f}'.format(test_y.iloc[i], pre[i]))

print("mse : ", model.evaluate(test_scaled, test_y)[1])
