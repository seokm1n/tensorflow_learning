import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

titanic_df = pd.read_csv("/home/sm/tf_env/20260611/titanic_passengers.csv")

# 딥러닝에서 타깃데이터 0과 1 그대로 사용
# titanic_df["Survived"] = titanic_df["Survived"].map({1: "survive", 0: "fail"})

titanic_df["gender"] = titanic_df["gender"].map({"female": 1, "male": 0})

titanic_df.dropna(subset="Age", inplace=True)  # Age 컬럼 기준 결측치 제거

onehot_pclass = pd.get_dummies(titanic_df["Pclass"], prefix="Class", dtype=int)
titanic_df = pd.concat([titanic_df, onehot_pclass], axis=1)

titanic_df_x = titanic_df[["gender", "Age", "Class_1", "Class_2"]]
print(titanic_df_x.head())
print("=" * 80)

titanic_df_y = titanic_df[["Survived"]]
print(titanic_df_y.head())
print("=" * 80)

from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(
    titanic_df_x, titanic_df_y, random_state=47
)

print(train_x[:5])
print("=" * 80)
print("타깃데이터 체크 : ", titanic_df_y)
print("=" * 80)
titanic_df.info()
print("=" * 80)

# 특성데이터의 스케일 변환(정규화) ==> 표준점수 정규화((각 특성 = 평균) / 표준편차)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_x)
test_scaled = scaler.transform(test_x)

print(train_scaled[:5])
print("=" * 80)

# 딥러닝 모델 설계

# 입력 특성데이터 4개
# batch_size=16
# epochs=200

from tensorflow.keras import optimizers  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore

model = Sequential()

model.add(Dense(units=8, input_dim=4, activation="leaky_relu"))
model.add(Dense(units=16, activation="leaky_relu"))
model.add(Dense(units=32, activation="leaky_relu"))
model.add(Dense(units=16, activation="leaky_relu"))
model.add(Dense(units=8, activation="leaky_relu"))
model.add(Dense(units=1, activation="sigmoid"))

# 모델 compile()
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# 모델 학습
model.fit(train_scaled, train_y, batch_size=16, epochs=500, verbose=1)

score = model.evaluate(test_scaled, test_y)
print('Test acc : ', score[1])

model.save('titanic_bestmodel.keras') # 모델 전체(네트워크 구조 및 가중치) 저장
