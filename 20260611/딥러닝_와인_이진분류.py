import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import optimizers  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore

# 출력 옵션 제어
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)
pd.set_option("max_colwidth", 1000)

wine_df = pd.read_csv("/home/sm/tf_env/20260611/wine_dataset.csv")
print(wine_df)
wine_df.info()

wine_df["style"] = wine_df["style"].map({"red": 1, "white": 0})

wine_df_x = wine_df[
    [
        "fixed_acidity",
        "volatile_acidity",
        "citric_acid",
        "residual_sugar",
        "chlorides",
        "free_sulfur_dioxide",
        "total_sulfur_dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
        "quality",
    ]
]
print(wine_df_x.head())
print("=" * 80)

wine_df_y = wine_df[["style"]]
print(wine_df_y.head())
print("=" * 80)

train_x, test_x, train_y, test_y = train_test_split(
    wine_df_x, wine_df_y, random_state=42
)

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_x)
test_scaled = scaler.transform(test_x)

# scaler 저장
joblib.dump(scaler, "/home/sm/tf_env/20260611/wine_scaler.pkl")

# 딥러닝 모델 설계

model = Sequential()

model.add(Dense(units=32, input_dim=12, activation="leaky_relu"))
model.add(Dense(units=64, activation="leaky_relu"))
model.add(Dense(units=32, activation="leaky_relu"))
model.add(Dense(units=1, activation="sigmoid"))

# 모델 compile()
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# 모델 학습
model.fit(train_scaled, train_y, batch_size=64, epochs=200, verbose=1)

score = model.evaluate(test_scaled, test_y)
print("정확도:", score[1])

model.save("wine_best_model.keras")  # 모델 전체(네트워크 구조 및 가중치) 저장
