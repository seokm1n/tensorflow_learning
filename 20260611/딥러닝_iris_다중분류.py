import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris  # sklearn iris 데이터셋 로드
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical  # type: ignore

iris = load_iris()

# iris['data'] ==> 4개의 특성 데이터 train_x
# iris['target'] ==> train_y
# iris_train = [['']]


# 3가지 붓꽃 클래스 분류

# 데이터 전처리
iris_train = iris["data"]
iris_target = iris["target"]

y_onehot = to_categorical(iris_target)
print(y_onehot)

# 데이터 분할
train_x, test_x, train_y, test_y = train_test_split(
    iris_train, y_onehot, random_state=42
)

# 스케일 조정
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_x)
test_scaled = scaler.transform(test_x)

# 다중분류 모델 설계
multi_model = Sequential()
multi_model.add(Dense(units=16, input_dim=4, activation="leaky_relu"))
multi_model.add(Dense(units=32, activation="leaky_relu"))
multi_model.add(Dense(units=16, activation="leaky_relu"))
multi_model.add(Dense(units=3, activation="softmax"))

multi_model.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

# 학습
multi_model.fit(train_scaled, train_y, batch_size=16, epochs=200, verbose=1)

print("Test acc : ", multi_model.evaluate(test_scaled, test_y)[1])

# 스케일과 모델 별도 저장
multi_model.save("iris_multi_clf.keras")
joblib.dump(scaler, "/home/sm/tf_env/20260611/iris_scaler.pkl")
