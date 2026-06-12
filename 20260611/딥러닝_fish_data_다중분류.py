import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical  # type: ignore

np.set_printoptions(precision=8, suppress=True)  # 과학적 표기 대신 소수점 이하 8자리까지 표현
np.set_printoptions(threshold=np.inf)  # 무한으로 출력

fishdf = pd.read_csv("/home/sm/tf_env/20260611/fish_data.csv")

fish_train = fishdf[["Weight", "Length", "Diagonal", "Height", "Width"]].to_numpy()
fish_target = fishdf["Species"].to_numpy()
# print(fish_target)
# print(fishdf.columns)

le = LabelEncoder()  # 라벨(문자열)을 수치형태로 변환
y_encoded = le.fit_transform(fish_target)  # 타깃이 수치 형태로 변환됨

print(le.classes_)  # 타깃데이터 클래스 목록

# categorical_crossentropy() ==> 정답을 원-핫 인코딩 상태로
y_onehot = to_categorical(y_encoded)


train_x, test_x, train_y, test_y = train_test_split(
    fish_train, y_onehot, random_state=42
)

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_x)
test_scaled = scaler.transform(test_x)

# scaler 저장
joblib.dump(scaler, "/home/sm/tf_env/20260611/fish_scaler.pkl")

multi_model = Sequential()
multi_model.add(Dense(units=10, input_dim=5, activation="leaky_relu"))
multi_model.add(Dense(units=7, activation="softmax"))
# mmulti_model.summary()

multi_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

multi_model.fit(train_scaled, train_y, batch_size=1, epochs=500, verbose=1)

print("Test acc : ", multi_model.evaluate(test_scaled, test_y)[1])

multi_model.save("fish_multi_clf.keras")
