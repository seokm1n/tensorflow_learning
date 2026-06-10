from tensorflow.keras import optimizers  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(0, 10, 10)
print(X)
print(X.shape, type(X.shape))  # (10,) <class 'tuple'>
print("=" * 80)

Y = X + np.random.randn(*X.shape)  # tuple을 unpack시킴
print(Y)
print("=" * 80)

# 딥러닝 선형회귀 모델 설계
# 입력데이터(x) 1개
# 입력층의 뉴런 1개
# 출력층의 활성화함수 : linear
# 손실함수 : MSE

linear_model = Sequential()  # 모델 설계 틀 준비
linear_model.add(Dense(units=1, input_dim=1, activation="linear", use_bias=False))
# use_bias=False ==> 편향 무시
linear_model.summary()

# 모델 사용 준비과정 ==> 환경설정과정
linear_model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])

# 학습 전의 가중치(w) 체크
weights = linear_model.layers[0].get_weights()
w = weights[0][0][0]
print("fit 전 가중치 체크 : ", w)

linear_model.fit(X, Y, batch_size=1, epochs=1000, verbose=1)

# 학습 완료 후의 가중치(w) 체크
weights = linear_model.layers[0].get_weights()
w = weights[0][0][0]
print("fit 후 가중치 체크 : ", w)

plt.plot(X, Y, label="data")
# 모델이 찾은 데이터를 선으로 표시
plt.plot(X, w * X, label="pred")
plt.legend()  # 라벨을 차트에 뿌림
plt.savefig("linear_model.jpeg")
