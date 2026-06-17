import numpy as np
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, SimpleRNN  # type: ignore

# 1. 데이터
x = np.array([[5, 6, 7], [1, 2, 3], [11, 12, 13], [6, 7, 8]])
y = np.array([8, 4, 14, 9])

x = x.reshape(4, 3, 1)  # (3 timesteps, 1 입력 차원수) ==> 4개의 샘플

rnn_model = Sequential()
rnn_model.add(SimpleRNN(10, return_sequences=False, input_shape=(3, 1)))  # 3개 timesteps , 1 입력차원수
rnn_model.add(Dense(1))  # default : linear , 별도 활성화함수 없이 입력 뉴런과 가중치 계산결과가 그대로 출력
rnn_model.summary()

rnn_model.compile(loss="mse", optimizer="adam", metrics=["mse"])
rnn_model.fit(x, y, epochs=1000, batch_size=1)

print(rnn_model.predict(x))  # 잘 예측 함

# 임의 데이터 예측
pre_input = np.array([6, 7, 8])
pre_input = pre_input.reshape((1, 3, 1))
pre_out = rnn_model.predict(pre_input)
print(pre_out)
