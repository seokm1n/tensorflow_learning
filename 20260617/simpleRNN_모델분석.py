import numpy as np
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, SimpleRNN  # type: ignore

# 1. 데이터
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
y = np.array([4, 5, 6, 7])

x = x.reshape(4,3,1) # (3 timesteps, 1 입력 차원수) ==> 4개의 샘플

model = Sequential()
model.add(SimpleRNN(10, return_sequences=False, input_shape=(3,1))) # 3개 timesteps , 1 입력차원수
model.add(Dense(1)) # default : linear , 별도 활성화함수 없이 입력 뉴런과 가중치 계산결과가 그대로 출력
model.summary()