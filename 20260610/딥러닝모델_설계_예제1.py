import tensorflow as tf
from tensorflow.keras.layers import Dense  # type: ignore # 새로운 층 만들 때 사용
from tensorflow.keras.models import Sequential  # type: ignore

model = Sequential()  # 딥러닝 층을 추가할 수 있는 전체 틀 생성
# Dense()의 첫 파라미터는 해당 층의 뉴런 개수 지정
model.add(Dense(30, input_dim=4, activation="relu"))
model.add(Dense(1, activation="relu"))

# 모델이 잘 설계 되었는지 체크
model.summary()
