import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist  # type: ignore
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D  # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore

# (28,28) train : 60000개 , test : 10000개 데이터 로드
(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

# print(len(train_x), train_x.shape)  # 60000 , (60000, 28, 28)
# print(test_x[0].shape, len(test_x))  # (28,28) , 길이 10000

# 정답의 클래스 분류와 어떤 클래스가 몇개 있는지
# print("target label 체크 : ")
# print(np.unique(train_y, return_counts=True))
# print(np.unique(test_y, return_counts=True))
# 0 ==> 6000, 1 ==> 6000 ...

# print(train_y[0])
# plt.imshow(train_y[0],cmap='gray') # label 9인 train_y 이미지 체크
# plt.show()

# 이미지 데이터 정규화
train_scaled = train_x.reshape(-1, 28, 28, 1) / 255.0
# print(train_scaled.shape)

train_x, val_x, train_y, val_y = train_test_split(train_scaled, to_categorical(train_y), test_size=0.2, 
                                                            random_state=42)
# print(len(train_x))  # 48000
# print(len(val_x))  # 12000

model = Sequential()
# 합성곱층 추가
# filters= ==> 필터의 개수
# kernel_size=3 ==> 3*3 필터
# padding='same' ==> zero padding
# 첫번째 층은 항상 입력 데이터 생각
# input_shape ==> 입력 이미지의 shape
model.add(Conv2D(filters=32, kernel_size=3, activation="relu", padding="same", input_shape=(28, 28, 1)))
# 풀링층 추가
model.add(MaxPooling2D(2))  # 2*2 필터가 2스트라이드 이동하면서 최댓값 선택
# 합성곱층 추가
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"))
# 풀링층 추가
model.add(MaxPooling2D(2))
model.add(Flatten())
# FC layer
model.add(Dense(100, activation="relu"))
# 과대 적합 방지
model.add(Dropout(0.4))
model.add(Dense(40, activation="relu"))
# 출력층 설계 ==> 분류하고자 하는 클래스의 수만큼 뉴런 필요
# fashion_minst 데이터의 라벨(정답)이 10개
# 출력층 활성화함수 ==> 다중분류 'softmax'
model.add(Dense(10, activation="softmax"))
model.summary()

# 모델 컴파일
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 모델 학습
# 콜백 기능 추가해서 best 모델 저장, 조기종료
checkpoint_cb = ModelCheckpoint(filepath="./20260612/best_fashion.keras", monitor="val_loss", 
                                verbose=1, save_best_only=True,)

earlystopping_cb = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

history = model.fit(train_x, train_y, epochs=100, verbose=1, validation_data=(val_x, val_y), 
                    callbacks=[checkpoint_cb, earlystopping_cb])

val_loss = history.history['val_loss']
train_loss = history.history['loss']

plt.plot(val_loss, c='red')
plt.plot(train_loss, c='blue')

plt.show()