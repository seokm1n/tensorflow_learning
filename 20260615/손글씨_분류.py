import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D  # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping  # type: ignore

np.set_printoptions(precision=8, suppress=True)  # 과학적 표기 대신 소수점 이하 8자리까지 표현
np.set_printoptions(threshold=np.inf)  # 무한으로 출력

mnist = datasets.load_digits()  # 손글씨 이미지
features = mnist["data"]
print(len(features[7]))
print(len(features))  # 1797
labels = mnist["target"]
print(len(labels))  # 1797
print(np.unique(labels, return_counts=True))

# print(features[7].reshape(8, 8))
# plt.imshow(features[7].reshape(8, 8), cmap="gray")
# plt.show()

print(features.shape)
# ==> (1797,8,8,1) ==> (batch_size, 이미지가로, 이미지세로, 채널)
features = features.reshape(-1, 8, 8, 1) / 255.0  # 사이즈변경 + 스케일정규화
print(features.shape)

# features와 labels를 train_x, val_x로 분할(분할 비율 = 0.2)
train_x, val_x, train_y, val_y = train_test_split(features, labels, test_size=0.2, random_state=42)
print(len(train_x))
print(len(val_x))
# 데이터 전처리 및 데이터 준비 완료

# 모델 준비 ==> 이미지를 분류하는 모델 설계(10개 클래스를 분류, 다중분류)
# 이미지 분류에 특화된 모델 --> CNN
mnist_model = Sequential()

# 2개의 Conv, 2개의 Pooling
mnist_model.add(Conv2D(filters=32, kernel_size=3, activation="leaky_relu", padding="same", input_shape=(8, 8, 1)))
mnist_model.add(MaxPooling2D(2))
mnist_model.add(Conv2D(64, kernel_size=3, activation="leaky_relu", padding="same"))
mnist_model.add(MaxPooling2D(2))

# Flatten, drop-out, FC layer층 추가
mnist_model.add(Flatten())
mnist_model.add(Dropout(0.4))
mnist_model.add(Dense(100, activation="leaky_relu"))

# 마지막 출력은 10개의 뉴런으로 설정
mnist_model.add(Dense(10, activation="softmax"))

# 모델 summary()
mnist_model.summary()

# 모델 compile()
mnist_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# 손실함수 ==> categorical_crossentropy
# target(정답) 정수형태 그대로 사용 ==> sparse_categorical_crossentropy

# 모델설계 이후 학습 (fit)
# val_x, val_y 전달, val_loss 모니터링으로 조기 종료 콜백 추가
checkpoint_cb = ModelCheckpoint(filepath="./20260615/best_mnist.keras", monitor="val_loss", 
                                verbose=1, save_best_only=True,)

earlystopping_cb = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

history = mnist_model.fit(train_x, train_y, batch_size=4, epochs=100, verbose=1, 
                          validation_data=(val_x, val_y), callbacks=[checkpoint_cb, earlystopping_cb])

val_loss = history.history['val_loss']
train_loss = history.history['loss']

plt.plot(val_loss, c='red')
plt.plot(train_loss, c='blue')

plt.show()