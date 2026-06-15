import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D  # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping  # type: ignore

# 이미지 증강 유형 생성
train_image_generator = ImageDataGenerator(
    rescale=1.0 / 255.0,  # 픽셀값 정규화
    rotation_range=20,  # 회전
    # width_shift_range=0.2,  # 좌우 이동
    # height_shift_range=0.2,  # 상하 이동
    zoom_range=0.2,  # 확대/축소
    shear_range=0.2,  # 기울이기
    # horizontal_flip=True,  # 좌우 반전
    brightness_range=(0.8, 1.2),  # 밝기 변화
    fill_mode="nearest",  # 빈 영역 채우기
)

# 불러올 이미지 경로
train_dir = "./20260615/cnn_cats_and_dogs_dataset/train"

# 이미지를 읽어들이면서 이미지를 증강시켜주는 제너레이터 생성
train_data_gen = train_image_generator.flow_from_directory(
    train_dir,
    batch_size=4,
    shuffle=True,
    target_size=(150, 150),  # CNN 모델 입력 사이즈로 리사이즈
    # 디렉토리 내부 이미지를 불러올 때 어떤 형식으로 라벨링해서 불러올지
    class_mode="binary",  # 이진분류 할 때, 다중분류 ==> 'categorical'
    # save_to_dir="./20260615/cnn_cats_and_dogs_dataset/temp",  # 증강 이미지 저장 위치
    # save_prefix="gen",  # 증강 이미지 파일명 앞에 'gen'을 붙임
    # save_format="jpg",  # 저장할 이미지 확장자 명시
)

# test 이미지는 스케일 변환만 해줌
test_image_generator = ImageDataGenerator(rescale=1.0 / 255.0)

test_dir = "./20260615/cnn_cats_and_dogs_dataset/test"

test_data_gen = test_image_generator.flow_from_directory(
    test_dir, batch_size=4, shuffle=True, target_size=(150, 150), class_mode="binary"
)

# 모델 설계
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=3, activation="leaky_relu", input_shape=(150, 150, 3)))
model.add(MaxPooling2D(2))
model.add(Conv2D(filters=32, kernel_size=3, activation="leaky_relu"))
model.add(MaxPooling2D(2))
model.add(Conv2D(filters=64, kernel_size=3, activation="leaky_relu"))
model.add(MaxPooling2D(2))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation="leaky_relu"))
model.add(Dense(1, activation="sigmoid"))
model.summary()

# 모델 컴파일
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

checkpoint_cb = ModelCheckpoint("./20260615/catdog_best.keras", save_best_only=True)
earlystopping_cb = EarlyStopping(patience=3, restore_best_weights=True)

# steps_per_epoch=200 ==> 총 4000개 데이터 / batch_size=20 = 200
# validation_steps=10 ==> 검증 데이터 200 / batch_size=20 = 10
# 모델 학습
history = model.fit(
    train_data_gen,
    validation_data=test_data_gen,
    # steps_per_epoch=200,
    # validation_steps=10,
    epochs=50,
    verbose=1,
    callbacks=[checkpoint_cb, earlystopping_cb]
)

# 성능 시각화
acc = history.history['accuracy']  # train 데이터 정확도
val_acc = history.history['val_accuracy']  # test 데이터 정확도
loss = history.history['loss']  # train 데이터 손실
val_loss = history.history['val_loss']  # test 데이터 손실

epochs = np.arange(len(acc))
plt.figure()
plt.plot(epochs, loss, label='train_loss')
plt.plot(epochs, val_loss, label='val_loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(epochs, acc, label='train_acc')
plt.plot(epochs, val_acc, label='val_acc')
plt.legend()
plt.show()