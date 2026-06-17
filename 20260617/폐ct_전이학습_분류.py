import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dropout, Flatten, Dense  # type: ignore
from tensorflow.keras.applications import VGG16  # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping  # type: ignore

train_dir = "/home/sm/tf_env/20260617/Covid19-dataset/train"
test_dir = "/home/sm/tf_env/20260617/Covid19-dataset/test"

batch_size = 4  # 이미지 제너레이터 동작시 추출할 이미지 batch size
image_size = 224  # 이미지 제너레이터 동작시 추출할 이미지 size

# train 사용될 이미지 데이터 생성기
train_datagen = ImageDataGenerator(
    rotation_range=180,  # 회전
    width_shift_range=0.2,  # 좌우 이동
    height_shift_range=0.2,  # 상하 이동
    horizontal_flip=True,  # 좌우 반전
    vertical_flip=True,  # 상하 반전
)

# 디렉토리 참조해서 가져온 데이터를 flow 시킴
train_data_gen = train_datagen.flow_from_directory(
    train_dir,  # train directory 참조
    target_size=(image_size, image_size),  # (224, 224) 크기
    batch_size=batch_size,  # 4
    class_mode="categorical",  # 파일 디렉토리로 카테고리 class 분류(2D one-hot 라벨)
    shuffle=True,  # 순서 무작위
)

# test 이미지 증강 없이 생성
test_datagen = ImageDataGenerator()

# test 이미지 읽어들이면서 제너레이터 생성
test_data_gen = test_datagen.flow_from_directory(
    test_dir,  # test directory 참조
    target_size=(image_size, image_size),  # (224, 224) 크기
    batch_size=batch_size,  # 4
    class_mode="categorical",
    shuffle=False,
)

print(train_data_gen.class_indices)

class_labels = list(test_data_gen.class_indices.keys())  # 2분류 라벨
print(class_labels)  # [0] = 'Covid', [1] = 'Normal'

vgg16_layer = VGG16(
    weights="imagenet",  # imagenet에서 학습된 가중치
    include_top=False,  # top층(FC layer)의 가중치는 가져오지 않고 재설계
    input_shape=(image_size, image_size, 3),
)

# vgg16_layer.summary()

for layer in vgg16_layer.layers:
    layer.trainable = False  # 튜닝할 Dense 이후 layer만 학습

newmodel = Sequential()
newmodel.add(vgg16_layer)

# 상단층 재설계
newmodel.add(Flatten())
newmodel.add(Dense(1024, activation="leaky_relu"))
newmodel.add(Dropout(0.3))
newmodel.add(Dense(units=2, activation="softmax"))  # 2class 분류

newmodel.summary()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)  # 0.00001
newmodel.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

print(int(np.ceil(train_data_gen.samples / train_data_gen.batch_size)))  # ==> batch_size = 46
# print(train_data_gen.samples) # 181 개 학습 이미지 샘플
# print(train_data_gen.batch_size) # 4 batch size

checkpoint_cb = ModelCheckpoint(filepath="/home/sm/tf_env/20260617/covid19_best.keras", 
                                verbose=1, save_best_only=True,)

earlystopping_cb = EarlyStopping(patience=5, restore_best_weights=True)

# 모델 훈련
history = newmodel.fit(
    train_data_gen,
    steps_per_epoch=int(np.ceil(train_data_gen.samples / train_data_gen.batch_size)),
    epochs=30,
    validation_data=test_data_gen,
    validation_steps=int(np.ceil(test_data_gen.samples / test_data_gen.batch_size)),
    verbose=1,
    callbacks=[checkpoint_cb, earlystopping_cb]
)
