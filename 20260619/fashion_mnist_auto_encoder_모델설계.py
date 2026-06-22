import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore
from tensorflow.keras.optimizers import Adam as Adam  # type: ignore

# 오토인코더 경우 훈련데이터를 타깃으로 이미지 재건 출력 학습모델(비지도)로서 타깃 데이터 불필요
(train_x, _), (test_x, _) = (fashion_mnist.load_data())  # (28, 28) 흑백 패션이미지 데이터 로딩
print(len(train_x))  # 60000
print(train_x.shape)  # (60000, 28, 28)

train_x = train_x.reshape(-1, 28, 28, 1) / 255.0  # CNN 입력을 위한 차원 변경
test_x = test_x.reshape(-1, 28, 28, 1) / 255.0  # -1 : 자동으로 결정
print(train_x.shape)  # (60000, 28, 28, 1) : (샘플수, 3차원 이미지)
print(test_x.shape)  # (10000, 28, 28, 1) : (샘플수, 3차원 이미지)
# print(train_x[0])  # / 255.0 으로 정규화

# 케라스 Sequential API 활용 autoencoder 모델 설계
autoencoder_model = Sequential()

# 인코더 부분
autoencoder_model.add(Input(shape=(28,28,1)))
autoencoder_model.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='leaky_relu'))
autoencoder_model.add(MaxPooling2D(pool_size=2))
autoencoder_model.add(Conv2D(filters=8, kernel_size=3, padding='same', activation='leaky_relu'))
autoencoder_model.add(MaxPooling2D(pool_size=2))
autoencoder_model.add(Conv2D(filters=8, kernel_size=3, strides=2, padding='same', activation='leaky_relu'))
# strides=2 ==> 잠재 벡터 압축 ==> (4, 4, 8)

# 디코더 부분
autoencoder_model.add(Conv2D(filters=8, kernel_size=3, padding='same', activation='leaky_relu'))
autoencoder_model.add(UpSampling2D())
autoencoder_model.add(Conv2D(filters=8, kernel_size=3, padding='same', activation='leaky_relu'))
autoencoder_model.add(UpSampling2D())
autoencoder_model.add(Conv2D(filters=16, kernel_size=3, padding='valid', activation='leaky_relu'))
# padding='valid' ==> default --> (14, 14, 16)
autoencoder_model.add(UpSampling2D())
autoencoder_model.add(Conv2D(filters=1, kernel_size=3, padding='same', activation='sigmoid'))
autoencoder_model.summary()

# 모델 컴파일
autoencoder_model.compile(loss='mse', optimizer=Adam(learning_rate=1e-5), metrics=['mae'])

# 모델 학습
EarlyStopCB = EarlyStopping(
    monitor="val_loss", verbose=1, patience=3, restore_best_weights=True
)
ModelCheckCB = ModelCheckpoint(
    "/home/sm/tf_env/20260619/original_fashion_mnist_autoencoder_model.keras",
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
)

# 'adam' ==> 디폴트 학습률(lr) ==> 0.001(1e-3)
history = autoencoder_model.fit(train_x, train_x, epochs=50, batch_size=128, 
                                validation_data=(test_x, test_x), verbose=1, 
                                callbacks=[EarlyStopCB, ModelCheckCB])

trainloss = history.history['loss'] # 훈련 손실
valloss = history.history['val_loss'] # 검증 손실

plt.plot(trainloss)
plt.plot(valloss)
plt.show()