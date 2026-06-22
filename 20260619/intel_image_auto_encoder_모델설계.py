import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory  # type: ignore
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore
# from tensorflow.keras.optimizers import Adam as Adam  # type: ignore

# -------------------------------------------------------------
# [1] 하이퍼파라미터 및 경로 설정
# -------------------------------------------------------------
# TODO: 본인의 Intel Image Classification 데이터셋 경로에 맞게 수정하세요.
TRAIN_DIR = "/home/sm/tf_env/20260615/intel_image_classification/seg_train"
TEST_DIR = "/home/sm/tf_env/20260615/intel_image_classification/seg_test"

IMG_SIZE = (64, 64)
BATCH_SIZE = 64

# -------------------------------------------------------------
# [2] 디렉토리로부터 데이터셋 읽어오기 (기본은 라벨 포함)
# -------------------------------------------------------------
raw_train_ds = image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode=None, # Autoencoder이므로 분류용 라벨은 로드하지 않음 (오직 이미지 데이터만)
    shuffle=True
)

raw_test_ds = image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode=None,
    shuffle=False
)

# -------------------------------------------------------------
# [3] Autoencoder를 위한 전처리 파이프라인 (정규화 및 X=Y 매핑)
# -------------------------------------------------------------
def preprocess_for_autoencoder(images):
    # 1. 픽셀 값을 [0, 255]에서 [0.0, 1.0] 범위로 정규화
    images = tf.cast(images, tf.float32) / 255.0
    # 2. ★ 핵심: 입력을 그대로 정답(Target)으로 반환 (X, Y) -> (images, images)
    return images, images

# .map()을 이용해 전체 데이터셋에 전처리 적용
# tf.data.AUTOTUNE을 주어 성능을 최적화합니다.
train_ds = raw_train_ds.map(preprocess_for_autoencoder, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = raw_test_ds.map(preprocess_for_autoencoder, num_parallel_calls=tf.data.AUTOTUNE)

# 메모리 버퍼링 및 프리페치 설정 (학습 속도 향상)
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

print("데이터셋 준비 완료!")

# 2. 데이터 로딩 확인 및 시각화 

# 하나의 배치를 가져와서 형태 확인
# for input_images, target_images in train_ds.take(1):
#     print(f"입력 배치 크기: {input_images.shape}")   # (64, 64, 64, 3)
#     print(f"타겟 배치 크기: {target_images.shape}")   # (64, 64, 64, 3)
#     print(f"최대 픽셀 값: {tf.reduce_max(input_images).numpy()}") # 1.0 내외인지 확인
    
#     # 입력과 타겟이 완벽히 똑같은지 한 번 더 확인
#     is_identical = tf.reduce_all(tf.equal(input_images, target_images)).numpy()
#     print(f"입력과 타겟이 완벽히 일치합니까?: {is_identical}")

#     # 샘플 이미지 한 장 띄워보기
#     plt.figure(figsize=(3, 3))
#     plt.imshow(input_images[0].numpy())
#     plt.title("Sample Input Image")
#     plt.axis("off")
#     plt.show()


# 케라스 Sequential API 활용 autoencoder 모델 설계
autoencoder_model = Sequential()

# 인코더 부분
autoencoder_model.add(Input(shape=(64, 64, 3)))
autoencoder_model.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='leaky_relu'))
autoencoder_model.add(MaxPooling2D(pool_size=2))
autoencoder_model.add(Conv2D(filters=8, kernel_size=3, padding='same', activation='leaky_relu'))
autoencoder_model.add(MaxPooling2D(pool_size=2))
autoencoder_model.add(Conv2D(filters=8, kernel_size=3, strides=2, padding='same', activation='leaky_relu'))

# 디코더 부분
autoencoder_model.add(Conv2D(filters=8, kernel_size=3, padding='same', activation='leaky_relu'))
autoencoder_model.add(UpSampling2D())
autoencoder_model.add(Conv2D(filters=8, kernel_size=3, padding='same', activation='leaky_relu'))
autoencoder_model.add(UpSampling2D())
autoencoder_model.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='leaky_relu'))
autoencoder_model.add(UpSampling2D())
autoencoder_model.add(Conv2D(filters=3, kernel_size=3, padding='same', activation='sigmoid'))
autoencoder_model.summary()

# 모델 컴파일
autoencoder_model.compile(loss='mse', optimizer='adam', metrics=['mae'])

# 모델 학습
EarlyStopCB = EarlyStopping(
    monitor='val_loss',
    verbose=1,
    patience=3,
    restore_best_weights=True
)
ModelCheckCB = ModelCheckpoint(
    '/home/sm/tf_env/20260619/intel_image_autoencoder_model.keras',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
)

history = autoencoder_model.fit(
    train_ds,
    epochs=20,
    validation_data=test_ds,
    callbacks=[EarlyStopCB, ModelCheckCB]
)

# 손실 곡선 그리기
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(train_loss)
plt.plot(val_loss)
plt.show()

# 복원 결과 확인
for original_batch, _ in test_ds.take(1):
    reconstructed = autoencoder_model.predict(original_batch, verbose=0)
    break

for i in range(6):
    plt.subplot(2, 6, i + 1)
    plt.imshow(original_batch[i].numpy())
    plt.axis('off')

    plt.subplot(2, 6, i + 7)
    plt.imshow(reconstructed[i])
    plt.axis('off')
plt.tight_layout()
plt.show()