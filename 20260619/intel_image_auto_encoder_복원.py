import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import image_dataset_from_directory

# 경로 설정
TEST_DIR = "/home/sm/tf_env/20260615/intel_image_classification/seg_test"
IMG_SIZE = (64, 64)
BATCH_SIZE = 64
MODEL_PATH = "/home/sm/tf_env/20260619/intel_image_autoencoder_model.keras"

# 테스트 데이터셋 준비
raw_test_ds = image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode=None,
    shuffle=False
)


def preprocess_for_autoencoder(images):
    images = tf.cast(images, tf.float32) / 255.0
    return images, images


test_ds = raw_test_ds.map(
    preprocess_for_autoencoder,
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

# 저장된 모델 로드
autoencoder_model = load_model(MODEL_PATH)

# 원본 vs 복원 결과 확인
for input_images, _ in test_ds.take(1):
    reconstructed_images = autoencoder_model.predict(input_images, verbose=0)
    break

num = 5
plt.figure(figsize=(15, 7))

for i in range(num):
    # 원본 이미지
    ax1 = plt.subplot(2, num, i + 1)
    ax1.imshow(input_images[i].numpy())
    ax1.set_title(f"original_{i}")
    ax1.axis("off")

    # 복원 이미지
    ax2 = plt.subplot(2, num, i + num + 1)
    ax2.imshow(reconstructed_images[i])
    ax2.set_title(f"reconstructed_{i}")
    ax2.axis("off")

plt.tight_layout()
plt.show()
