from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D  # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore

# 이미지 증강 유형 생성
train_image_generator = ImageDataGenerator(
    rescale=1.0 / 255.0,          # 픽셀값 정규화
    rotation_range=90,            # 회전
    width_shift_range=0.2,        # 좌우 이동
    height_shift_range=0.2,       # 상하 이동
    zoom_range=0.2,               # 확대/축소
    shear_range=0.2,              # 기울이기
    horizontal_flip=True,         # 좌우 반전
    brightness_range=(0.8, 1.2),  # 밝기 변화
    fill_mode="nearest"           # 빈 영역 채우기
)

# 불러올 이미지 경로
train_dir = "./20260615/cnn_cats_and_dogs_dataset/train"

# 이미지를 읽어들이면서 이미지를 증강시켜주는 제너레이터 생성
train_data_gen = train_image_generator.flow_from_directory(
    train_dir,
    batch_size=2,
    shuffle=False,
    target_size=(150, 150),  # CNN 모델 입력 사이즈로 리사이즈
    save_to_dir="./20260615/cnn_cats_and_dogs_dataset/temp",  # 증강 이미지 저장 위치
    save_prefix="gen",  # 증강 이미지 파일명 앞에 'gen'을 붙임
    save_format="jpg",  # 저장할 이미지 확장자 명시
    class_mode='binary'
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
model.add(Dense(512, activation="leaky_relu"))
model.add(Dense(1, activation="sigmoid"))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

i = 0
for b in train_data_gen:
    i += 1
    if i > 2:
        break