from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore

# 이미지 증강 유형 생성
train_image_generator = ImageDataGenerator(
    rescale=1.0 / 255.0, rotation_range=20, height_shift_range=0.2
)

# 불러올 이미지 경로
train_dir = "./20260615/cnn_cats_and_dogs_dataset/train"

# 이미지를 읽어들이면서 이미지를 증강시켜주는 제너레이터 생성
train_data_gen = train_image_generator.flow_from_directory(
    train_dir,
    batch_size=2,
    shuffle=False,
    target_size=(150, 150),  # CNN 모델 입력 사이즈로 리사이즈 해라
    save_to_dir="./20260615/cnn_cats_and_dogs_dataset/temp",  # 증강 이미지 저장 위치
    save_prefix="gen",  # 증강 이미지 파일명 앞에 'gen'을 붙임
    save_format="jpg",  # 저장할 이미지 확장자 명시
)

model.fit(
    train_data_gen,
    validation_data=test_data_gen,
    steps_per_epoch=200,
    epochs=100,
    validation_steps=10,
    verbose=1,
)
