import os
import numpy as np
from tensorflow.keras.applications import vgg16  # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # type: ignore

vgg16model = vgg16.VGG16()
vgg16model.summary()

model = vgg16.VGG16() # VGG16 모델 ==> 네트워크 구조, 가중치 모두 로딩
model.summary() # 모델 구조 출력

# 새로 예측할 사용자 이미지파일 경로 + 파일 불러오기
predict_img_dir = "/home/sm/tf_env/20260616/testimage_dataset/"
os.chdir("/home/sm/tf_env/20260616/testimage_dataset/")  # 경로 변경

file_info_list = []
for file_info in os.listdir(predict_img_dir):
    file_info_list.append(predict_img_dir + file_info)
print(file_info_list)

img = load_img(file_info_list[6], target_size=(224,224))
image = img_to_array(img)  # 이미지 객체를 넘파이 배열로 변경
# print("load image array : ")
# print(image) # rgb(255, 255, 255) : 흰색, rgb(0, 0, 0) : 검정색
# print(image.shape) # (224, 224, 3)
# vgg16 은 (None, 224, 224, 3) 입력 형태를 기대함으로
# shape 변경
image = image.reshape( ( 1, 224, 224, 3) )

# Vgg 모델 입력을 위한 픽셀값 조정 전처리
# -255 ~ 255 사이값으로 정규화, rgb -> bgr 순서로 바꿈
image = vgg16.preprocess_input(image)
# print('preprocess image : ')
# print(image)
# 이미지 분류 예측
pred = model.predict(image)
# print(pred)
# print(np.argmax(pred[0]))

# 예측 결과물을 파싱(디코딩)
label = vgg16.decode_predictions(pred)
print(label)
label = label[0][0]
print(label)