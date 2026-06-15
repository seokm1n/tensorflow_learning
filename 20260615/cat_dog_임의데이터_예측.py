import cv2  # opencv-python
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore

np.set_printoptions(
    precision=8, suppress=True
)  # 과학적 표기 대신 소수점 이하 8자리까지 표현
np.set_printoptions(threshold=np.inf)  # 무한으로 출력

img = cv2.imread("./20260615/cat.jpg")
# print(img)
print(img.shape)
# 원 이미지 사이즈를 학습한 모델의 입력 사이즈로 변경(맞춰줘야함)
# cv2.INTER_AREA ==> 보간법 사용
img_resize = cv2.resize(img, (150, 150))


img_resize = img_resize.reshape(-1, 150, 150, 3) / 255.0  # 스케일 변환

print(img_resize.shape)

cdmodel = load_model("./20260615/catdog_best.keras")

pred = cdmodel.predict(img_resize, verbose=0)

for i, p in enumerate(pred):
    if p[0] >= 0.5:
        print(f"이미지가 강아지일 확률 : {p[0]*100:.2f}%")
        print("예측결과 : 강아지")
    else:
        print(f"이미지가 고양이일 확률 : {100-p[0]*100:.2f}%")
        print("예측결과 : 고양이")
        