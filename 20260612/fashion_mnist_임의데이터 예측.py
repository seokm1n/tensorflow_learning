import cv2  # opencv-python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model  # type: ignore

np.set_printoptions(precision=8, suppress=True)  # 과학적 표기 대신 소수점 이하 8자리까지 표현
np.set_printoptions(threshold=np.inf)  # 무한으로 출력

img = cv2.imread("./20260612/man.jpg", cv2.IMREAD_GRAYSCALE)
# print(img)
print(img.shape)  # 330*330 ==> 28*28
# 원 이미지 사이즈를 학습한 모델의 입력 사이즈로 변경(맞춰줘야함)
# cv2.INTER_AREA ==> 보간법 사용
img_resize = cv2.resize(img, dsize=(28,28), interpolation=cv2.INTER_AREA)
print('img resize', img_resize)


# 이미지 색상 반전
img_reverted = cv2.bitwise_not(img_resize)

img_reverted = img_reverted.reshape(-1, 28, 28, 1) / 255.0  # 스케일 변환

print(img_reverted.shape)

fsmodel = load_model("./20260612/best_fashion.keras")

pred = fsmodel.predict(img_reverted)
pred_label = np.argmax(pred)

classes = [
    "티셔츠/상의 (T-shirt/top)",
    "바지 (Trouser)",
    "풀오버 (Pullover)",
    "드레스 (Dress)",
    "코트 (Coat)",
    "샌들 (Sandal)",
    "셔츠 (Shirt)",
    "스니커즈 (Sneaker)",
    "가방 (Bag)",
    "발목 부츠 (Ankle boot)",
]

print("예측 :", classes[pred_label])

plt.imshow(img_resize, cmap='gray')
plt.show()