import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical  # type: ignore

from tensorflow.keras.models import load_model  # type: ignore

np.set_printoptions(precision=8, suppress=True)  # 과학적 표기 대신 소수점 이하 8자리까지 표현
np.set_printoptions(threshold=np.inf)  # 무한으로 출력

mnist = datasets.load_digits()  # 손글씨 이미지
features = mnist["data"]
print(len(features[7]))
print(len(features))  # 1797
labels = mnist["target"]
print(len(labels))  # 1797
print(np.unique(labels, return_counts=True))


train_x, val_x, train_y, val_y = train_test_split(features, labels, test_size=0.2, random_state=42)
print(len(val_x[0:1].shape))
print(len(val_y[0:1].shape))

mnist_model = load_model("./20260615/best_mnist.keras")

new_data = np.array([[0., 0., 9., 9., 9., 9., 0., 0.],
                     [0., 0., 9., 0., 0., 0., 9., 0.],
                     [0., 9., 0., 0., 0., 0., 9., 7.],
                     [0., 0., 0., 0., 0., 0., 9., 7.],
                     [0., 0., 0., 0., 0., 0., 9., 0.],
                     [0., 0., 0., 0., 9., 9., 0., 0.],
                     [0., 7., 7., 9., 0., 0., 0., 0.],
                     [7., 9., 9., 9., 9., 9., 9., 9.]])

new_data = new_data.reshape(-1, 8, 8, 1) / 255.0

pred = mnist_model.predict(new_data)
pred_label = np.argmax(pred)
print(pred)
print("예측 결과 : ", pred_label)