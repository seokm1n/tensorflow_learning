import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore

(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

# print(test_x[0])
# plt.imshow(test_x[0],cmap='gray')
# plt.show()

test_scaled = test_x.reshape(-1, 28, 28, 1) / 255.0
test_y_onehot = to_categorical(test_y)

fsmodel = load_model("./20260612/best_fashion.keras")

# 예측
preds = fsmodel.predict(test_scaled)  # 검증 데이터 예측
print("pred[0] : ", preds[1])  # 검증 데이터로 예측한 첫번째 결과값
pred_label = np.argmax(preds[1])  # 예측 결과값에 대한 최대 예측값(약 1) 인덱스 추출
print("pred_label : ", pred_label)
print("test_y[0] : ", test_y_onehot[1])  # 검증 데이터의 타깃값 확인

# 모델 성능 평가
val_loss, val_accuracy = fsmodel.evaluate(test_scaled, test_y_onehot)
print("\nval_loss : %.4f, val_accuracy : %.4f" % (val_loss, val_accuracy))

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
print("정답 :", classes[test_y[1]])
