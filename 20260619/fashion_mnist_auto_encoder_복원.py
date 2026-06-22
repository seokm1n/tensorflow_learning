import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore

(train_x, _), (test_x, _) = (fashion_mnist.load_data())  # (28, 28) 흑백 패션이미지 데이터 로딩
# print(len(train_x))  # 60000
# print(train_x.shape)  # (60000, 28, 28)
train_x = train_x.reshape(-1, 28, 28, 1) / 255.0  # CNN 입력을 위한 차원 변경
test_x = test_x.reshape(-1, 28, 28, 1) / 255.0  # -1 : 자동으로 결정
# print(train_x.shape)  # (60000, 28, 28, 1) : (샘플수, 3차원 이미지)
# print(test_x.shape)  # (10000, 28, 28, 1) : (샘플수, 3차원 이미지)

autoencoder_model = load_model('/home/sm/tf_env/20260619/original_fashion_mnist_autoencoder_model.keras')
ae_predict_imge = autoencoder_model.predict(test_x)
print(ae_predict_imge.shape)
print(ae_predict_imge[0])

num = 5 # 원본과 5개 비교
plt.figure(figsize=(15,7))
for i in range(num): # 원본 이미지
    ax1 = plt.subplot(2, num, i+1) # (2, 5) 플롯 중 윗줄 서브플롯
    ax1.imshow(test_x[i].reshape(28,28), cmap='gray')
    ax1.set_title('original_image %d' %i) # 서브플롯에 타이틀 추가
    ax1.axis('off')
    # 복원 이미지
    ax2 = plt.subplot(2, num, i + num + 1) # (2, 5) 플롯 중 아래줄 서브플롯
    ax2.imshow(ae_predict_imge[i].reshape(28, 28) , cmap='gray')
    ax2.set_title('autoenc_imge %d' % i) # 서브플롯에 타이틀 추가
    ax2.axis('off')
plt.show()