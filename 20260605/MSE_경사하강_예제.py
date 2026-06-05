import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# 몸무게 x 와 키 y 데이터 셋 준비

x_data = np.array([0.65, 0.70, 0.75, 0.80, 0.68, 0.55, 0.60, 0.45, 1.00, 0.40])
# 연산을 위해 몸무게 스케일 조정
y_data = np.array([160, 175, 170, 180, 170, 156, 180, 149, 190, 154])

# 기울기(가중치) w 와  절편(편향) 값 초기화
w = 0
b = 0
# 학습률 지정
lr = 0.03

# 반복 횟수 지정
epochs = 110001
# 경사하강법 시작
for i in range(epochs):
    y_pred = w * x_data + b  # y = wx + b 예측 가설 함수 활용한 예측치
    error = y_data - y_pred  # 실제값 - 예측치, 즉 오차 값

    # 평균 제곱 오차를 w로 편미분한 결과
    w_diff = -(2 / len(x_data)) * sum((error) * x_data)

    # 평균 제곱 오차를 b로 편미분한 결과
    b_diff = -(2 / len(x_data)) * sum(error)

    w = w - lr * w_diff  # 미분 결과에 학습률을 곱한 후 기존 w 값 갱신
    b = b - lr * b_diff  # 미분 결과에 학습률을 곱한 후 기존 b 값 갱신

    # time.sleep(0.001)
    if i % 100 == 0:  # 100회 마다  갱신된 가중치, 편향 출력
        print("epoch = %.f , 가중치 = %.4f , 편향 = %.4f" % (i, w, b))

y_pred = w * x_data + b  # 위에서 구한 최적 w, b 로  데이터 예측
print("최적해 방정식 적용 예측 : ", y_pred)
plt.scatter(x_data * 100, y_data, c="red")
plt.plot(x_data * 100, y_pred)  # 직선 그래프
plt.xlabel("weight")
plt.ylabel("height")
plt.savefig("mse_.jpeg")
