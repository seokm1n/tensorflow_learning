import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# 1. 학습시킬 데이터셋 준비
# 도미데이터 ( 캐글Fish Market 데이터 참조 )
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

# 빙어데이터 ( 캐글Fish Market 데이터 참조 )
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

length = bream_length + smelt_length
weight = bream_weight + smelt_weight
# length와 weight 두가지 특성데이터를 모델 학습에 입력데이터(train_x)로 활용
fish_data = [[l,w] for l,w in zip(length, weight)]
print(fish_data)

fish_data = np.array(fish_data)  # 2차원 배열로 변환
print(fish_data)
print(len(fish_data))  # 49개의 train_x

# 학습에 사용할 정답데이터(train_y)를 생성
# 1 : bream, 0 : smelt
fish_target = [1]*35 + [0]*14
print(fish_target)
# ====== 학습시킬 데이터셋 train_x, train_y 준비 완료 ======

# 모델 준비
knnmodel = KNeighborsClassifier()
# k = 5 (기본값) : 가장 가까운 5개의 이웃을 기준으로 분류 (hyperparameter)

# 모델 학습
knnmodel.fit(fish_data, fish_target)

# 모델 성능 평가
print(knnmodel.score(fish_data, fish_target))  # 정확도 1.0 ==> 과대적합

# 모델 예측
# predict() 함수는 하나의 데이터 전달이라도 2차원 배열로 전달해야함
print(knnmodel.predict([[30, 420]]))  # bream
print(knnmodel.predict([[9, 12]]))   # smelt

    