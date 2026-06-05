import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # train/test 분리
from sklearn.neighbors import KNeighborsRegressor  # 특정값을 예측하는 회귀모델

#농어길이데이터 (캐글Fish Market 데이터참조)
perch_length=np.array([8.4,13.7,15.0,16.2,17.4,18.0,18.7,19.0,19.6,20.0,21.0,
21.0,21.0,21.3,22.0,22.0,22.0,22.0,22.0,22.5,22.5,22.7,
23.0,23.5,24.0,24.0,24.6,25.0,25.6,26.5,27.3,27.5,27.5,
27.5,28.0,28.7,30.0,32.8,34.5,35.0,36.5,36.0,37.0,37.0,
39.0,39.0,39.0,40.0,40.0,40.0,40.0,42.0,43.0,43.0,43.5,
44.0])

# 농어무게데이터 (캐글FishMarket 데이터참조)
perch_weight=np.array([5.9,32.0,40.0,51.5,70.0,100.0,78.0,80.0,85.0,85.0,110.0,
115.0,125.0,130.0,120.0,120.0,130.0,135.0,110.0,130.0,
150.0,145.0,150.0,170.0,225.0,145.0,188.0,180.0,197.0,
218.0,300.0,260.0,265.0,250.0,250.0,300.0,320.0,514.0,
556.0,840.0,685.0,700.0,700.0,690.0,900.0,650.0,820.0,
850.0,900.0,1015.0,820.0,1100.0,1000.0,1100.0,1000.0,
1000.0])

# 모델 입력 특성(x) ==> 농어의 길이
# 정답(traget) ==> 농어의 무게
# print(len(perch_length))
# print(len(perch_weight))

# train 데이터와 test 데이터로 분리
train_x, test_x, train_y, test_y = train_test_split(perch_length, perch_weight, 
                                                    random_state=42)

# print(len(train_x), len(test_x))
# print(len(train_y), len(test_y))
# print(train_x.shape, test_x.shape)

# 1차원 shape ==> 2차원으로 변경
train_x = train_x.reshape(-1,1)  # 넘파이 배열 크기 자동 지정 : -1, 원래 (42,1)
test_x = test_x.reshape(-1,1)
print(train_x.shape, test_x.shape)

# knn 회귀 모델 준비
knn_reg = KNeighborsRegressor(n_neighbors=3)

# 모델 학습(훈련)
# train_x ==> 농어 길이, train_y ==> 농어 무게
knn_reg.fit(train_x, train_y)

# 모델 성능 평가(과소 적합)
print('test : ', knn_reg.score(test_x, test_y))  # test 데이터에 대한 성능 0.992

print('train : ', knn_reg.score(train_x, train_y))  # train 데이터에 대한 성능 0.969 (과소 적합)

# test 데이터를 이용해서 예측
test_pred = knn_reg.predict(test_x)
print(test_pred)  # 14개 테스트 데이터 길이에 따른 무게 예측

# from sklearn.metrics import mean_absolute_error  # MAE

# mae = mean_absolute_error(test_y, test_pred)  # |y - ^y| 평균 절대값 오차
# print('mae : ', mae)  # 오차 : 19.157

# 길이가 40인 농어의 무게 예측
pred = knn_reg.predict([[40]])
print('40 : ', pred)  # 921.7

# 길이가 80, 120인 농어의 무게 예측
pred = knn_reg.predict([[80],[120]])
print('[80 120]: ', pred)  # 결과 1033.3 동일

# 시각화
# train_x(길이), train_y(무게) 산점도 시각화
plt.scatter(train_x, train_y)
# 길이가 50인 농어의 무게 예측해서 산점도 출력
plt.scatter(50, knn_reg.predict([[50]]), marker='^', c='red')  # y=1033
plt.savefig("knn_reg.jpeg")
