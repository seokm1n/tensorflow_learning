import pandas as pd
from sklearn.datasets import load_iris # 붓꽃 데이터셋
from sklearn.model_selection import train_test_split # 분리할때 사용

# 나이브 베이즈 ==> 조건부 확률로 분류를 하는 모델
from sklearn.naive_bayes import GaussianNB # 데이터특징 가우시안 정규분포를 취할때
# 분류모델로 사용 ==> GaussianNB
from sklearn import metrics # 혼동행렬
from sklearn.metrics import accuracy_score # 정확도 평가

dataset = load_iris()
print(dataset)

# 모델에 학습 시킬 데이터셋 준비
# train / test 데이터를 분할
train_x, test_x , train_y, test_y = \
    train_test_split(dataset['data'], dataset['target'], test_size=0.2, random_state=42)

print(train_x[:5])
print(test_x[:5])

# 가우시안 나이브베이즈  모델 준비
gnb_model = GaussianNB()

# 모델 학습
gnb_model.fit(train_x, train_y)

# 모델 예측
pred = gnb_model.predict(test_x[:3])
print(pred)
print('실제  정답 : ', test_y[:3])

print('test acc : ', gnb_model.score(test_x, test_y)) # 모델 성능 평가

