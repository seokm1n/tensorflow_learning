import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 출력 옵션
pd.set_option("display.max_rows", 20)
pd.set_option("display.max_columns", 500)

titanic_df = pd.read_csv("/home/sm/tf_env/20260608/titanic_passengers.csv")
# print(titanic_df.head())
# print("=" * 80)

# Survived 컬럼데이터를 타깃으로 활용(0, 1)
# 머신러닝 sklearn은 타깃이 문자열이어도 성능평가 가능
# Survived 컬럼열 데이터를 변경
# 0 --> fail, 1 --> survive
# titanic_df.info()
# print("=" * 80)
titanic_df["Survived"] = titanic_df["Survived"].map({1: "survive", 0: "fail"})
# print(titanic_df.head())
# print("=" * 80)

# 모델 입력 데이터 준비
# gender, Age, Pclass 3가지 컬럼 데이터가 생존/비생존에 많은 영향을 미침
# print(titanic_df["gender"])
# "female": 1, "male": 0 으로 변경
titanic_df["gender"] = titanic_df["gender"].map({"female": 1, "male": 0})
# print(titanic_df.head())
# print("=" * 80)
# titanic_df.info()  # age 컬럼에 np.NaN 결측치가 존재 ==> 결측치 제거 필요
# print("=" * 80)
titanic_df.dropna(subset='Age', inplace=True)  # Age 컬럼 기준 결측치 제거

# age 컬럼에 결측치를 평균데이터로 채워서 사용
# titanic_df['Age'].fillna(value=titanic_df['Age'].mean(), inplace=True)
# print(titanic_df.head())
# print("=" * 80)
# titanic_df.info()
# print("=" * 80)

# print(titanic_df['Pclass'].head())  # 1등석과 2등석 데이터만 추출
# print("=" * 80)
# pandas 원핫인코딩으로 변환해주는 메서드 ==> get_dummies()
# 원핫인코딩 ==> 모든 수치데이터를 0과 1로만 표현(binary)
# 1 ==> 001, 2 ==> 010, 3 ==> 100
onehot_pclass = pd.get_dummies(titanic_df["Pclass"], prefix='Class', dtype=int)
# print(onehot_pclass.head())
# print("=" * 80)
# axis=1 ==> 열축으로 두 DataFrame을 병합
titanic_df = pd.concat([titanic_df, onehot_pclass], axis=1)
# print(titanic_df.head())
# print("=" * 80)

# Age, gender, Class_1, Class_2 4개 컬럼 데이터를 모델 입력 데이터로 사용
# 'Survived' 컬럼은 모델 정답(타깃) 데이터로 사용
titanic_df_x = titanic_df[['gender', 'Age', 'Class_1', 'Class_2']]
print(titanic_df_x.head())
print("=" * 80)

titanic_df_y = titanic_df[['Survived']]
print(titanic_df_y.head())
print("=" * 80)

from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(titanic_df_x, titanic_df_y, 
                                                    random_state=42)

print(train_x[:5])
print("=" * 80)

# 특성데이터의 스케일 변환(정규화) ==> 표준점수 정규화((각 특성 = 평균) / 표준편차)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_x)  # train 데이터를 정규화하는 방법을 학습하고
                                              # 학습이 끝나면 변환 작업 수행
# test 데이터셋은 transform() 만 해서 적용만 해야함
test_scaled = scaler.transform(test_x)

print(train_scaled[:5])  # DataFrame --> 넘파이 배열
print("=" * 80)

# 모델 생성
# 로지스틱 회귀(분류) 모델 준비
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()

# 모델 학습
lr_model.fit(train_scaled, train_y)

# 모델 성능 평가
print('test acc : ', lr_model.score(test_scaled, test_y))
print('train acc : ', lr_model.score(train_scaled, train_y))
# 둘다 성능 별로 ==> 과소적합

# 가중치(w), 절편(b)
# : coef_, intercept_
print('coef_ : ', lr_model.coef_, ', intercept_ : ', lr_model.intercept_)
