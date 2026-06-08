import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # sklearn 선형회귀 모델
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

df = pd.read_csv("/home/sm/tf_env/20260605/basketball_stat.csv")
print(df.head())
print("=" * 80)
print(df.info())
print("=" * 80)
print(df["Pos"].value_counts())
print("=" * 80)

# (‘TRB’, ‘3P’ 특성 항목을 이용한 ‘Pos’ 분류 시각화)
sns.lmplot(x="TRB",y="3P",data=df,fit_reg=False,
           scatter_kws={"s": 150},markers=["o", "x"],hue="Pos",)
plt.title("TRB and 3P in 2d plane")
plt.show()

# (‘BLK’, ‘3P’ 특성 항목을 이용한 ‘Pos’ 분류 시각화)
sns.lmplot(x="BLK",y="3P",data=df,fit_reg=False,
           scatter_kws={"s": 150},markers=["o", "x"],hue="Pos",)
plt.title("BLK and 3P in 2d plane")
plt.show()

df.drop(["2P", "AST", "STL"], axis=1, inplace=True)
print(df.head())
print("=" * 80)

train, test = train_test_split(df, test_size=0.2, random_state=45)  # 20 비율로 훈련셋, 테스트셋 분리
print(train.shape[0])  # 80
print(test.shape[0])  # 20
print("=" * 80)

# k-fold 교차 검증으로 최적의 Knn 파라미터( k 값 ) 찾기
# cross_val_score() 활용
# 최적의 k를 찾기 위한 교차검증 수행할 k의 범위를 3부터 학습 데이터 절반까지 설정, 홀수로 설정이 적절

max_k_range = train.shape[0] // 2
k_list = []
for i in range(3, max_k_range, 2):
    k_list.append(i)
print(k_list)  # k 값 리스트
print("=" * 80)

# 데이터추출
x_train = train[["3P", "BLK", "TRB"]]
y_train = train[["Pos"]]


cross_validation_scores = []
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x_train, y_train.values.ravel(), cv=10, scoring="accuracy")
    cross_validation_scores.append(scores.mean())
    print('k : ', k, '\n' , scores)
    print("-"*80)

print(cross_validation_scores)
print("=" * 80)

plt.plot(k_list, cross_validation_scores)
plt.xlabel("number of k")
plt.ylabel("Accuracy")
plt.show()

# train/test 데이터 분리
train, test = train_test_split(df, test_size=0.2, random_state=45)  # 20 비율로 훈련셋, 테스트셋 분리
print(train.shape[0])  # 80
print(test.shape[0])  # 20
print("=" * 80)


knn = KNeighborsClassifier(n_neighbors=3)
# 데이터추출
x_train = train[["3P", "BLK", "TRB"]]
y_train = train[["Pos"]]

# KNN 모델학습
knn.fit(x_train, y_train.values.ravel())

# 테스트 데이터에서 분류를 위해 사용될 속성을 지정
x_test = test[["3P", "BLK", "TRB"]]  # x_test 데이터로 예측
y_test = test[["Pos"]]  # 분류 실제 값

# 테스트 데이터 예측 시작
pred = knn.predict(x_test)
print('예측 결과 : ', pred)
print("=" * 80)

# 모델 예측 정확도 출력
# print(y_test.values) # numpy (20,1) 2차원 배열  
# ravel() ==> 2차원 -> 1차원
# print(y_test.values.ravel()) # numpy 1차원 배열 데이터
# accuracy_score() : confusion matrix(혼동행렬) 활용 정확도 계산
print('accuracy : ' + str( accuracy_score(y_test.values.ravel(), pred)) ) # 0.95
print('acc : ', knn.score(x_test, y_test)) # 0.95
print("=" * 80)

# 실제 데이터와 예측 데이터 Dataframe 변환 출력
comparison = pd.DataFrame({'prediction':pred, 'truth value':y_test.values.ravel()})
print(comparison)
