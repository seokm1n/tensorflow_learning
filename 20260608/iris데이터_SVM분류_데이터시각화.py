import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris  # sklearn iris 데이터셋 로드
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import svm  # svm 모델 추가

iris = load_iris()
# print(iris)

iris_df = pd.DataFrame(
    np.column_stack([iris["data"], iris["target"]]),
    columns=["sepal_len", "sepal-wd", "petal_len", "petal_wd", "target"],
)
print(iris_df)
print("=" * 80)

x_petal_len_wd = iris_df[["petal_len", "petal_wd"]]
print(x_petal_len_wd.sample(5))
y_target = iris_df[["target"]]
print(y_target.sample(5))
print("=" * 80)

# print(y_target.values.ravel())  # ravel() ==> 2차원 --> 1차원 변경

# 최적의 하이퍼파라미터를 갖는 모델을 교차검증을 활용해서 찾기
# ==> GridSearch


def svc_param_selection(x, y, nfolds):
    svm_parameters = [
        {
            "kernel": ["rbf"],
            "gamma": [0.1, 0.3, 0.5, 0.7, 1.0],
            "C": [0.3, 0.7, 1, 1.3, 1.5],
        }
    ]
    clf = GridSearchCV(svm.SVC(), svm_parameters, cv=nfolds)  # 10번 교차검증
    clf.fit(x, y)  # 10번 교차검증 진행하고 마지막 최적 하이퍼파라미터로 업데이트
    print("최적 파라미터 : ", clf.best_params_)  # 최적 파라미터
    print("최적 성능 : ", clf.best_score_)  # 최적 성능
    print("최적 모델 : ", clf.best_estimator_)  # 최적 모델

    return clf.best_estimator_


svc_param_selection(x_petal_len_wd, y_target.values.ravel(), 10)
print("=" * 80)

# 'C': 0.3, 'gamma': 0.7

# train/test 분리
train_x, test_x, train_y, test_y = train_test_split(
    x_petal_len_wd, y_target, random_state=42  # , test_size=0.3
)

print(train_x[:5])
print(test_x[:5])
print("=" * 80)

# 모델 준비 ==> svm 모델로 분류
svm_model = svm.SVC(C=0.3, kernel="rbf", gamma=0.7)  # cost=0.3, gamma=0.7

# train 데이터 활용해서 모델 학습
svm_model.fit(train_x.values, train_y.values.ravel())  # train_y ==> 2차원 데이터프레임

# 성능 평가
print("train acc : ", svm_model.score(train_x.values, train_y.values.ravel()))
print(
    "test acc : ", svm_model.score(test_x.values, test_y.values.ravel())
)  # 테스트 데이터에 대한 성능 평가

# 예측
pred = svm_model.predict([[4.7, 1.7]])
print(pred)

# 산점도 시각화
lnames = iris["target_names"]  # 꽃 이름 정보
markers = ["o", "^", "s"]  # 표시할 마커 리스트
colors = ["blue", "green", "red"]  # 표시할 마커 색상

# X, Y 좌표(꽃잎 길이, 꽃잎 너비) 학습(train) 데이터 scatter 출력
for i in set(train_y["target"]):
    idx = np.where(train_y["target"] == i)  # np.where() ==> 특정 데이터 추출
    print(idx[0])
    # iloc ==> FancyIndexing으로 train.iloc[idx[0]], trainx.iloc[idx] 둘다 가능
    plt.scatter(
        train_x.iloc[idx[0]]["petal_len"],
        train_x.iloc[idx[0]]["petal_wd"],
        c=colors[int(i)],
        marker=markers[int(i)],
        label=lnames[int(i)] + "(train)",
        s=80,
        alpha=0.3,
    )

# X, Y 좌표(꽃잎 길이, 꽃잎 너비) 테스트(test) 데이터 scatter 출력
for i in set(test_y["target"]):
    idx = np.where(test_y["target"] == i)
    print(idx[0])
    plt.scatter(
        test_x.iloc[idx[0]]["petal_len"],
        test_x.iloc[idx[0]]["petal_wd"],
        marker=markers[int(i)],
        label=lnames[int(i)] + "(train)",
        s=130,
        edgecolors="black",  # 경계선 색
        facecolors="none",  # 속을 비움(색칠 x)
    )

plt.legend(loc="best")
plt.show()
