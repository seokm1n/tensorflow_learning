import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris  # sklearn iris 데이터셋 로드

iris = load_iris()
# print(iris)  # iris = 딕셔너리 형태
# print(iris['data'][:5])  # iris['data'] = 넘파이 배열
# print(iris['feature_names'][:5])
# print(iris['target'][:5])

iris_data = np.column_stack([iris['data'], iris['target']])  # 열방향으로 합침, 컬럼 추가도 같은 결과
print(iris_data[:5])
print('='*80)
iris_df = pd.DataFrame(data=iris_data, columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'target'])

print(iris_df.head())
print('='*80)

# plt.scatter(iris_df['sepal_len'], iris_df['sepal_wid'], data=iris_df)
# plt.savefig('iris_scatter.jpeg')

# iris 붓꽃 KNN 분류 모델에서 모델 입력 특성데이터(train_x)는 'petal_len', 'petal_wid' 사용
# 타겟데이터(train_y)는 'target' 사용
iris_train_x = iris_df[['petal_len', 'petal_wid', 'target']].copy()#.values
print(iris_train_x)
print('='*80)

# # iris_train_x 데이터 시각화
# for i in range(3):
#     plt.scatter(iris_train_x.loc[iris_train_x['target']==i,:]['petal_len'],
#                 iris_train_x.loc[iris_train_x['target']==i,:]['petal_wid'])

# plt.savefig('iris.jpeg')

# KNN 모델 준비  (k=5 디폴트값 사용)
knnmodel = KNeighborsClassifier(n_neighbors=5)  # hyperparameter k = 5

# 모델 학습
knnmodel.fit(iris_train_x[['petal_len', 'petal_wid']], iris_train_x['target'])  # train_x의 'target' 컬럼이 정답데이터(train_y)

# 모델 성능 평가
print('acc : ',knnmodel.score(iris_train_x[['petal_len', 'petal_wid']].values, 
                     iris_train_x['target'].values))  # 정확도
print('='*80)

# 새로운 데이터 붓꽃 분류 (예측)
pred = knnmodel.predict([[5.9, 2.3], [3.4, 1.8], [5.4, 2.2]])  # 0 : setosa, 1 : versicolor, 2 : virginica
print([iris['target_names'][int(i)] for i in pred])
