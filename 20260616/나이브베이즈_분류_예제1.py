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

iris_df = pd.DataFrame(dataset['data'] , columns=dataset['feature_names'])
print(iris_df)
iris_df['target'] = dataset['target']
print(iris_df)

# 타깃을 시각화 시키기 위해서 
# 0 ==> 'setosa' ,  1 ==> 'versicolor' ,  2 ==> 'virginica' 

iris_df['target'] = iris_df['target'].map({0:'setosa', 1:'versicolor', 2:'virginica'})
print(iris_df)

# setosa DF
setosa_df = iris_df.loc[iris_df['target'] == 'setosa'].copy()
# veriscolor DF
veriscolor_df = iris_df.loc[iris_df['target'] == 'versicolor'].copy()
# virginica DF
virginica_df = iris_df.loc[iris_df['target'] == 'virginica'].copy()

import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1,3, figsize = (15,7))

sns.histplot(data=setosa_df, x='sepal length (cm)', kde =True, ax=axes[0])
sns.histplot(data=veriscolor_df, x='sepal length (cm)', kde =True, ax=axes[1])
sns.histplot(data=virginica_df, x='sepal length (cm)', kde =True, ax=axes[2])
plt.savefig('iris_histplot.jpeg')