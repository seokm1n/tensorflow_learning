import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

mnist = load_digits() # 0 ~ 9 손글씨 숫자 데이터 셋 ( 분류용 )

print(mnist['data'][:3])
print(len(mnist['data']))
print(mnist['target'])
print(len(mnist['target']))

features = mnist['data'] # 1797 개의 8 * 8 이미지 데이터셋
labels = mnist['target']

RFmodel = RandomForestClassifier()
RF_scores = cross_validate(RFmodel, features, labels, cv=10)  # 랜덤포레스트모델 10-fold 교차 검증
print(RF_scores['test_score'])  # 랜덤 포레스트 앙상블 검증평가 점수

DT_scores = cross_validate(tree.DecisionTreeClassifier(), features, labels, cv=10)  # 의사결정트리 10-fold 교차 검증
print(DT_scores['test_score'])

print('random_forest accuracy : ', np.mean(RF_scores['test_score']))
print('decision tree accuracy : ', np.mean(DT_scores['test_score']))
# 랜덤 포레스트 앙상블이 별도의 하이퍼 파라미터 설정 없는 의사결정트리보다 월등히 높은 성능 발휘

df = pd.DataFrame({'random_forest': RF_scores['test_score'], 
                   'decision_tree': DT_scores['test_score']})
print(df)

df.plot()
plt.show()