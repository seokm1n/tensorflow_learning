import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

mnist = load_digits() # 0 ~ 9 손글씨 숫자 데이터 셋 ( 분류용 )

print(mnist['data'][-3:])
print(mnist['target'][-3:])

features = mnist['data'] # 1797 개의 8 * 8 이미지 데이터셋
labels = mnist['target']

RFmodel = RandomForestClassifier()
RFmodel.fit(features, labels)

print('score : ', RFmodel.score(features, labels))

predicted = RFmodel.predict(features[-5:])
print('labels : ', labels[-5:])
print('pre : ', predicted)

tempdata = [0.,  2., 10., 14.,  8., 11., 9., 3., 
            0.,  2., 10., 14.,  6., 15., 9., 3.,
            0.,  2.,  2.,  2.,  8., 15., 9., 3., 
            0.,  0.,  0.,  0.,  2., 11., 9., 0.,
            0.,  0.,  0.,  2., 13., 12., 0., 0.,
            0.,  0.,  0.,  8.,  9.,  5., 0., 0.,
            0.,  0.,  2., 10.,  8.,  0., 0., 0.,
            0.,  2.,  7., 12., 14.,  0., 0., 0.]

temparr = np.array(tempdata)  # reshape 하기위해 numpy array로 변환
print(temparr)
temp_pred = RFmodel.predict([temparr])

print('temp_pred : ', temp_pred)

plt.imshow(temparr.reshape(8,8), cmap='gray')  # 디폴트 색상 virdis
plt.show()