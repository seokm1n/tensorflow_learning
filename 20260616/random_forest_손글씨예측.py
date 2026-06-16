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

tempdata = [0.,  0., 10., 14.,  8.,  1., 0., 0., 
            0.,  2., 16., 14.,  6., 15., 6., 0.,
            0.,  0., 12., 15.,  8., 15., 0., 0., 
            0., 15.,  5., 16., 16., 10., 0., 0.,
            0.,  0., 12., 15., 13., 12., 0., 0.,
            0.,  4., 16.,  5.,  4., 16., 6., 0.,
            0.,  8., 16., 10.,  8., 16., 8., 1.,
            0.,  1.,  7., 12., 14., 12., 1., 0.]

temparr = np.array(tempdata)  # reshape 하기위해 numpy array로 변환
print(temparr)
temp_pred = RFmodel.predict([temparr])

print('temp_pred : ', temp_pred)

plt.imshow(temparr.reshape(8,8))  # 디폴트 색상 virdis
plt.show()