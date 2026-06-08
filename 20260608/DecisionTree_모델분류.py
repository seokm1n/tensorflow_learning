import pandas as pd

district_dict_list = [
    {'district': 'Gangseo-gu', 'latitude': 37.551000, 'longitude': 126.849500, 'label': 'Gangseo'},
    {'district': 'Yangcheon-gu', 'latitude': 37.52424, 'longitude': 126.855396, 'label': 'Gangseo'},
    {'district': 'Guro-gu', 'latitude': 37.4954, 'longitude': 126.8874, 'label': 'Gangseo'},
    {'district': 'Geumcheon-gu', 'latitude': 37.4519, 'longitude': 126.9020, 'label': 'Gangseo'},
    {'district': 'Mapo-gu', 'latitude': 37.560229, 'longitude': 126.908728, 'label': 'Gangseo'},

    {'district': 'Gwanak-gu', 'latitude': 37.487517, 'longitude': 126.915065, 'label': 'Gangnam'},
    {'district': 'Dongjak-gu', 'latitude': 37.5124, 'longitude': 126.9393, 'label': 'Gangnam'},
    {'district': 'Seocho-gu', 'latitude': 37.4837, 'longitude': 127.0324, 'label': 'Gangnam'},
    {'district': 'Gangnam-gu', 'latitude': 37.5172, 'longitude': 127.0473, 'label': 'Gangnam'},
    {'district': 'Songpa-gu', 'latitude': 37.503510, 'longitude': 127.117898, 'label': 'Gangnam'},

    {'district': 'Yongsan-gu', 'latitude': 37.532561, 'longitude': 127.008605, 'label': 'Gangbuk'},
    {'district': 'Jongro-gu', 'latitude': 37.5730, 'longitude': 126.9794, 'label': 'Gangbuk'},
    {'district': 'Seongbuk-gu', 'latitude': 37.603979, 'longitude': 127.056344, 'label': 'Gangbuk'},
    {'district': 'Nowon-gu', 'latitude': 37.6542, 'longitude': 127.0568, 'label': 'Gangbuk'},
    {'district': 'Dobong-gu', 'latitude': 37.6688, 'longitude': 127.0471, 'label': 'Gangbuk'},

    {'district': 'Seongdong-gu', 'latitude': 37.557340, 'longitude': 127.041667, 'label': 'Gangdong'},
    {'district': 'Dongdaemun-gu', 'latitude': 37.575759, 'longitude': 127.025288, 'label': 'Gangdong'},
    {'district': 'Gwangjin-gu', 'latitude': 37.557562, 'longitude': 127.083467, 'label': 'Gangdong'},
    {'district': 'Gangdong-gu', 'latitude': 37.554194, 'longitude': 127.151405, 'label': 'Gangdong'},
    {'district': 'Jungrang-gu', 'latitude': 37.593684, 'longitude': 127.090384, 'label': 'Gangdong'}
]

train_df = pd.DataFrame(district_dict_list)
#print(train_df)

# 컬럼 데이터의 순서변경
train_df = train_df[['district','longitude','latitude','label']]
print(train_df)
print("=" * 80)

dong_dict_list = [
    {'dong': 'Gaebong-dong', 'latitude': 37.489853, 'longitude': 126.854547, 'label': 'Gangseo'},
    {'dong': 'Gochuk-dong', 'latitude': 37.501394, 'longitude': 126.859245, 'label': 'Gangseo'},
    {'dong': 'Hwagok-dong', 'latitude': 37.537759, 'longitude': 126.847951, 'label': 'Gangseo'},
    {'dong': 'Banghwa-dong', 'latitude': 37.575817, 'longitude': 126.815719, 'label': 'Gangseo'},
    {'dong': 'Sangam-dong', 'latitude': 37.577039, 'longitude': 126.891620, 'label': 'Gangseo'},

    {'dong': 'Nonhyun-dong', 'latitude': 37.508838, 'longitude': 127.030720, 'label': 'Gangnam'},
    {'dong': 'Daechi-dong', 'latitude': 37.501163, 'longitude': 127.057193, 'label': 'Gangnam'},
    {'dong': 'Seocho-dong', 'latitude': 37.486401, 'longitude': 127.018281, 'label': 'Gangnam'},
    {'dong': 'Bangbae-dong', 'latitude': 37.483279, 'longitude': 126.988194, 'label': 'Gangnam'},
    {'dong': 'Dogok-dong', 'latitude': 37.492896, 'longitude': 127.043159, 'label': 'Gangnam'},

    {'dong': 'Pyoungchang-dong', 'latitude': 37.612129, 'longitude': 126.975724, 'label': 'Gangbuk'},
    {'dong': 'Sungbuk-dong', 'latitude': 37.597916, 'longitude': 126.998067, 'label': 'Gangbuk'},
    {'dong': 'Ssangmoon-dong', 'latitude': 37.648094, 'longitude': 127.030421, 'label': 'Gangbuk'},
    {'dong': 'Ui-dong', 'latitude': 37.648446, 'longitude': 127.011396, 'label': 'Gangbuk'},
    {'dong': 'Samcheong-dong', 'latitude': 37.591109, 'longitude': 126.980488, 'label': 'Gangbuk'},

    {'dong': 'Hwayang-dong', 'latitude': 37.544234, 'longitude': 127.071648, 'label': 'Gangdong'},
    {'dong': 'Gui-dong', 'latitude': 37.543757, 'longitude': 127.086803, 'label': 'Gangdong'},
    {'dong': 'Neung-dong', 'latitude': 37.553102, 'longitude': 127.080248, 'label': 'Gangdong'},
    {'dong': 'Amsa-dong', 'latitude': 37.552370, 'longitude': 127.127124, 'label': 'Gangdong'},
    {'dong': 'Chunho-dong', 'latitude': 37.547436, 'longitude': 127.137382, 'label': 'Gangdong'}
]

test_df = pd.DataFrame(dong_dict_list)
test_df = test_df[['dong', 'longitude', 'latitude', 'label']]  # 테스트 데이터 준비

print(train_df['label'].value_counts())
print(test_df['label'].value_counts())
print("=" * 80)

train_df.drop(['district'], axis=1, inplace=True)
test_df.drop(['dong'], axis=1, inplace=True)

train_x = train_df[['longitude', 'latitude']]
train_y = train_df[['label']]  # 정답 데이터

test_x = test_df[['longitude', 'latitude']]
test_y = test_df[['label']]

# 의사결정 트리 모델
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

le = preprocessing.LabelEncoder()  # 특정 문자열(라벨)을 수치데이터로 치환

# 특정 데이터를 어떤 방법으로 변환할지 먼저 학습(fit)하고
# 학습이 완료되었으면 수치데이터로 변환
y_encoded = le.fit_transform(train_y.values.ravel())
print(y_encoded)
print("=" * 80)
# 알파벳 순으로 다른 클래스 자동 분류
print(le.classes_)  # 어떤것을 0으로, 어떤것을 1로, ... 등 변환했는지 체크

# 모델 설계 (의사결정 트리 모델)
dt_model = tree.DecisionTreeClassifier(criterion='entropy', random_state=70, max_depth=4,
                                       min_samples_leaf=2, min_samples_split=2)

# 모델 학습
clf = dt_model.fit(train_x.values, y_encoded)


# 창천동 위도, 경도
print("=" * 80)
pred = clf.predict([[126.9368, 37.5568]])
print('pred : ', pred)
print(le.classes_[pred[0]])
print("=" * 80)


def display_decision_surface(clf, x, y):
    x_min = x['longitude'].min() - 0.01
    x_max = x['longitude'].max() + 0.01
    y_min = x['latitude'].min() - 0.01
    y_max = x['latitude'].max() + 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.001), 
                         np.arange(y_min, y_max, 0.001))
    np.set_printoptions(threshold=np.inf)
    Z = clf.predict(np.column_stack([xx.ravel(), yy.ravel()]))
    print(Z.shape)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
    
    n_classes = len(le.classes_)  # 4
    plot_colors = 'rywb'

    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        print(idx)
        plt.scatter(x.loc[idx, 'longitude'], x.loc[idx, 'latitude'],
                    label = le.classes_[i], c=color, edgecolors='black', s=150)

    # 창천동
    plt.scatter(126.9368, 37.5568, c="black", marker="v", s=200)

    plt.title('Decision surface of a decision tree', fontsize=16)
    plt.legend(loc='best', fontsize=14)
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.show()


display_decision_surface(clf, train_x, y_encoded)

# from sklearn.tree import plot_tree

# plt.figure(figsize=(8, 8))

# plot_tree(clf, filled=True, feature_names=['longitude', 'latitude'], 
#           class_names=['Gangbuk', 'Gangdong', 'Gangnam', 'Gangseo'])

# plt.show()
