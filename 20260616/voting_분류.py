from sklearn import datasets
from sklearn import tree  # 의사결정 트리 모델 --> 불순도(엔트로피) 낮아지는 방향으로 트리를 성장시켜 분류
from sklearn.neighbors import KNeighborsClassifier  # KNN 분류 모델 --> k개의 최근접 이웃
from sklearn.svm import SVC  # SVM 모델 --> 결정경계를 활용한 분류
from sklearn.ensemble import VotingClassifier  # 보팅 분류
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mnist = datasets.load_digits()  # 0 ~ 9 로 이루어진 손글씨 데이터셋
features = mnist["data"]  # 8 * 8 = 총 64개 특징
labels = mnist["target"]
print(len(features))  # 1797
print(len(labels))  # 1797

train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.2)

# tree
model_tree = tree.DecisionTreeClassifier(criterion='gini', max_depth=8, max_features=32, random_state=46)
model_tree.fit(train_x, train_y)
print('tree acc : ', model_tree.score(test_x, test_y))

# KNN
model_knn = KNeighborsClassifier(n_neighbors=299)
model_knn.fit(train_x, train_y)
print('knn acc : ', model_knn.score(test_x, test_y))


# SVM
# probability = True : predict_proba() 함수 사용하기 위해 True 설정
model_SVC = SVC(C=0.1, gamma=0.003, probability=True, random_state=46)
model_SVC.fit(train_x, train_y)
svm_predicted = model_SVC.predict(test_x)
print('SVC acc : ', model_SVC.score(test_x, test_y))

# hard voting 정확도
hard_voting_model = VotingClassifier(estimators=[('decision_tree', model_tree),
                                                 ('knn', model_knn),
                                                 ('svm', model_SVC)], 
                                    weights=[1,1,1], 
                                    voting='hard')
hard_voting_model.fit(train_x, train_y)  # 하드보팅 모델 학습
print('hardvoting acc : ', hard_voting_model.score(test_x, test_y))

# soft voting 정확도 ==> 가중치를 적용한 투표 예측
soft_voting_model = VotingClassifier(estimators=[('decision_tree', model_tree),
                                                 ('knn', model_knn),
                                                 ('svm', model_SVC)], 
                                    weights=[1,1,1], 
                                    voting='soft')
soft_voting_model.fit(train_x, train_y)  # 소프트보팅 모델 학습
print('softvoting acc : ', soft_voting_model.score(test_x, test_y))
