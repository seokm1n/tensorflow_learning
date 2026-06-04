from sklearn.metrics import classification_report  # 정확도, 정밀도 출력
from sklearn.metrics import confusion_matrix  # 혼동행렬 생성

y_true = [0, 1, 1, 1, 0, 1, 1, 1, 2, 2, 3, 3, 2]
y_pred = [0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 3, 3, 1]

print(confusion_matrix(y_true, y_pred))  # 혼동행렬(배열) 생성해서 출력
print(classification_report(y_true, y_pred))

