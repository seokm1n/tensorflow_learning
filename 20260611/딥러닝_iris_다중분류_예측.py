import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model  # type: ignore

# scaler 로드
scaler = joblib.load("/home/sm/tf_env/20260611/iris_scaler.pkl")

# 모델 로드
iris_best_model = load_model("/home/sm/tf_env/20260611/iris_multi_clf.keras")

# 예측할 사람들
new_x = pd.DataFrame(
    [
        [1,2,3,4],
        [5,6,7,8]
    ]
)

new_x_scaled = scaler.transform(new_x)

# 예측
pred = iris_best_model.predict(new_x_scaled)

irisclass = np.array(['setosa', 'versicolor', 'virginica'])

# 결과 출력
print(irisclass[np.argmax(pred, axis=1)])