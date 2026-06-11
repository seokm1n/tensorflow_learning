import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model  # type: ignore

# scaler 로드
scaler = joblib.load("/home/sm/tf_env/20260611/titanic_scaler.pkl")

# 모델 로드
titanic_bestmodel = load_model("/home/sm/tf_env/20260611/titanic_bestmodel.keras")

# 예측할 사람들
new_x = pd.DataFrame([[0, 26, 0, 1], [1, 24, 0, 1], [0, 52, 0, 1], [1, 52, 0, 1]])

new_x_scaled = scaler.fit_transform(new_x)

# 예측
pred = titanic_bestmodel.predict(new_x_scaled)

# 결과 출력
for i, p in enumerate(pred):
    print(f"사람{i + 1} 생존확률 : {p[0]}")
