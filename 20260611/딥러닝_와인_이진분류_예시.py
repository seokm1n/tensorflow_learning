import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model  # type: ignore

# scaler 로드
scaler = joblib.load("/home/sm/tf_env/20260611/wine_scaler.pkl")

# 모델 로드
wine_best_model = load_model("/home/sm/tf_env/20260611/wine_best_model.keras")

# 예측할 사람들
new_x = pd.DataFrame(
    [
        [7.4, 0.70, 0.00, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4, 7],
        [6.8, 0.28, 0.40, 8.7, 0.046, 30, 133, 0.9956, 3.18, 0.47, 10.7, 7],
        [7.0, 0.45, 0.25, 3.2, 0.065, 18, 55, 0.9968, 3.35, 0.58, 10.2, 6]
    ]
)

new_x_scaled = scaler.fit_transform(new_x)

# 예측
pred = wine_best_model.predict(new_x_scaled)

# 결과 출력
for i, p in enumerate(pred):
    print(f"와인{i+1} 레드 확률 : {p[0]:.4f}")
    if p[0] >= 0.5:
        print("red")
    else:
        print("white")
