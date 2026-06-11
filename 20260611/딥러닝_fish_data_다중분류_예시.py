import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model  # type: ignore

# scaler 로드
scaler = joblib.load("/home/sm/tf_env/20260611/fish_scaler.pkl")

# 모델 로드
fish_best_model = load_model("/home/sm/tf_env/20260611/fish_multi_clf.keras")
fish_best_model.summary()


new_x = pd.DataFrame(
    [
        [200, 27, 30, 10, 4],
        [90, 18, 20, 7, 2],
        [25, 12, 13, 3, 1],
    ]
)

new_x_scaled = scaler.transform(new_x)

# 예측
pred = fish_best_model.predict(new_x_scaled)

fishclass = ["Bream", "Parkki", "Perch", "Pike", "Roach", "Smelt", "Whitefish"]

# 결과 출력
for i, p in enumerate(pred):
    print(f"물고기{i+1} 종류 : {fishclass[np.argmax(p)]}")
    print(f"확률 : {p[np.argmax(p)]:.4f}")
