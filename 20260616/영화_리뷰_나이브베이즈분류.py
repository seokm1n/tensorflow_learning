import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# 출력 옵션 제어
# pd.set_option("display.max_rows", 1000)
# pd.set_option("display.max_columns", 500)
# pd.set_option("display.width", 1000)
# pd.set_option("max_colwidth", 1000)

# np.set_printoptions(precision=8, suppress=True)  # 과학적 표기 대신 소수점 이하 8자리까지 표현
np.set_printoptions(threshold=np.inf)  # 무한으로 출력

movie_df = pd.read_csv("/home/sm/tf_env/20260616/IMDB Dataset.csv")
print(movie_df[:5])
print("=" * 80)

# movie_df = movie_df[:10000].copy()


# 정보 전처리
def MovieReviewControl(arg):
    return re.sub(r"[^a-zA-Z\s]", "", arg)


movie_df["review"] = movie_df["review"].apply(MovieReviewControl)
movie_df.info()

# 'sentiment' 컬럼 라벨을 수치데이터로 변경
movie_df["sentiment"] = movie_df["sentiment"].map({"positive": 1, "negative": 0})

train_x = movie_df["review"]
train_y = movie_df["sentiment"]

cv = CountVectorizer()
train_cv = cv.fit_transform(train_x)
# train_cv_encoded = train_cv.toarray()
# print(train_cv_encoded[0])

mnb = MultinomialNB()
mnb.fit(train_cv, train_y)

print("acc : ", mnb.score(train_cv, train_y))

# 새로운 영화 리뷰 데이터 입력해서 예측?

temp_review = cv.transform(
    [
        "This movie is awesome",
        "This movie is terrible",
        "I really enjoyed this film",
        "Worst movie ever",
    ]
)

pred = mnb.predict(temp_review)
prob = mnb.predict_proba(temp_review)
print(pred)
print(prob)

for idx, item in enumerate(pred):
    if item == 1:
        print(f"{idx + 1} 번째 리뷰는 긍정 ", end = '')
        print(f"(확률 : {prob[idx][1]*100:.2f}%)")
        
    else:
        print(f"{idx + 1} 번째 리뷰는 부정 ", end = '')
        print(f"(확률 : {prob[idx][0]*100:.2f}%)")
