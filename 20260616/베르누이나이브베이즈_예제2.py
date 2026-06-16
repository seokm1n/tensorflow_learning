import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

# 출력 옵션 제어
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)
pd.set_option("max_colwidth", 1000)

np.set_printoptions(
    precision=8, suppress=True
)  # 과학적 표기 대신 소수점 이하 8자리까지 표현
np.set_printoptions(threshold=np.inf)  # 무한으로 출력

spamdf = pd.read_csv("/home/sm/tf_env/20260616/spam.csv")
# print(spamdf)

spamdf_subset = spamdf[:].copy()
# print(spamdf_subset)


# Message 컬럼에 있는 email titil 정보를 전처리
def EmailMessageControl(arg):
    return re.sub(r"[^a-zA-Z\s]", "", arg)


spamdf_subset["Message"] = spamdf_subset["Message"].apply(EmailMessageControl)
spamdf_subset.info()
# print(spamdf_subset)

spamdf_subset["Category"] = spamdf_subset["Category"].map({"ham": 0, "spam": 1})

cv = CountVectorizer(binary=True)
train_x = cv.fit_transform(spamdf_subset["Message"])
train_x_encoded = train_x.toarray()
# print(train_x_encoded)
print(len(cv.get_feature_names_out()))

train_y = spamdf_subset["Category"]

bnb = BernoulliNB()
train_y = train_y.astype("int")

bnb.fit(train_x_encoded, train_y)

print(bnb.score(train_x_encoded, train_y))

# 새로운 이메일 데이터 추가 후 예측

temp_mail = cv.transform(
    [
        "Thanks for your subscription. Please confirm by replying YES or NO. If you reply NO you will not be charged",
        "Long time no see, how about eat lunch together?",
        "Congratulations you have won a free iphone claim your prize now",
        "Urgent your account has been selected for a cash reward click here to receive money",
    ]
)

pred = bnb.predict(temp_mail)
print(pred)

for idx, item in enumerate(pred):
    if item == 1:
        print(f"{idx + 1} 번째 메일은 스팸")
    else:
        print(f"{idx + 1} 번째 메일은 정상")
