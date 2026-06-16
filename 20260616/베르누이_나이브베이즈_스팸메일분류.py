import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

# 스팸메일분류를위한이메일제목과스팸레이블준비
email_list = [
    {'email title': 'free game only today', 'spam':True},
    {'email title': 'cheapest flight deal', 'spam':True},
    {'email title': 'limited time offer only today only today','spam':True},
    {'email title': 'today meeting schedule', 'spam':False},
    {'email title': 'your flight schedule attached', 'spam':False},
    {'email title': 'your credit card statement', 'spam':False}
]

email_df = pd.DataFrame(email_list)

# 분류를 위해 label 을 수치로 변환
email_df['spam'] = email_df['spam'].map({True:1, False:0})
print(email_df)

train_x = email_df['email title']
train_y = email_df['spam']

cv = CountVectorizer(binary=True)
train_x_cv = cv.fit_transform(train_x)
print(train_x_cv)
train_encoded = train_x_cv.toarray()
print(train_encoded)
print(cv.get_feature_names_out()) # ['attached' 'card' 'cheapest' 'credit' 'deal' 'flight' 'free' 'game'
 #'limited' 'meeting' 'offer' 'only' 'schedule' 'statement' 'time' 'today'
 #'your']

 # 모델 준비 ==> 베르누이 나이브베이즈 분류 ==>  특성데이터가 0, 1 로 이루어져야 함
bnb = BernoulliNB()
train_y = train_y.astype('int')

# 모델 학습
bnb.fit(train_encoded, train_y)

print('acc : ', bnb.score(train_encoded, train_y))


