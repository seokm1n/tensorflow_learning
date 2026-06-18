import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam as adam  # type: ignore
from konlpy.tag import Okt
from tqdm import tqdm  # 진행바 출력

# Tokenizer ==> 특정 단어를 특정 수치(정수)로 매핑 치환하는 역할
from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore

# pad_sequence ==> 고정 길이 정수 벡터를 생성할 때 사용
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
from tensorflow.keras.layers import Embedding, Dense, LSTM, Input  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore

# 출력 옵션 제어
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)
pd.set_option("max_colwidth", 1000)

train_df = pd.read_csv(
    "/home/sm/tf_env/20260618/train_stopwords_reviews.csv", index_col=0
)
test_df = pd.read_csv(
    "/home/sm/tf_env/20260618/test_stopwords_reviews.csv", index_col=0
)

train_df.dropna(how="any", inplace=True)
test_df.dropna(how="any", inplace=True)

train_df.info()
test_df.info()
print("=" * 80)

word_size = 11775  # 단어 빈도수 체크 결과에 따른 11775개 단어 집합 사용
# ==> IMDB의 num_words의 역할

tokenizer = Tokenizer(word_size)
tokenizer.fit_on_texts(train_df["document"])
# print(tokenizer.word_index)

# for word, index in tokenizer.word_index.items():
#     if index == 2:
#         print(word)

# texts_to_sequences ==> tokenizer, word_index를 활용해서 리뷰데이터를 정수데이터로 치환
train_df["sequence"] = tokenizer.texts_to_sequences(train_df["document"])
test_df["sequence"] = tokenizer.texts_to_sequences(test_df["document"])

# print(train_df.head())
# print(test_df.head())

train_df.reset_index(drop=True, inplace=True)  # 인덱스 초기화
test_df.reset_index(drop=True, inplace=True)  # 인덱스 초기화

print(train_df[25:30])
print(test_df[57:62])

# 11775 개 단어 집합만 고려 했음으로 빈도수가 1 이하인 단어로 이루어진 문장은 텅빈( [ ] )
# 형태로 변환 됨, 따라서 해당 문장의 인덱스를 찾아 제거 해줌
drop_train_idx = [
    idx for idx, sentence in enumerate(train_df["sequence"]) if len(sentence) < 1
]
print("drop_train_idx : \n", drop_train_idx)

drop_test_idx = [
    idx for idx, sentence in enumerate(test_df["sequence"]) if len(sentence) < 1
]
print("drop_test_idx : \n", drop_test_idx)

# 텅빈([ ]) sequence 데이터 위치 인덱스 활용해서  Dataframe 해당 행 삭제
train_df.drop(drop_train_idx, axis=0, inplace=True)
test_df.drop(drop_test_idx, axis=0, inplace=True)

train_df.reset_index(drop=True, inplace=True)  # 인덱스 초기화
test_df.reset_index(drop=True, inplace=True)  # 인덱스 초기화

print("========= 삭제 완료 검증 수행 ===========")
for idx, sequence in enumerate(train_df["sequence"]):
    if len(sequence) < 1:
        print(idx, sequence)

print(train_df[25:30])
print(test_df[57:62])

# 타깃 라벨 추출
y_train = np.array(train_df["label"])
y_test = np.array(test_df["label"])

print(len(train_df["sequence"]))  # 최종 훈련데이터 31901 개 샘플
print(len(y_train))  # 최종 훈련데이터 라벨 31901 개
print(len(test_df["sequence"]))  # 최종 테스트데이터 31554 개 샘플
print(len(y_test))  # 최종 테스트데이터 라벨 31554 개

train_review_sequence_len = [len(sequence) for sequence in train_df["sequence"]]
train_review_sequence_arr = np.array(train_review_sequence_len)
print("max : ", np.max(train_review_sequence_arr))  # 훈련 리뷰데이터 최대 길이 63
print("mean : ", np.mean(train_review_sequence_arr))  # 평균 길이 10.734114918027648
#
# # import matplotlib.pyplot as plt
# # plt.hist(train_review_sequence_len, bins=50)
# # plt.show() # pad 적용 30 길이로 동일하게 맞추자
#
X_train_pades = pad_sequences(train_df["sequence"], maxlen=30)
X_test_pades = pad_sequences(test_df["sequence"], maxlen=30)

print(len(X_train_pades[0]))
print(X_train_pades[:1])
print(len(X_test_pades[0]))
print(X_test_pades[:1])

# 최종 LSTM 모델 훈련 데이터 준비 완료
# 훈련데이터 ( X_train_pades , y_train )
# 테스트데이터 ( X_test_pades, y_test )

# LSTM 모델 설계
embedding_dim = 100  # embedding 밀집벡터 차원
hidden_units = 128  # LSTM 뉴런수

model = Sequential()
model.add(Input(shape=(30,)))
model.add(Embedding(word_size, embedding_dim))  # 11775, 100
model.add(LSTM(hidden_units))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer=adam(learning_rate=1e-5), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# 모델 Callbacks 지정
EarlyStopCB = EarlyStopping(
    monitor="val_loss", verbose=1, patience=5, restore_best_weights=True
)
ModelCheckCB = ModelCheckpoint(
    "/home/sm/tf_env/20260618/movie_review_best_model.keras",
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
)

# 모델 훈련
history = model.fit(
    X_train_pades,
    y_train,
    validation_data=(X_test_pades, y_test),
    epochs=30,
    callbacks=[EarlyStopCB, ModelCheckCB],
    batch_size=32,
)
