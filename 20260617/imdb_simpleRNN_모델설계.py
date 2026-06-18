import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import imdb  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Input  # type: ignore
from tensorflow.keras.optimizers import RMSprop  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore

# num_words=500 ==> 500개 단어만 사용, 없는 단어는 2로 치환
(train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=10000)  # 안쓰면 _로 표현
# 내부적으로 0,1,2,3은 특수토큰 ==> index + 3
# print(len(train_x))
# print(train_x[0])  # 정수 벡터화 되어있음
# print(np.unique(train_y[0]), return_counts=True)  # 라벨도 수치화 되어있음
# 타깃 이진 분류 : 0(부정), 1(긍정)

# word_index = imdb.get_word_index()  # 단어별 정수 매핑된 데이터셋 확인(사전형태)
# print(word_index)

# for word, idx in word_index.items():
#     if idx == 1:  # imdb 데이터셋 중 가장 빈도수가 높은 단어
#         print(word, idx)  # the 1

# train_x[0] ==> 정수벡터를 다시 단어로 치환시키는 작업
# conv_word_index = dict([(idx + 3, word) for (word, idx) in word_index.items()])
# print(conv_word_index)

# for idx, word in conv_word_index.items():
#     if idx == 4:
#         print(idx, word)  # 4 the

# decoded_sentence = " ".join(
#     [conv_word_index[i] if i in conv_word_index else "?" for i in train_x[0]]
# )
# print(decoded_sentence)

train_x, val_x, train_y, val_y = train_test_split(
    train_x, train_y, test_size=0.2, random_state=43
)
# print(len(train_x))  # 20000
# print(len(val_x))  # 5000

# 1차 ==> 길이가 다른 리뷰 정수 데이터 배열을 길이가 동일한 배열로 변경
# 길이를 100으로 모두 변경할 때 짧은건 0으로 채우고 긴건 버림
train_seq = pad_sequences(train_x, maxlen=100)  # 훈련셋 전처리
val_seq = pad_sequences(val_x, maxlen=100)
# print(train_seq.shape) # (20000, 100) val_y = pad_sequences(val_x, maxlen=100) # 검증셋 전처리
# print(val_y.shape) # (5000, 100)
# print(val_y[0])

# 원핫인코딩으로 차원 추가 ==> 배열이 너무 커짐
# train_oh = to_categorical(train_seq)
# val_oh = to_categorical(val_seq)

model = Sequential()
model.add(Input(shape=(100,)))
model.add(Embedding(10000, 16, input_length=100))  # 길이 100인 정수를 16차원 밀집벡터로
model.add(SimpleRNN(32))  # 뉴런 개수 8개
# 출력층 추가 ==> 이진분류, 뉴런 1개 사용, 시그모이드활성화 함수
model.add(Dense(1, activation="sigmoid"))
model.summary()

# 모델 설정(컴파일)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)  # 0.00001
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# 최상의 검증 점수를 낸 모델 저장
checkpoint_cb = ModelCheckpoint("/home/sm/tf_env/20260617/best-simplernn-model.keras", save_best_only=True, verbose=1)
# 3번 이상 검증 손실이 감소하지 않으면 조기 종료
earlystopping_cb = EarlyStopping(patience=5, restore_best_weights=True)
# 모델 훈련 : train_seq, val_seq 사용
history = model.fit(
    train_seq,
    train_y,
    epochs=100,
    batch_size=64,
    validation_data=(val_seq, val_y),
    callbacks=[checkpoint_cb, earlystopping_cb]
)
