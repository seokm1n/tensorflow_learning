import re
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.datasets import imdb  # 정수로 변환된 영화리뷰 db  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore

(train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=10000)

test_seq = pad_sequences(test_x, maxlen=100)
print(test_seq.shape)  # (25000, 100) # test 데이셋도 길이 100으로 동일 panding
print(test_seq[0])

# 단어 집합 개수(500), 문장길이 패딩(100) 으로 설정되어 정확도가 다소 떨어짐
model = load_model("/home/sm/tf_env/20260617/best-lstm-model.keras")  # 앞에서 저장한 모델 복원
# test_seq 데이터셋으로 정확도 측정
print("정확도 : %.4f " % model.evaluate(test_seq, test_y)[1])  # 정확도 : 0.7685

print(model.predict(test_seq[0:1]))  # [[0.42196107]] 부정으로 예측
print(test_y[0])  # 정답 : 0 ( 부정 )

# word_to_index ==> {단어:정수, 단어:정수,..} , 정수 1부터 매핑
word_to_index = imdb.get_word_index()  # <=== imdb 인덱스 매핑 사전 반환
for key, value in word_to_index.items():
    if value == 1:
        print("key , value :", key, value)  # key , value : the 1
print(word_to_index["this"])  # 11로 정수매핑 되있지만 --> 정수 토큰화된 내용은 14

p_review = "I really enjoyed this movie from start to finish. The acting was convincing, the story was engaging, and the characters felt realistic. There were a few predictable moments, but overall it was entertaining and worth watching. I would definitely recommend it to anyone looking for a good film."
n_review = "This movie was a disappointment. The plot was confusing, the pacing was slow, and I never felt connected to any of the characters. Several scenes seemed unnecessary and the ending was unsatisfying. I would not recommend spending time on this film."
t_review = "The movie had some interesting ideas and a few strong performances, but it also had several problems. Parts of the story were engaging while others felt dragged out. I didn't hate it, but I wasn't particularly impressed either. It was an average experience overall."


def new_sentence_tokenization(
    sentence_arg,
):  # 임의의 문장을 정수 데이터로 인코딩(토큰화)
    # 정규화를 이용한 문장 정리
    # 숫자, 알파벳,공백문자를 제외한 모든 문자를 ''로 치환(즉 제거) , 이후 소문자화
    new_sentence = re.sub("[^0-9a-zA-Z\s]", "", sentence_arg).lower()
    # 정수 인코딩
    encoded = []
    for word in new_sentence.split():  # 단어 집합 크기를 훈련데이터 와 동일하게 500으로 제한
        try:
            if word_to_index[word] <= 10000:
                encoded.append(
                    word_to_index[word] + 3
                )  # 예) 'the'의 value값 1 에 3을 더해 4를 저장
            else:
                encoded.append(2)  # 500 이상의 숫자는 <untoken> 알수없는 토큰으로 취급
        # 단어 집합에 없는 단어, 즉 word_to_index 단어 사전에 word 키 값이 없는 경우
        # <untoken> 알수없는 토큰으로 취급
        except KeyError:
            encoded.append(2)

    # pad_sequences() : 길이를 맞춰주는 패딩 진행 ( 잘라내기 또는 0 으로 채워짐 )
    pad_new = pad_sequences([encoded], maxlen=100)  # 타임 스템프 형성을 위한 2차원 배열 형태전달# 훈련,테스트 데이터와 동일하게 길이를 100으로 패딩(타임 스템프 크기)
    # 예측
    # print(pad_new)
    score = float(model.predict(pad_new, verbose=0)[0][0])
    print("score : ", score)
    if score > 0.5:
        print("{:.2f}% 확률로 긍정 리뷰".format(score * 100))
    else:
        print("{:.2f}% 확률로 부정 리뷰".format((1 - score) * 100))

print("1번째 긍정 리뷰")
new_sentence_tokenization(p_review)
print("2번째 부정 리뷰")
new_sentence_tokenization(n_review)
print("3번째 테스트 리뷰")
new_sentence_tokenization(t_review)
