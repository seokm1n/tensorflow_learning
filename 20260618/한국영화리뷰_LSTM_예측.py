import re
import pandas as pd
from konlpy.tag import Okt
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore

best_model = load_model("/home/sm/tf_env/20260618/movie_review_best_model.keras") # 앞서 저장한 모델 로드
okt = Okt() # KoNLPy 제공 형태소 분석기

# 조사 위주의 한국어 불용어 제거 리스트
stopwords = [
    '의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도',
    '를', '으로', '자', '에', '와', '한', '하다', '것', '수', '때',
    '점', '그', '저', '등', '만', '나', '나도', '있다', '없다',
    '같다', '더', '안', '되다', '돼다', '다', '하지만', '그래도',
    '그러나', '그런데', '아니', '왜', '뭐', '우리', '너', '너무',
    '매우', '정말', '진짜', '아주', '좀더', '이제', '여기', '저기'
]


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

def new_review_predict(review_string):
    new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣\s]','', review_string) # 한국어와 공백 이외의내용삭제
    new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
    new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어제거
    print(new_sentence) # ['영화', '굿', '잼']
    # [new_sentence] : 불용어 처리된 단어 리스트를 정수 인코딩 sequences 데이터 형성을 위해 하나로 묶어서([ ]) 변환해 줘야함
    encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
    print(encoded) # [[1, 363, 334]]
    sentence_padding = pad_sequences(encoded, maxlen = 30) # 패딩 적용 동일 길이 Sequences 형성
    print(sentence_padding)
    #[[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0 0 0 1 363 334]]
    score = float(best_model.predict(sentence_padding) ) # new_sentence 예측
    if(score > 0.5):
        print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(score * 100))
    else:
        print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - score) * 100))
    

new_review_predict('이 영화는 초반에는 조금 지루했지만 후반으로 갈수록 이야기가 흥미로워지고 배우들의 연기가 정말 좋았습니다.')
new_review_predict('영화의 분위기는 좋았지만 전반적으로 스토리가 너무 산만해서 끝까지 집중하기가 어려웠어요.')
new_review_predict('배우들이 열심히 연기한 점은 좋았는데 전체적인 전개가 너무 느려서 시간이 많이 지루하게 느껴졌습니다.')
new_review_predict('이 영화는 정말 기대를 많이 했는데 생각보다 너무 실망스러웠고 돈을 내고 보기 아까운 작품이었습니다.')
new_review_predict('재미있는 장면도 많았고 특히 음악과 촬영이 매우 인상적이어서 다시 보고 싶을 정도로 만족스러웠습니다.')
