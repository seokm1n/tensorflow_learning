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

def predict_single_clause(clause):
    clean_clause = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣\s]', '', clause)
    tokens = okt.morphs(clean_clause, stem=True)
    tokens = [word for word in tokens if word not in stopwords and word]

    if not tokens:
        return 0.5

    encoded = tokenizer.texts_to_sequences([tokens])
    padded = pad_sequences(encoded, maxlen=30)
    score = float(best_model.predict(padded, verbose=0)[0][0])
    return score


def new_review_predict(review_string):
    # 문장 부호나 연결어를 기준으로 나누고, 뒤에 있는 문맥에 더 큰 가중치 부여
    clauses = re.split(
        r'(?:[.!?]|(?:는데|지만|하지만|그런데|근데|다만|그래도|그치만))',
        review_string
    )
    clauses = [clause.strip() for clause in clauses if clause.strip()]

    if not clauses:
        clauses = [review_string]

    clause_scores = []
    weights = []

    for i, clause in enumerate(clauses):
        score = predict_single_clause(clause)
        clause_scores.append(score)
        weights.append((i + 1) / len(clauses))

    final_score = sum(s * w for s, w in zip(clause_scores, weights)) / sum(weights)

    print(f"문장 분절: {clauses}")
    print(f"분절별 점수: {clause_scores}")
    print(f"최종 점수: {final_score:.4f}")

    if final_score > 0.5:
        print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(final_score * 100))
    else:
        print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - final_score) * 100))


new_review_predict('초반엔 좀 졸렸는데 뒤로 갈수록 나아지더라 ㅋㅋ 배우들 연기력은 진짜 좋았음.')
new_review_predict('분위기랑 연출은 괜찮았는데 내용이 좀 산만해서 중간에 몇 번은 딴 생각함 ㅠㅠ')
new_review_predict('배우들은 열심히 한 거 같은데 영화 전체가 너무 느려서 시간 가는 줄 모르고 졸았어요 ^^;')
new_review_predict('기대 많이 했는데 생각보다 너무 별로였음. 돈 주고 보기 진짜 개짜증 났다 ㅋㅋ')
new_review_predict('재밌는 장면도 몇 개 있었고 음악도 좋았는데 끝나고 나니까 좀 아쉽다.. 그래도 볼만함!')
