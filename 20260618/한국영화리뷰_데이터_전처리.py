import re
import numpy as np
import pandas as pd
from konlpy.tag import Okt
from tqdm import tqdm # 진행바 출력

# 출력 옵션 제어
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)
pd.set_option("max_colwidth", 1000)

review_df = pd.read_csv(
    "/home/sm/tf_env/20260618/ratings_train.csv", header=0, delimiter="\t", quoting=3
)
review_df.dropna(how="any", inplace=True)

# 라벨(타겟) 컬럼데이터의 타입 실수 -> 정수
review_df["label"] = review_df["label"].astype("int64")

# 리뷰 데이터 항목 중 중복데이터 체크
print(review_df["document"].nunique())  # nunique() ==> 유니크한 정보 개수 반환
review_df.drop_duplicates(subset=["document"], inplace=True)


review_df.info()
print(review_df.head())
print("=" * 80)
# 결측치, 중복 제거된 데이터의 개수 ==> 32163개


def review_filtering(arg):
    return re.sub(r"[^ㄱ-힣\s]", "", arg)


# 한글과 공백을 제외한 모든 문자를 제거
review_df["document"] = review_df["document"].apply(review_filtering)


def multi_space_filtering(arg):
    return re.sub(r"^\s+", "", arg)


def null_replace(arg):
    return np.nan if pd.isna(arg) or str(arg).strip() == "" else arg


# 한글과 공백을 제외한 모든 문자를 제거
review_df["document"] = review_df["document"].apply(multi_space_filtering)
review_df["document"] = review_df["document"].apply(null_replace)
review_df.dropna(how='any',inplace=True)
print(review_df.sample(100))
review_df.info()

okt = Okt() # KoNLPy 제공 형태소 분석기
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

# stem = True : 어간 추출 수행, 예) '이런' => '이렇다'로 변환
X_train = []
for sentence in tqdm(review_df['document']):
    tokenized_sentence = okt.morphs(sentence, stem=True) # 각 문장을 토큰화
    sentence_removed_stopwords = \
    [word for word in tokenized_sentence if not word in stopwords] # 불용어제거#불용어 제거된 단어 리스트를 한 문장으로 합친 다음 X_train list 에 추가
    X_train.append(' '.join(sentence_removed_stopwords))
# print(X_train[:5]) # 불용어가 제거된 문장 모음 리스트
review_df['document'] = X_train
