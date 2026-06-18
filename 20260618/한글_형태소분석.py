import re
from konlpy.tag import Okt

okt = Okt() # KoNLPy 제공 한국어 형태소 분석기
wordlist = okt.morphs('파이썬을 활용한 한국어 형태소 분석 예제입니다.', norm=True, stem=True)
print(wordlist)
print("=" * 80)

wordlist = okt.pos('파이썬을 활용한 한국어 형태소 분석 예제입니다.', norm=True, stem=True)
print(wordlist) # 텍스틀 형태소 단위로 나눈 뒤, 형태소와 품사 정보를 튜플로 묶어 리스트 결과로 반환
print("=" * 80)

dictwordlist = dict(wordlist)
print(dictwordlist)
print("=" * 80)

nounlist = okt.nouns('파이썬을 활용한 한국어 형태소 분석 예제입니다.')
print(nounlist) # 텍스트에서 명사만 추출
print("=" * 80)

sentence1 = "이 영화는 최고 점수를 줘야해^^. 모든 영화가 다 재미있지는 않거든! 진짜로배우 casting, 스토리 모두 너무 좋은데 표현할 방법이 없구만 굿굿~~"
new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣\s]','',sentence1)
print(new_sentence)
print("=" * 80)

# 자연어처리에 불필요한 조사 위주로 불용어 목록 설정
stop_words = """이 로 가 다 가까스로 가령 각 각각 각자 각종 갖고 말하자면 같다 같이개의치않고 거니와 거바 거의 것 것과 해
것들 게다가 게우다 겨우 는 를 과 관하여 관한 할 그때 그래 그래도 그래서 그러나 그러니그러니까 그러면"""
stop_words = set(stop_words.split(' '))
#print(stop_words)
word_tokens = okt.morphs(new_sentence)

result = [word for word in word_tokens if not word in stop_words]

print('불용어 제거 전 :',word_tokens)
print('불용어 제거 후 :',result)
print("=" * 80)