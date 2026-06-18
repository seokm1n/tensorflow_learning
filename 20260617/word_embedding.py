import numpy as np
from tensorflow.keras.layers import Embedding  # type: ignore

input_data = np.array([[3,4,7], [9,2,3], [1,6,499]])

# input_dim : 변환할 입력값의 최대값 ==> maximum integer index + 1
# output_dim : 결과값을 몇개의 벡터로 생성할지 지정
# input_length: Length of input sequences
embedding = Embedding(input_dim=500, output_dim=16, input_length = 3)
output = embedding(input_data)
print(output)