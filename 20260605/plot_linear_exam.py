import numpy as np
import matplotlib.pyplot as plt
import matplotlib

print(matplotlib.get_backend())

# [5, 5]와 [50, 50] 위치를 직선으로 연결해서 표시
# x축 인덱스 ==> [10, 20]
# y축 인덱스 ==> [5, 20]
plt.plot([10, 20, 30], [5, 20, 60])  # 직선 그래프
plt.show()
