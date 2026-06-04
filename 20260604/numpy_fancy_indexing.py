import numpy as np
import matplotlib.pyplot as plt

# range()
arr = np.arange(5, 19).reshape(7,2)
print(arr)
# fancy indexing ==> 추출할 위치의 인덱스를 배열 형태로 전달해서 추출

xdata = arr[[1,3,5], 0]
# xdata = arr[1::2, 0]
ydata = arr[[1,3,5], 1]

# print(xdata)  # [7 11 15]
# print(ydata)  # [8 12 16]

plt.scatter(xdata, ydata)
plt.xlabel("xdata")
plt.ylabel("ydata")
plt.savefig("fancy_indexing.jpeg")