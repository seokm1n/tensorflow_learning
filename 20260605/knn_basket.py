import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # sklearn 선형회귀 모델

df = pd.read_csv('basketball_stat.csv')
print(df.head())
print(df.info())
print(df["Pos"].value_counts())
print("=" * 80)
