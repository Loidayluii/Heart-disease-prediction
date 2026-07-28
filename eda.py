import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/heart.csv")
# biểu đồ phân bố nhãn Target , nếu 2 cột gần bằng nhau thì data_set cân bằng 
sns.countplot(x="target", data=df)

plt.title("Heart Disease Distribution")
plt.show()
#biểu đồ phân bố tuổi 
plt.figure(figsize=(8,5))

sns.histplot(df["age"], bins=20)

plt.title("Age Distribution")

plt.show()
# biểu đồ nam và nữ ( 0 = female , 1 = male )
sns.countplot(x="sex", data=df)

plt.title("Gender Distribution")

plt.show()

# biểu đồ tuổi theo bệnh tim 
plt.figure(figsize=(8,5))

sns.boxplot(x="target", y="age", data=df)

plt.show()

# Correlation Matrix ( Heatmap )
plt.figure(figsize=(12, 10))

sns.heatmap(df.corr(), annot=True, cmap="coolwarm")

plt.title("Correlation Matrix")

plt.show() 