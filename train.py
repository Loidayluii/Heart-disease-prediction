import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("data/heart.csv")
# x là input  
# y là output 
X = df.drop("target", axis=1)
y = df["target"]
'''
print(X.head())

print()

print(y.head())
'''

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2, # xử lý overfitting bằng validation 
    #( học 20 %  xấp xỉ 1025*20% = 205 kiểm tra còn 820 để học )
    random_state=0 #CROSS VALIDATION
)
print(X_train.shape)
print(X_test.shape)

print(y_train.shape)
print(y_test.shape)