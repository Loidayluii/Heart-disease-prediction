import pandas as pd 
# đọc dữ liệu 
df = pd.read_csv("data/heart.csv")
# in ra 5 dòng đầu tiên 
print(df.head())
# in ra kích thước
print(df.shape)
# in ra các cột
print(df.columns)
# in ra thông tin 
print(df.info())
#thống kê dữ liệu 
print(df.describe())
# đếm số người mắc bệnh 
print(df["target"].value_counts())