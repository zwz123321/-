import pandas as pd

# 读取数据
train_data = pd.read_csv('加州房价预测/data/train.csv')

total = train_data.isnull().sum().sort_values(ascending=False)
percent =(train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))