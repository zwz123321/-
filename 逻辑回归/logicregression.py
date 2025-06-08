import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

#读取数据
dataset = pd.read_csv('D:/python代码调试/项目实战/逻辑回归/data/Breast Cancer Wisconsin (Diagnostic) Data Set.csv')
# print(dataset)

#特征提取
x = dataset.iloc[:,1:-1]
# print(x)

#标签提取
Y = dataset['diagnosis']
# print(Y)

#划分数据集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, Y, test_size = 0.2)

#进行数据的归一化
sc = MinMaxScaler(feature_range=(0,1))
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
#print(x_train)

#逻辑回归模型
lr = LogisticRegression()
lr.fit(x_train, y_train)

#打印模型的参数
# print('w:', lr.coef_)
# print('b:', lr.intercept_)

#利用模型进行测试
pre_result = lr.predict(x_test)
# print(pre_result)

#预测结果的概率
pre_result_proba = lr.predict_proba(x_test)

#获取恶性肿瘤的概率
pre_list = pre_result_proba[:, 1]
# print(pre_list)

#设置分类阈值
thresholds = 0.3

#设置保存结果的列表
result = []
result_name = []

for i in range(len(pre_list)):
    if pre_list[i] > thresholds:
        result.append(1)
        result_name.append('恶性')
    else:
        result.append(0)
        result_name.append('良性') 
        
# 打印阈值调整后的结果
# print(result)
# print(result_name)  

#输出结果的精确率和赵辉率和F1值
report = classification_report(y_test, result, labels=['1','0'], target_names=(['良性肿瘤','恶性肿瘤']))
print(report)