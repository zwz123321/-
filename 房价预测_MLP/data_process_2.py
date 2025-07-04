import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取数据
train_data = pd.read_csv('加州房价预测/data/train.csv')
test_data = pd.read_csv('加州房价预测/data/test.csv')

batch_size = 1024
# 数据预处理
all_features = pd.concat((train_data.iloc[:,4:],test_data.iloc[:,4:]))
all_features = all_features[['Bathrooms','Full bathrooms','Elementary School Score','High School Score','Listed Price','Tax assessed value','Type','Listed On','Bedrooms','Heating','Parking','City','Region','Cooling','Appliances included']]

# 处理 Bedrooms 列，计算数量
def convert_bedrooms(bedrooms):
    # 尝试转换为数字
    try:
    # 如果是纯数字字符串，返回它的浮点数形式
        return float(bedrooms)
    except ValueError:
        # 如果不是数字字符串，尝试分割字符串并计算总和
        values = bedrooms.split(',')  # 用,分开
        return len(values)
# 应用转换函数
all_features['Bedrooms'] = all_features['Bedrooms'].apply(convert_bedrooms)
all_features['Bedrooms'] = all_features['Bedrooms'].fillna(2.0)

all_features['Bathrooms'] = all_features['Bathrooms'].fillna(all_features['Full bathrooms'])
all_features['Bathrooms'] = all_features['Bathrooms'].fillna(all_features['Bathrooms'].mode().values[0])

all_features['Full bathrooms'] = all_features['Full bathrooms'].fillna(all_features['Bathrooms'])
all_features['Full bathrooms'] = all_features['Full bathrooms'].fillna(all_features['Full bathrooms'].mode().values[0])

all_features['Elementary School Score'] = all_features['Elementary School Score'].fillna(all_features['Elementary School Score'].mode().values[0])
all_features['High School Score'] = all_features['High School Score'].fillna(all_features['High School Score'].mode().values[0])
all_features['Type'] = all_features['Type'].fillna(all_features['Type'].mode().values[0])
all_features['Region'] = all_features['Region'].fillna(all_features['Region'].mode().values[0])
all_features['Tax assessed value'] = all_features['Tax assessed value'].fillna(all_features['Listed Price'] * 0.66)


def convert_parkings(parkings):
    if parkings == '0 spaces' :
        return 0
    elif isinstance(parkings, float):
        return parkings
    else:
        return 1

all_features['Parking'] = all_features['Parking'].apply(convert_parkings)

def convert_heating(heating):
    if heating == 'None' :
        return 1
    elif isinstance(heating, float):
        return heating
    else:
        values = heating.split(',')
        return len(values)

def convert_appliance(appliances):
    if isinstance(appliances, float):
        return appliances
    else:
        values = appliances.split(',')
        return len(values)


all_features['Heating'] = all_features['Heating'].apply(convert_heating)

all_features['Appliances included'] = all_features['Appliances included'].apply(convert_appliance)
all_features['Appliances included'] = all_features['Appliances included'].fillna(all_features['Appliances included'].mode().values[0])

all_features['Cooling'] = all_features['Cooling'].fillna(all_features['Cooling'].mode().values[0])

# 新特征提取
all_features['Listed On'] = 2023  - pd.to_datetime(all_features['Listed On']).dt.year


# 独热编码
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features = all_features.fillna(0)
all_features = all_features.astype('float32')


# 标准化
scaler = StandardScaler()
numeric_features = all_features.select_dtypes(include=['float32', 'int']).columns
all_features[numeric_features] = scaler.fit_transform(all_features[numeric_features])
all_features[numeric_features] = all_features[numeric_features].fillna(0)


# 划分训练集和测试集
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32).to(device)
train_labels = torch.tensor(train_data['Sold Price'].values.reshape(-1, 1), dtype=torch.float32).to(device)
train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32).to(device)


# 定义数据加载器
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

input_size = train_features.shape[1]
# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()
        )
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.layers(x)

model = Net().to(device)
# 定义损失函数
loss = torch.nn.SmoothL1Loss()
# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
infeatures = train_features.shape[1]

# 计算相对误差
def log_rmse(net, features, labels):
    clipped_preds = torch.clamp(net(features), 1,float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),torch.log(labels)))
    return rmse.item()

#训练
def train(num_epochs, train_features, train_labels, test_features, test_labels,device):
    train_ls, test_ls = [], []
    for epoch in range(num_epochs):
        for batch_idx, data in enumerate(train_loader,0):
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            optimizer.zero_grad()
            l = loss(model(inputs), target)
            l.backward()
            optimizer.step()


        train_ls.append(log_rmse(model, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(model, test_features, test_labels))
            print(f'Epoch {epoch + 1}, train log rmse {train_ls[-1]:.4f}, test log rmse {test_ls[-1]:.4f}')
        else:
            print(f'Epoch {epoch + 1}, train log rmse {train_ls[-1]:.4f}')
    return train_ls, test_ls

# K折交叉验证
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid

# 返回训练和验证误差的平均值
def k_fold(k, X_train, y_train, num_epochs):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        train_l, valid_l = train(num_epochs, *data, device=device)
        train_l_sum += train_l[-1]
        valid_l_sum += valid_l[-1]
        print(f'fold {i + 1}, train log rmse {train_l[-1]:.4f}, valid log rmse {valid_l[-1]:.4f}')
    return train_l_sum / k, valid_l_sum / k

k,num_epochs = 5, 100
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs)
print(f'{k}-fold validation: avg train log rmse {train_l:.4f}, avg valid log rmse {valid_l:.4f}')

# 预测
def predict(model, features):
    model.eval()
    with torch.no_grad():
        features = features.to(device)
        preds = model(features).detach().cpu().numpy()
    return preds
# 保存预测结果
preds = predict(model, test_features)
test_data['Sold Price'] = preds
test_data[['Id', 'Sold Price']].to_csv('加州房价预测/house-price-submission.csv', index=False)