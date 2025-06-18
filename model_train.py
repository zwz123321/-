from sympy import arg
from torchvision import transforms
from torchvision.datasets import FashionMNIST
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from model import LeNet
import torch
import torch.nn as nn
import copy
import time
import pandas as pd

# 示例路径（请替换为你自己的路径）
data_path = 'D:/python代码调试/项目实战/卷积神经网络实战LeNet_5/fashion-mnist-master/fashion-mnist-master/data'

def train_val_data_process():
    train_data = FashionMNIST(root=data_path, train=True, transform=transforms.Compose([transforms.Resize(size = 28), transforms.ToTensor()]), download = True)
    train_data, val_data = Data.random_split(train_data, lengths=[round(0.8*len(train_data)), round(0.2*len(train_data))])
    
    train_dataloader = Data.DataLoader(dataset=train_data, batch_size =32, shuffle=True, num_workers=0)
    val_dataloader = Data.DataLoader(dataset=val_data, batch_size =32, shuffle=True, num_workers=0)
    
    return train_dataloader, val_dataloader

def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    
    device = device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #损失函数为交叉熵函数
    criterion = nn.CrossEntropyLoss()
    
    model = model.to(device)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    
    #初始化参数
    #最高准确度
    best_acc =  0.0
    #训练集损失列表
    train_loss_all = []
    #验证集损失列表
    val_loss_all = []
    #训练集准确度列表
    train_acc_all =[]
    #验证集准确度列表
    val_acc_all = []
    
    #当前时间
    since = time.time()
    
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs-1))
        print("-"*10)
        
        #初始化参数
        #验证集准确度
        train_loss = 0.0
        #训练集准确度
        train_corrects = 0
        #验证集损失函数
        val_loss = 0.0
        #验证集准确度
        val_corrects = 0
        #训练集的样本数量
        train_num = 0
        #验证集的样本数量
        val_num = 0
        #对每一个mini-batch训练和计算
        for step,(b_x, b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            #设置模型为训练模式
            model.train()
            #输入为一个batch，输出为一个预测结果
            output = model(b_x)
            
            pre_lab = torch.argmax(output, dim=1)
            
            loss = criterion(output, b_y)
            
            #将梯度初始化为0
            optimizer.zero_grad()
            #反向传播计算
            loss.backward()
            #根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值的作用
            optimizer.step()
            #对损失函数进行累加
            train_loss+=loss.item() * b_x.size(0)
            #如果预测正确，则准确度train_corrects加1
            train_corrects += torch.sum(pre_lab==b_y.data)
            
            train_num += b_x.size(0)
            
        for step,(b_x, b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            #设置模型为评估模式
            model.eval()
            #前向传播过程，输入一个batch，输出为又给batch中对应的预测
            output = model(b_x)
            
            #查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim =1)
            
            #计算每一个batch的损失函数
            loss = criterion(output, b_y)
            
            #对损失函数进行累加
            val_loss+=loss.item() * b_x.size(0)
            
            #如果预测正确，则准确度train_corrects加1
            val_corrects += torch.sum(pre_lab==b_y.data)
            
            #用于验证的样本数量
            val_num += b_x.size(0)
        
        #计算并保存每一次迭代的loss值和准确率
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)
        
        print('{} Train Loss:{:.4f} Train Acc: {:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print('{} val Loss:{:.4f} val Acc: {:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))
        
        if val_acc_all[-1] > best_acc:
            #保存当前的最高准确度
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        # 训练耗费时间
        time_use = time.time() - since
        print("训练和验证耗费的实践{:.0f}m{:.0f}s".format(time_use//60, time_use%60))
        
    #选择最优参数
    #加载最高准确率下的模型参数
    
    torch.save(model.state_dict(best_model_wts), 'D:/python代码调试/项目实战/卷积神经网络实战LeNet_5/best_model.pth')
    # torch.save(best_model_wts, 'D:/python代码调试/项目实战/卷积神经网络实战LeNet_5/best_model.pth')
    
    train_process = pd.DataFrame(data = {"epoch":range(num_epochs),
                                        "train_loss_all":train_loss_all,
                                        "val_loss_all":val_loss_all,
                                        "train_acc_all":train_acc_all,
                                        "val_acc_all":val_acc_all})
    return  train_process

def matplot_acc_loss(train_process):
    plt.figure(figsize = (12,4))
    plt.subplot(1,2,1)
    plt.plot(train_process["epoch"], train_process.train_loss_all, 'ro-', label = "train loss")
    plt.plot(train_process["epoch"], train_process.val_loss_all, 'bs-', label = "val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    
    plt.subplot(1,2,2)
    plt.plot(train_process["epoch"], train_process.train_acc_all, 'ro-', label = "train acc")
    plt.plot(train_process["epoch"], train_process.val_acc_all, 'bs-', label = "val acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    #将模型实例化
    LeNet =LeNet()
    train_dataloader, val_dataloader = train_val_data_process()
    train_process = train_model_process(LeNet, train_dataloader, val_dataloader, 20)
    matplot_acc_loss(train_process)
    