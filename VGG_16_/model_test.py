import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import VGG16

data_path = 'D:/python代码调试/项目实战/VGG-16/data'

def test_data_process():
    test_data = FashionMNIST(root=data_path, train=False, transform=transforms.Compose([transforms.Resize(size = 224), transforms.ToTensor()]), download = True)
    
    test_dataloader = Data.DataLoader(dataset=test_data, batch_size =1, shuffle=True, num_workers=0)
    
    return test_dataloader

def test_model_process(model, test_dataloader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    #初始化的参数
    test_corrects = 0.0
    test_num = 0
    #只进行前向传播，不进行梯度计算
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)
            
            model.eval()
            #前向传播预测结果
            output = model(test_data_x)
            #查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim = 1)
            #统计正确预测个数
            test_corrects += torch.sum(pre_lab == test_data_y.data)
            
            test_num += test_data_x.size(0)
        #计算准确率    
        test_acc = test_corrects.double().item() / test_num
        print("测试准确率为：", test_acc)
if __name__=="__main__":
    #加载模型
    model = VGG16()
    
    model.load_state_dict(torch.load('D:/python代码调试/项目实战/VGG-16/best_model.pth'))
    #加载测试数据
    test_dataloader = test_data_process()
    #加载模型测试的函数
    test_model_process(model, test_dataloader)
    
    #打印测试结果
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    classes = ['T-shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boots']
    with torch.no_grad():
        for b_x, b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            
            model.eval()
            output = model(b_x)
            pre_lab = torch.argmax(output, dim = 1)
            result = pre_lab.item()
            label = b_y.item()
            
            print("预测值:",classes[result],"-----","真实值",classes[label])