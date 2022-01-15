import torch as pt
import torchvision as ptv
import numpy as np
import os

# 载入数据
train_set = ptv.datasets.MNIST("/MNIST_Data/train",train=True,transform=ptv.transforms.ToTensor(),download=True)
test_set = ptv.datasets.MNIST("/MNIST_Data/test",train=False,transform=ptv.transforms.ToTensor(),download=True)
train_dataset = pt.utils.data.DataLoader(train_set,batch_size=100)
test_dataset = pt.utils.data.DataLoader(test_set,batch_size=100)
# 网络使用最简单的MLP模型，使用最简单的线性层即可构建,本次网络一共有3层全连接层，分别为28*28->512,512->128,128->10,除了输出层的激活函数使用softmax以外，其他均采用relu
class MLP(pt.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = pt.nn.Linear(784, 512)
        self.fc2 = pt.nn.Linear(512, 128)
        self.fc3 = pt.nn.Linear(128, 10)
    def forward(self, din):
        din = din.view(-1, 28 * 28)
        dout = pt.nn.functional.relu(self.fc1(din))
        dout = pt.nn.functional.relu(self.fc2(dout))
        return pt.nn.functional.softmax(self.fc3(dout), dim=1)
model = MLP().cuda()

# 损失函数和优化器设计
optimizer = pt.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # SGD优化器
lossfunc = pt.nn.CrossEntropyLoss().cuda() # 交叉熵损失

# 准确率
def AccuarcyCompute(pred,label):
    pred = pred.cpu().data.numpy()
    label = label.cpu().data.numpy()
    test_np = np.argmax(pred, 1) == label
    test_np = np.float32(test_np)
    return np.mean(test_np)

# 训练网络
# 训练网络的步骤分为以下几步：
# 1.初始化，清空网络内上一训练得到的梯度
# 2.载入数据为Variable，送入网络进行前向传播
# 3.计算代价函数，并进行反向传播计算梯度
# 4.调用优化器进行优化
for x in range(4):
    for i, data in enumerate(train_dataset):
        optimizer.zero_grad()        # 1.初始化，清空上一次训练得到的梯度
        (inputs, labels) = data
        inputs = pt.autograd.Variable(inputs).cuda()   # 传入数据计算梯度
        labels = pt.autograd.Variable(labels).cuda()
        outputs = model(inputs)      # 2.数据载入网络得到输出
        loss = lossfunc(outputs, labels)   # 计算误差
        loss.backward()                    # 3.误差反向传播
        optimizer.step()                   # 4.调用优化器进行优化
        if i % 100 == 0:
            print(i, ":", AccuarcyCompute(outputs, labels))

# 测试网络,使用测试集训练网络，直接计算结果并将计算准确率即可
accuarcy_list = []
for i, (inputs, labels) in enumerate(test_dataset):
    inputs = pt.autograd.Variable(inputs).cuda()
    labels = pt.autograd.Variable(labels).cuda()
    outputs = model(inputs)
    accuarcy_list.append(AccuarcyCompute(outputs,labels))
print('测试集训练网络的准确率为：', sum(accuarcy_list) / len(accuarcy_list))

#保存网络
'''
pytorch提供了两种保存网络的方法，分别是保存参数和保存模型
1.保存参数：仅仅保存网络中的参数，不保存模型，在load的时候要预先定义模型
2.保存模型：保存全部参数与模型，load后直接使用
'''
# 用绝对路径保存
# # only save paramters
# if os.path.exists('H:\PycharmProjects\pythonProject\mlp') == True:
#     pt.save(model.state_dict(), "H:\PycharmProjects\pythonProject\mlp\mlp_params.pt")
# else:
#     os.makedirs('H:\PycharmProjects\pythonProject\mlp')
#     pt.save(model.state_dict(), "H:\PycharmProjects\pythonProject\mlp\mlp_params.pt")
# # save model
# if os.path.exists('H:\PycharmProjects\pythonProject\mlp') == True:
#     pt.save(model, "H:\PycharmProjects\pythonProject\mlp\mlp_model.pt")
# else:
#     os.makedirs('H:\PycharmProjects\pythonProject\mlp')
#     pt.save(model, "H:\PycharmProjects\pythonProject\mlp\mlp_model.pt")

# 用相对路径保存
# only save paramters
if os.path.exists('./mlp') == True:
    pt.save(model.state_dict(), "./mlp/mlp_params.pt")
else:
    os.makedirs('.\mlp')
    pt.save(model.state_dict(), "./mlp/mlp_params.pt")
# save model
if os.path.exists('./mlp') == True:
    pt.save(model, "./mlp/mlp_model.pt")
else:
    os.makedirs('.\mlp')
    pt.save(model.state_dict(), "./mlp/mlp_model.pt")

# 用保存的参数进行测试得到准确率
test_save_net = MLP().cuda()
test_save_net.load_state_dict(pt.load("./mlp/mlp_params.pt"))
accuarcy_list = []
for i,(inputs,labels) in enumerate(test_dataset):
    inputs = pt.autograd.Variable(inputs).cuda()
    labels = pt.autograd.Variable(labels).cuda()
    outputs = model(inputs)
    accuarcy_list.append(AccuarcyCompute(outputs,labels))
print(sum(accuarcy_list) / len(accuarcy_list))
# 用保存的模型进行测试得到准确率
test_save_model = pt.load("./mlp/mlp_model.pt")
accuarcy_list = []
for i,(inputs,labels) in enumerate(test_dataset):
    inputs = pt.autograd.Variable(inputs).cuda()
    labels = pt.autograd.Variable(labels).cuda()
    outputs = model(inputs)
    accuarcy_list.append(AccuarcyCompute(outputs,labels))
print(sum(accuarcy_list) / len(accuarcy_list))