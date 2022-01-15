import os
from skimage import io
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.utils import data
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.datasets import mnist



# 第一步 数据集加载
# 1.定义加载数据集函数
def load_data_fashion_mnist(batch_size):
    '''下载Fashion-MNIST数据集然后加载到内存中'''
    train_dataset = datasets.FashionMNIST(root='./MNIST_Data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.FashionMNIST(root='./MNIST_Data', train=False, transform=transforms.ToTensor(), download=True)
    return data.DataLoader(train_dataset, batch_size, shuffle=True), data.DataLoader(test_dataset, batch_size, shuffle=False)

train_dataset = datasets.FashionMNIST(root='./MNIST_Data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.FashionMNIST(root='./MNIST_Data', train=False, transform=transforms.ToTensor(), download=True)
print('训练集长度为：', len(train_dataset), '测试集长度为：', len(test_dataset)) # 查看数据集的长度
# print(train_dataset[0][0].shape)  # 查看训练集的第一个样本数据的形状
# print(train_dataset[0][1]) # 查看训练集的第一个样本标签

# 2.加载数据集
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)
batch_size_for_vision = 20

# 3.训练集可视化
train_loader = DataLoader(train_dataset, batch_size=batch_size_for_vision, shuffle=True)
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# obtain one batch of training images
data_iter = iter(train_loader)
images, labels = data_iter.next() # data_iter.next()加载一个随机批次的图像/标签数据
images = images.numpy()
# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(batch_size_for_vision):
    ax = fig.add_subplot(2, batch_size_for_vision / 2, idx + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]))
    ax.set_title(classes[labels[idx]])
plt.show()

# 4.更详细地查看图像
# select an image by index
idx = 3
img = np.squeeze(images[idx])
# display the pixel values in that image
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')
width, height = img.shape
thresh = img.max() / 2.5
for x in range(width):
    for y in range(height):
        val = round(img[x][y], 2) if img[x][y] != 0 else 0
        ax.annotate(str(val), xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='white' if img[x][y] < thresh else 'black')
plt.show()

# 5.将数据集图片转化为图片可以逐张查看
root="./MNIST_Data/raw/"
train_set = (mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
             mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte')))
test_set = (mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte')))
def convert_to_img(train=True):
    if (train):
        f = open(root + '/ train.txt', 'w')
        data_path = './MNIST_Data/data_vision/train/'
        if (not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img, label) in enumerate(zip(train_set[0], train_set[1])):
            img_path = data_path + str(i) + '.jpg'
            io.imsave(img_path, img.numpy())
            f.write(img_path + ' ' + str(label) + 'n')
        f.close()
    else:
        f = open(root + '/ test.txt', 'w')
        data_path = './MNIST_Data/data_vision/test/'
        if (not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img, label) in enumerate(zip(test_set[0], test_set[1])):
            img_path = data_path + str(i) + '.jpg'
            io.imsave(img_path, img.numpy())
            f.write(img_path + ' ' + str(label) + 'n')
        f.close()
convert_to_img(True)  # 转换训练集
convert_to_img(False)  # 转换测试集



# 第二步 定义网络模型、网络参数、损失函数以及优化器
# 1.实现单隐藏层的多层感知机，其中有256个隐藏单元
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

# 2.初始化线性层的网络参数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights)

# 3.损失函数
loss_function = nn.CrossEntropyLoss()

# 4.优化器
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)





# 第三步:用训练集数据训练网络
# 1.开始训练
num_epochs = 10
train_loss = []
for epoch in range(num_epochs):
    for batch_idx, (x, y) in enumerate(train_iter):
        x = x.view(x.size(0), 28 * 28)
        out = net(x)
        y_onehot = F.one_hot(y, num_classes=10).float()  # 转为one-hot编码
        loss = loss_function(out, y_onehot)  # 均方差
        # 清零梯度
        optimizer.zero_grad()
        loss.backward()
        # w' = w -lr * grad
        optimizer.step()
        train_loss.append(loss.item())
        if batch_idx % 10 == 0:
            print(epoch, batch_idx, loss.item())

            
            
# 第四步：评估模型（损失和测试集准确率）
# 1.绘制损失曲线
plt.figure(figsize=(16, 8))
plt.grid(True, linestyle='--', alpha=0.5)
plt.plot(train_loss, label='loss')
plt.legend(loc="best")
df = pd.DataFrame(train_loss, columns=['loss'])
# df.plot(figsize=(16,8), alpha=0.1, marker='.', grid=True)
plt.show()

# 2.测试集准确率计算
total_correct = 0
for batch_idx, (x, y) in enumerate(test_iter):
    x = x.view(x.size(0), 28 * 28)
    out = net(x)
    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum().float().item()
    total_correct += correct
total_num = len(test_iter.dataset)
test_acc = total_correct / total_num
print("test acc:", test_acc)
