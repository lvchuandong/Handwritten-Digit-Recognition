import torch
import torch.nn as nn
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import pickle


# 网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 6, 5), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(nn.Conv2d(6, 16, 3), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.fc1 = nn.Sequential(nn.Linear(400, 120), nn.ReLU(), nn.Linear(120, 84), nn.ReLU())
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)  
        x = self.conv2(x)  
        x = x.view(x.size()[0], -1)  
        x = self.fc1(x)  
        out = self.fc2(x)  
        return out


# 训练过程
def train():
    train_loss = 0
    for i, data in enumerate(train_dadaset_loader):  # 938
        model.train()
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = Loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss = train_loss + loss.item()
        if i % 100 == 99:
            print("epoch:{} batch:{} loss:{}".format(epoch, i+1, train_loss/100))
            train_loss_list.append(train_loss/100)
            if train_loss < 0.1:
                torch.save(model, 'model_all.pth')
                torch.save(model.state_dict(), 'model.pth')
            train_loss = 0
            test()


# 测试过程
def test():
    test_loss = 0
    model.eval()
    with torch.no_grad():
        correct_number = 0
        for data, target in test_dadaset_loader:
            output_test = model(data)
            _, predict_test = torch.max(output_test, 1)
            correct_number += (predict_test == target).sum()
        print("test accuracy is {}/{} ({:.4f}%)".format(correct_number.item(), len(test_dadaset_loader.dataset), 100*correct_number.item()/len(test_dadaset_loader.dataset)))
        test_accuracy_list.append(correct_number.item()/len(test_dadaset_loader.dataset))


# 保存损失和准确率变化以及绘制相应的曲线
def show_image():
    # 保存损失和准确率数据
    loss_and_accuracy = {}
    loss_and_accuracy['loss'] = train_loss_list
    loss_and_accuracy['accuracy'] = test_accuracy_list
    with open('loss_and_accuracy.pkl', 'wb') as f:
        pickle.dump(loss_and_accuracy, f, pickle.HIGHEST_PROTOCOL)
    # 进行绘制损失和准确率曲线
    plt.figure(1)
    plt.plot(range(len(train_loss_list)), train_loss_list)
    plt.xlabel('iteration')
    plt.ylabel('Loss')
    plt.savefig('loss.png')

    plt.figure(2)
    plt.plot(range(len(test_accuracy_list)), test_accuracy_list)
    plt.xlabel('iteration')
    plt.ylabel('Accuracy')
    plt.savefig('accuracy.png')

batch_size = 256  # batch大小
epochs = 50  # 总共训练
device = torch.device("cpu")

# 相应的训练集
train_dadaset_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
              transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1037,), (0.3081,))])),
    batch_size=batch_size, shuffle=True)
# 相应的测试集
test_dadaset_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.1037,), (0.3081,))])),batch_size=batch_size, shuffle=True)
model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0002)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
Loss_function = nn.CrossEntropyLoss()
train_loss_list, test_accuracy_list = [], []

if __name__ == '__main__':
    for epoch in range(epochs):
        train()
        show_image()