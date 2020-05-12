from tkinter import *
from tkinter.filedialog import askopenfile
import torch
import torch.nn as nn
import cv2
from PIL import Image, ImageTk


# 网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 4, 5), nn.BatchNorm2d(4), nn.ReLU(), nn.Conv2d(4, 16, 5), nn.BatchNorm2d(16), nn.ReLU(), nn.Conv2d(16, 16, 2, stride=2))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 64, 3), nn.BatchNorm2d(64), nn.ReLU())
        self.fc1 = nn.Sequential(nn.Linear(4096, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                 nn.Linear(512, 84), nn.BatchNorm1d(84), nn.ReLU())
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        # torch.Size([256, 1, 28, 28])
        x = self.conv1(x)  # torch.Size([256, 16, 10, 10])
        x = self.conv2(x)  # torch.Size([256, 64, 8, 8])
        x = x.view(x.size()[0], -1)  # torch.Size([256, 4096])
        x = self.fc1(x)  # torch.Size([256, 84])
        out = self.fc2(x)  # torch.Size([256, 10])
        return out


def cal_result():
    global correct_mnist_number, all_mnist_number
    image = cv2.imread(img_path, 0)
    image = torch.Tensor(image)
    image = image.unsqueeze(0)
    image = image.unsqueeze(0)
    model = Net()
    model.load_state_dict(torch.load('model/model.pth'))
    model.eval()
    output_test = model(image)
    _, predict_test = torch.max(output_test, 1)
    predict_test = predict_test.numpy()[0]
    if predict_test == int(img_path[-5]):
        correct_mnist_number += 1
    correct_mnist_number_var.set(correct_mnist_number)
    all_mnist_number += 1
    all_mnist_number_var.set(all_mnist_number)
    correct_mnist_var.set(img_path[-5])
    # print(predict_test)
    result_number.set(predict_test)




# 选择路径函数
def selectpath():
    global img_path, img_open, img_png, label_img
    path_ = askopenfile()
    img_path = path_.name
    img_open = Image.open(img_path)
    img_png = ImageTk.PhotoImage(img_open)
    label_img.configure(image=img_png)
    root.update_idletasks()  # 更新图片，必须update


correct_mnist_number = 0
all_mnist_number = 0

root = Tk()
root.title("手写数字识别")
img_open = Image.open('images/lvh.png')
img_png = ImageTk.PhotoImage(img_open)
result_number = StringVar()
correct_mnist_number_var = StringVar()
all_mnist_number_var = StringVar()
correct_mnist_var = StringVar()
Button(root, text="选择图片", command=selectpath).grid(row=0, column=0)
label_img = Label(root, image=img_png)
label_img.grid(row=0, column=1)
Button(root, text="开始识别", command=cal_result).grid(row=0, column=2)
Label(root, text="识别结果:").grid(row=1, column=0)
Entry(root, textvariable=result_number).grid(row=1, column=1)
Label(root, text="正确结果:").grid(row=2, column=0)
Entry(root, textvariable=correct_mnist_var).grid(row=2, column=1)
Label(root, text="测试总数量:").grid(row=3, column=0)
Entry(root, textvariable=all_mnist_number_var).grid(row=3, column=1)
Label(root, text="测试正确数量:").grid(row=4, column=0)
Entry(root, textvariable=correct_mnist_number_var).grid(row=4, column=1)
root.mainloop()
