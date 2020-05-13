# Handwritten-Digit-Recognition

##  train and test
运行train.py，训练50个epoch之后，测试识别准确率如下图所示，可以看到最高准确率可以到达99.32%，lenet5可以达到99%的准确率，自己的结构加入了BN层，将maxpool用步长为2的conv2D代替
<div align="center">
<img src="https://github.com/lvchuandong/Handwritten-Digit-Recognition/blob/master/images/%E6%88%AA%E5%9B%BE.png" width="800"  />
</div>
相应的损失和准确率变化
<div align="center">
<img src="https://github.com/lvchuandong/Handwritten-Digit-Recognition/blob/master/images/loss.png" width="400"/><img src="https://github.com/lvchuandong/Handwritten-Digit-Recognition/blob/master/images/accuracy.png" width="400"/>
</div>

## 网络结构
相应网络结构可以查看images下的网络结构图片
##  GUI界面
运行GUI.py，可以利用训练好的模型进行手写数字的识别
<div align="center">
<img src="https://github.com/lvchuandong/Handwritten-Digit-Recognition/blob/master/images/exe1.png" width="400"/><img src="https://github.com/lvchuandong/Handwritten-Digit-Recognition/blob/master/images/exe2.png" width="400"/>
</div>

##  打包成exe
可使用这个命令pyinstaller -F -i images/lcd.ico GUI.py -w进行打包，其中lcd.ico是自己做的exe图标图片，可以使用编辑好的png图片进行[转换](https://www.convertico.com/)成.ico形式，会在生成的dist文件夹下
<div align="center">
<img src="https://github.com/lvchuandong/Handwritten-Digit-Recognition/blob/master/images/jietu2.png" width="400"  />
</div>
相应的exe文件[网盘链接](https://www.zhihu.com/question/23378396)，提取码：
