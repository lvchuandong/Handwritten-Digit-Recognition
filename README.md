# Handwritten-Digit-Recognition

##  train and test
运行train.py，训练50个epoch之后，测试识别准确率如下图所示，可以看到最高准确率可以到达99.32%
<div align="center">
<img src="https://github.com/lvchuandong/Handwritten-Digit-Recognition/blob/master/images/%E6%88%AA%E5%9B%BE.png" width="800"  />
</div>
相应的损失和准确率变化
<div align="center">
<img src="https://github.com/lvchuandong/Handwritten-Digit-Recognition/blob/master/images/loss.png" width="400"/><img src="https://github.com/lvchuandong/Handwritten-Digit-Recognition/blob/master/images/accuracy.png" width="400"/>
</div>

##  GUI界面
##  打包成exe
可使用这个命令pyinstaller -F -i images/lcd.ico GUI.py -w进行打包
