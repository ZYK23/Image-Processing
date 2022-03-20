# -*-coding:utf-8-*-
"""
Created on 2022年1月5日20:07:05
@author：Yukuan zhang
第三章：数字图像处理基础
"""


"""
1、常见的数字图像处理方法包括：算术处理、几何处理、图像增强、图像识别、图像分类、图像复原、图像重建、图像编码、图像理解
"""

"""
2、图像都是由像素（pixel）构成的，像素表示为图像中的小方格，这些小方格都有一个明确的位置和被分配的色彩数值，而这些小方格的颜色和位置
就决定了该图像呈现出来的样子。
像素是图像中的最小单位，每一个点阵图像包含了一定量的像素。这些像素决定图像在屏幕上所呈现出的大小。
"""
"""
3、图像通常分为二值图像、灰度图像和彩色图像
二值图像：即黑白图像，图像中任何一点非黑即白，要么为白色（pixel=255），要么为黑色（pixel=0）
灰度图像：每个像素的信息由一个量化的灰度级别来描述的图像，没有彩色信息。将彩色图转换为灰度图是图像处理的最基本预处理操作。
彩色图像：即RGB图像，表示红、绿、蓝三原色。即三通道。
"""

"""
4、图像处理所使用的模块：OpenCV
OpenCV通过imread（）函数读取图像，它将从指定的文件加载图像并返回矩阵，如果无法读取图像，则返回空矩阵。
基本格式：retval = cv2.imread(filename[,flags])
其中,filename表示需要载入的图片路径名，其支持Windows位图、jpeg、png、便携文件格式、sun rasters光栅文件、tiff文件、hdr文件。
flags为int类型，表示载入标识，指定一个加载图像的颜色类型，默认值为1.

通过imshow（）函数，将在指定窗口中显示一副图像，窗口自动调整为图像大小。
基本格式：retval = cv2.imshow（winname，mat）
其中，winname表示窗口的名称、mat表示要显示的图像。

在显示图像过程中通常要执行两个操作窗口的函数，cv2.waitKey（）返回类型为ASCII值、cv2.destroyAllWindows（），这俩函数结合起来就是按任意键关闭显示的图像
"""
"""
import cv2


img = cv2.imread("tuxiangchuli.jpg")
cv2.imshow("school", img)
cv2.waitKey()
cv2.destroyAllWindows()
"""

"""
5、Numpy是python提供的数值计算扩展包。
Array是Numpy库中最基础的数据结构，表示数组。
"""
#------创建一维数组，并获取其最大值、最小值、形状和维度
"""
import numpy as np
a = np.array([0,2,4,6,8,63,8,5])
print(a)
print(a.max())
print(a.min())
print(a.shape)
print(a.ndim)
"""
"""
6、在python图像处理中，主要通过Numpy库绘制一幅初始的背景图像，调用np.zeros（（size，3），np.uint8）函数绘制一幅3通道的长宽为size的黑色图像。
"""
"""
import cv2
import numpy as np

img = np.zeros((255,255,3),np.uint8)
cv2.imshow("ceshi", img)
cv2.waitKey()
cv2.destroyAllWindows()
"""

"""
7、python中的绘图库：Matplotlib
常用的函数：
Plot（）：用于绘制二维图、折线图，其格式为plt.plot（x,y,s）。其中x为横坐标，y为纵坐标，s为指定绘图的类型、样式和颜色。
Pie（）:用于绘制饼状图
Bar（）：用于绘制条装图
Hist（）：用于绘制二维条形直方图
Scatter（）：用于绘制散点图
"""
#--------绘制散点图---------
"""
import numpy as np
import matplotlib.pyplot as plt

x = np.random.randn(2000) # 生成2000随机点坐标
y = np.random.randn(2000)
size = 50*np.random.randn(2000)  # 设置2000个随机点的大小（随机生成大小）
colors = np.random.rand(2000)  # 设置2000个随机点的颜色（随机生成颜色）
plt.rc('font', family='SimHei', size=10)  # y用来正常显示中文标签 设置字体和大小
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.scatter(x,y,s=size,c=colors)  # 绘制散点图
plt.xlabel(u"X坐标")
plt.ylabel(u"Y坐标")
plt.title(u"Matpltlib 绘制散点图")
plt.show() # 显示图像
"""

#----------在一个窗口上显示多张图片---------
"""
import cv2
import matplotlib.pyplot as plt


img1 = cv2.imread("tuxiangchuli.jpg")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) # 将图像转换为RGB颜色空间
img2 = cv2.imread("tuxiangchuli.jpg")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img3 = cv2.imread("tuxiangchuli.jpg")
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
img4 = cv2.imread("tuxiangchuli.jpg")
img4 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
title = [1, 2, 3, 4]
image = [img1,img2,img3,img4]
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(image[i], 'gray')
    plt.title(title[i])
    plt.xticks([])
    plt.yticks([])
plt.show()
"""

"""
8、几何图像绘制
绘制方法：cv2.line() cv2.circle() cv2.rectangle() cv2.ellipse() cv2.putText()函数
"""

"""
绘制直线：
cv2.line(img,pt1,pt2,color[,thickness[,lineType[,shift]]])
其中，img表示需要绘制直线的图像
pt1表示线段第一个点的坐标
pt2表示线段第二个点的坐标
color表示线条颜色，需要转入一个RGB元组
thickness 表示线条粗细
lineType表示线条类型
shift表示点坐标中的小数位数
"""
#-----------绘制直线----------------
"""
import cv2
import numpy as np
img = np.zeros((640,640,3), np.uint8)
cv2.line(img, (0, 0), (100, 10), (0,0,255), 1)
cv2.line(img, (100, 10), (150, 100), (12,25,99), 1)
cv2.line(img, (150, 100), (170, 200), (44,99,88), 1)
cv2.line(img, (170, 200), (640, 640), (88,77,66), 1)
cv2.line(img, (640, 640), (0, 0), (230,240,213), 1)
cv2.imshow("zhixian", img)

img2 = np.zeros((640,640,3),np.uint8)
i = 0
while i < 640:
    cv2.line(img2, (0, i), (640, 640-i), (i,i,i), 10)
    i += 1

cv2.imshow("zhixian2", img2)
cv2.waitKey()
cv2.destroyAllWindows()

"""
"""
绘制矩形：
img = cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
其中，pt1表示矩形的左上角位置坐标，pt2表示矩形的右下角坐标
"""
#---------绘制矩形-----------
"""
import cv2
import numpy as np

img = np.zeros((640,640),np.uint8)
cv2.rectangle(img, (0, 0), (256, 256),(234, 0, 123),5)
cv2.imshow("ceshi", img)
img2 = np.zeros((640, 640), np.uint8)
i = 0
while i < 640:
    cv2.rectangle(img2, (0,i), (640-i, i), (255-i, 100+i, i), 5)
    i += 1
cv2.imshow("ceshi2", img2)
cv2.waitKey()
cv2.destroyAllWindows()

"""

"""
绘制圆形
img = cv2.circle(img , center, radius,  color[,thickness[,lineType[,shift]]])
其中，thickness如果为正值，表示圆轮廓的厚度，如果为负值表示要绘制一个填充圆
"""
#---------绘制圆------------------
"""
import cv2
import numpy as np

img = np.zeros((640,640,3), np.uint8)


i = 0
while i <= 255:
    cv2.circle(img,(300,300), 1 + i,(255-i,255-i,255-i),-1)
    i += 1
    if i <= 255:
        cv2.imshow("ceshi", img)
        cv2.waitKey(100)
cv2.destroyAllWindows()
"""

"""
绘制椭圆：
cv2.ellipse(img,center,axes,angle,statAngle,endAngle,color[,thickness[,lineType[,shift]]])
其中，axes表示轴的长度（短半径和长半径）
angle：表示偏转的角度（逆时针旋转）
starangle表示圆弧起始角的角度（逆时针旋转）
endangle表示圆弧终结角的角度（逆时针旋转）
thickness如果为正表示椭圆轮廓的厚度，如果为负表示要绘制一个填充椭圆

"""
#----------绘制椭圆------------
"""
import cv2
import numpy as np

img = np.zeros((640,640,3),np.uint8)
i = 0
while i <= 255:
    cv2.ellipse(img,(320,323),(100,50+i),20,0,100+i,(255-i,200+i,i),-2)
    i += 1
    cv2.imshow("ceshi", img)
    cv2.waitKey(100)
cv2.destroyAllWindows()
"""

"""
绘制多边形：
img = cv2.polyines(img,pts,isclosed,color[,thickness[,lineType[,shift]]])
其中，pts表示多边形曲线阵列
isclosed表示绘制的多边形是否闭合，False表示不闭合
"""
#------------绘制多边形-------------

"""
import cv2
import numpy as np


img = np.zeros((880,880,3), np.uint8)
i = 0
while i <= 255:
    pts = np.array([[10+i, 80+i], [120+i, 80+i], [60+i, 200+i], [30+i, 250]])
    cv2.polylines(img,[pts],False,(255-i,100+i,65+i),5+i)
    i += 1
    cv2.imshow("ceshi",img)
    cv2.waitKey(100)
cv2.destroyAllWindows()
"""

"""
img = cv2.putText(img,text,org,fontFace,fontScale,color[,thickness[,lineType[,bottomLeftOrigin]]])
其中，text表示要绘制的文字
org表示要绘制的位置，图像中文本字符串的左下角。
fontFace表示字体类型，具体查看see cv：：HersheyFonts。
fontScale表示字体的大小
bootomleftorigin如果是真，则图像数据原点位于左下角，否则它咋左上角
"""

#----------------绘制文字-------------
"""
import cv2
import numpy as np

img = np.zeros((6400,6400,3),np.uint8)
i = 0
while i <= 255:
    cv2.putText(img,'How are you ！',(50,320),cv2.FONT_HERSHEY_SIMPLEX,5,(255-i,i,100+i),2)
    cv2.putText(img,"Are you eating shit ？",(50,620),cv2.FONT_HERSHEY_SIMPLEX,5,(21+i,126,100-i),2)

    i += 1
    cv2.imshow("ceshi",img)
    cv2.waitKey(10)
cv2.destroyAllWindows()
"""