#-*-conding:utf-8-*-
"""
Creaed on 2022年1月20日20:04:15
@anthor Yukuan Zhang

第七章：python图像的点运算处理
"""
import cv2

"""
1、图像点运算是指对图像中的每个像素一次进行灰度变换的运算，主要用于改变一幅图像的灰度分布范围，通过一定的变换函数将图像的灰度值进行转换
，生成新的图像的过程。点运算是图像处理中的基础技术，常见的包括灰度化处理，灰度线性变换，灰度非线性便函，阈值化处理等。
"""

"""
2、图像点运算是指对于一幅输入图像，将产生一幅输出图像，输出图像的每个像素点的灰度值由输入像素点决定。
点运算实际上是灰度到灰度的映射过程，通过映射变换来增强或者减弱图像的灰度。还可以对图像进行求灰度直方图、线性变换、非线性变换以及图像骨架的提取。它与相邻的像素点之间没有运算关系，是一种简单有效的图形处理方法。
它实际中有很多应用：光度学标定、对比度增强、显示标定、轮廓线确定
"""
"""
3、图像的灰度变换可以通过有选择地突出图像感兴趣的特征或者抑制图像中不需要的特征，从而改善图像的质量，凸显图像的细节，提高图像的对比度。
它也可以有效地改变图像的直方图分布，使图像的像素值分布更为均匀。
"""
"""
4、图像灰度化处理
图像灰度化是将一幅彩色图像转换为灰度化图像的过程。彩色图像通常包括R、G、B三个分量，分别显示出红绿蓝等各种颜色，灰度化就是使彩色图像的
RGB三个分量相等的过程。
灰度图像中每个像素仅具有一种样本颜色，其灰度是位于黑色与白色之间的多级色彩深度，灰度值大的像素点比较亮，反之比较暗，像素值最大为255，最小为0.

"""
"""
5、灰度处理算法：
最大值灰度处理：gray = max(R,G,B)
浮点灰度处理: gray = R*0.3+ G*0.59 + B*0.11
整数灰度处理:gray = (R*30+G*59+B*77)/100
移位灰度处理 : gray = (R*28 + G*151 + B*77)>>8
平均灰度处理: gray = (R+G+B)/3
加权平均灰度处理 : gray=  R*0.299+G*0.587+B*0.144
"""
"""
6、灰度处理流程
上面gray表示灰度处理之后的颜色，然后将原始RGB（R,G,B）颜色均匀地替换成新颜色RGB（gray，grya，gray），从而将彩色图像转换为灰度图像。
灰度图将一个像素点的三个颜色变量设置为相等，R=G=B，此时该值称为灰度值。
"""
"""
7、将rgb图像转换为其他颜色图像，使用函数cv2.cvtColor（src,code[,dst[,dstcn]]）
"""

#-----------------使用cv2.cvtcolor()将彩色图像灰度化---------------------------

"""
import cv2
import matplotlib.pyplot as plt

img0 = cv2.imread("tuxiangchuli.jpg")
img = cv2.cvtColor(img0,cv2.COLOR_BGR2RGB) # 之所以要先转换成RGB图像，是因为原始图像是BGR图像，当使用plt.imshow显示图像时，由于plt显示图像的接口是RGB，所以提前将原始图像转换成rgb图像，若不转换成rgb图像，则显示颜色不正常。当使用cv2.imshow显示图像时，不需要提前转换成RGB，因为cv2.imshow接口是BGR。
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)

plt.rcParams['font.sans-serif'] = ['SimHei']
titles = ['BGR原始图像','RGB原始图像','灰度图','HSV图像']
imges = [img0,img,gray,hsv]
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(imges[i],'gray') # 使用plt.imshow显示灰度图像必须加cmap=‘gray’,显示其他颜色图时，加不加都可以。
    plt.title(titles[i])
plt.show()
"""

"""
8、基于像素操作的图像灰度化处理方法，主要是最大值灰度化处理方法、平均灰度处理方法和加权平均灰度处理方法。
该方法需要遍历图像中的每一个像素点。
"""

#-------------------------最大值灰度化处理方法、平均灰度处理方法和加权平均灰度处理方法-------------------------
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np


img = cv2.imread("tuxiangchuli.jpg")
img = cv2.resize(img,None,fx=0.2,fy=0.2)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
height = img.shape[0]
weight = img.shape[1]

new_img_gray1 = np.zeros((height,weight,3),dtype="uint8")
new_img_gray2 = np.zeros((height,weight,3),dtype="uint8")
new_img_gray3 = np.zeros((height,weight,3),dtype="uint8")

for i in range(height):
    for j in range(weight):
        max_gray = max(img[i,j][0],img[i,j][1],img[i,j][2]) # 最大值灰度化处理方法
        new_img_gray1[i,j] = np.uint8(max_gray)
        ave_gray = (int(img[i,j][0]) + int(img[i,j][1]) + int(img[i,j][2])) / 3 # 平均灰度处理方法
        new_img_gray2[i,j] = np.uint8(ave_gray)
        w_gray = 0.30 * img[i,j][0] +0.59* img[i,j][1] + 0.11*img[i,j][2] # 加权平均灰度处理方法
        new_img_gray3[i,j] = np.uint8(w_gray)
plt.rcParams['font.sans-serif'] = ['SimHei']
titles = ['原始图像','最大平均灰度图','平均灰度处理方法','加权平均灰度处理方法']
imgse = [img ,new_img_gray1,new_img_gray2,new_img_gray3]

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(imgse[i])
    plt.title(titles[i])
plt.show()
"""


"""
9、图像的灰度线性变换
图像的灰度线性变换是通过建立灰度映射来调整原始图形的灰度，
从而改善图像的质量，凸显图像的细节，提高图像的对比度。
灰度线性变换公式：y = f（x）=ax+b
其中，当 a = 1，b=0时，保持原始图像。
a=1，b！=0时，图像所有的灰度值上移或下移。
a=-1，b=255时，原始图像的灰度值反转。
a>1时，输出图像的对比度增强。
a<0时原始图像暗区域变亮，亮区域变暗，图像求补。
0<a<1时，输出图像的对比度减小。
"""

#--------------------图像灰度线性变换--------------------
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("tuxiangchuli.jpg")
img = cv2.resize(img,None,fx=0.2,fy=0.2)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
height = img_gray.shape[0]
width = img_gray.shape[1]
rezult = np.zeros((height,width),dtype="uint8")
rezult2 = np.zeros((height,width),dtype="uint8")
rezult3 = np.zeros((height,width),dtype="uint8")
rezult4 = np.zeros((height,width),dtype="uint8")

for i in range(height):
    for j in range(width):
        if int(img_gray[i,j]+50) > 255:  # 灰度值上移50
            rezult[i,j] = np.uint8(255)
        else:
            rezult[i,j] = np.uint8(int(img_gray[i,j]+50))
        if int(img_gray[i,j]*1.5) > 255: # 对比度增强
            rezult2[i,j] = np.uint8(255)
        else:
            rezult2[i,j] = np.uint8(int(img_gray[i,j]*1.5))
        rezult3[i,j] = np.uint8(int(img_gray[i,j]*0.8)) # 对比度减弱
        rezult4[i,j] = np.uint8(255 - img_gray[i,j]) # 图像灰度反色变换
plt.rcParams['font.sans-serif'] = ['SimHei']
imgs = [img_gray,rezult,rezult2,rezult3,rezult4]
titles = ["原始图像", "灰度值上移50","对比度增强","对比度减弱","反色变换"]
for i in range(5):
    plt.subplot(2,3,i+1)
    plt.imshow(imgs[i],'gray')
    plt.title(titles[i])
plt.show()
"""
"""
10、图像灰度非线性变换

图像的灰度非线性变换主要包括对数变换、幂次变换、指数变换、分段函数变换，通过非线性关系对图像进行灰度变换。
对数变换能够提升较暗区域的对比度，增强图像的暗部细节，扩展被压缩的高值图像中的较暗像素。
伽马变换：又称为指数变换或幂次变换。公式为：
y = c*x^a
若，a > 1,会拉伸图像中灰度级较高的区域，压缩灰度级较低的部分。
若，a < 1,会拉伸图像中灰度级较低的区域，压缩灰度级较高的部分。
若，a = 1，该灰度变换为线性的。
"""
#----------灰度非线性变换---------------------
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("tuxiangchuli.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
height = img_gray.shape[0]
width = img_gray.shape[1]
rezult = np.zeros((height,width),dtype="uint8")
for i in range(height):
    for j in range(width):
        rezult[i,j] = np.uint8((int(img_gray[i,j] * int(img_gray[i,j]))) / 255) # y = x*x/255
rezult2 = np.uint8(42 * np.log(1.0 + img_gray)) # 对数变换

plt.rcParams['font.sans-serif'] = ['SimHei']
imgs = [img_gray,rezult,rezult2]
titles = ["原始图像", "乘方变换","对数变换"]
for i in range(3):
    plt.subplot(2,2,i+1)
    plt.imshow(imgs[i],'gray')
    plt.title(titles[i])
plt.show()
"""

#---------------------伽马变换----------------
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


# 绘制伽马函数
def gamma_plot(c,v):
    x = np.arange(0,256,0.01)
    y = c * x ** v
    plt.plot(x,y,'r',linewidth=1)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title("伽马变换函数")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([0,255])
    plt.ylim([0,255])
    plt.show()

gamma_plot(2.9,0.8)

# 伽马变换
def gamma(img,c,v):
    lut = np.zeros(256,dtype="float32")
    for i in range(256):
        lut[i] = c * i ** v
    output_img = cv2.LUT(img, lut) # 像素灰度值映射
    return output_img
img = cv2.imread("tuxiangchuli.jpg")
img = cv2.resize(img,None,fx = 0.2,fy = 0.2)
output = gamma(img,2.9,4.0)
cv2.imwrite("gama.jpg",output,[int(cv2.IMWRITE_JPEG_QUALITY),5])
cv2.imshow("gama",output)
cv2.waitKey()
cv2.destroyAllWindows()
"""

"""
11、图像阈值化处理
图像阈值化旨在剔除掉图像中一些低于或高于一定值的像素，从而提取图像中的物体，将图像的背景和噪声区分开。
进行阈值化处理时，首先需要将图像灰度化。
opencv中提供了固定阈值化函数threshold（）、自适应阈值化函数adaptiveThreshold（）
"""
"""
12、固定阈值化函数
cv2.threshold(src,thresh,maxval,type[,dst])
其中，src表示输入图像的数组，8位或32位浮点类型的多通道数。
thresh表示阈值
maxval表示最大值
type表示阈值类型。
阈值类型有
cv2.THRESH_BINARY:表示大于阈值的像素点的灰度值设为最大值，小于阈值的灰度值设定为0.
cv2.THRESH_BINARY_INV：表示大于阈值的灰度值设定为0，小于阈值的灰度值设定为最大值。
cv2.THRESH_TRUNC：表示小于阈值的灰度值不改变，大于阈值的灰度值设定为该阈值。
cv2.THRESH_TOZERO：表示小于阈值的灰度值不改变，大于阈值的灰度值设定0。
cv2.THRESH_TOZERO_INV：表示大于阈值的灰度值不改变，小于阈值的灰度值设定为0。
"""
"""
13、阈值化处理广泛应用于各行各业，如生物学中的细胞图分割、交通领域的车牌识别等。
通过阈值化处理将图像转换为黑白两色，从而为后续的图像识别和图像分割提供更好的支撑作用。
"""
#---------------------阈值化操作----------------------
"""
import cv2
import numpy as np

img = cv2.imread("tuxiangchuli.jpg")
img = cv2.resize(img,None,fx=0.2,fy=0.2)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
thresh,result = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY) # 二进制阈值化
thresh1,result1 = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY_INV) # 反二进制阈值化
thresh2,result2 = cv2.threshold(img_gray,127,255,cv2.THRESH_TRUNC) # 截断阈值化
thresh3,result3 = cv2.threshold(img_gray,127,255,cv2.THRESH_TOZERO) # 阈值化为0
thresh4,result4 = cv2.threshold(img_gray,127,255,cv2.THRESH_TOZERO_INV) # 反阈值化为0
cv2.imshow("threshold1",result1)
cv2.imshow("threshold",result)
cv2.imshow("threshold4",result4)
cv2.imshow("threshold3",result3)
cv2.imshow("threshold2",result2)
cv2.waitKey()
cv2.destroyAllWindows()
"""

"""
14、自适应阈值化处理
当同一幅图像上的不同部分具有不同亮度时，固定阈值化不再适用。此时需要采用自适应阈值化处理方法，根据图像上的每一个小区域，计算与其
对应的阈值，从而使得同一幅图像上的不同区域采用不同的阈值，在亮度不同情况下得到更好的结果。
opencv实现函数为：
cv2.adaptiveThreshold(src,maxValue,adaptiveMethod,thresholdType,blockSize,C[,dst])
src表示输入图像
maxValue表示给像素赋的满足条件的最大值
adaptivemethod表示适用的自适应阈值算法
thresholdType表示阈值类型
blocksize表示计算阈值的像素邻域大小，取值为3,5,7等
c表示一个常数，阈值等于平均值或者加权平均减去这个常数

当阈值类型为THRESH_BINARY时，像素灰度值大于阈值T(X,Y)时，取最大值，小于阈值时，取0
当阈值类型为THRESH_BINARY_INV时，灰度值大于阈值时，取0，小于阈值时，取最大值
当adaptivemethod参数采用ADAPTIVE_MEAN_C时，阈值T（x，y）为blocksize*blocksize邻域内（x,y）减去参数C的平均值。
当adaptivemethod参数采用ADAPTIVE_GAUSSIAN_C时，阈值T（x，y）为blocksize*blocksize邻域内（x,y）减去参数C与高斯窗交叉相关的加权总和。。

"""
#--------------自适应阈值化---------------------

import cv2


img = cv2.imread("tuxiangchuli.jpg")
img = cv2.resize(img,None,fx=0.2,fy=0.2)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

b = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2) # 邻域平均值分隔，效果差
b1 = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2) # 邻域加权平均值分隔，采用高斯函数分布，效果好
b2 = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2) # 邻域加权平均值分隔，采用高斯函数分布，效果好

cv2.imshow("yuzhi1",b)
cv2.imshow("yuzhi2",b1)
cv2.imshow("yuzhi3",b2)
cv2.imwrite("yuzhi.png",b2)
cv2.waitKey()
cv2.destroyAllWindows()
