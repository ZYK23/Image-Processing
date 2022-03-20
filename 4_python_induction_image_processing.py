#-*-coding:utf-8-*-

"""
Creaded on 2022年1月6日13:32:20
@author： Yukuan Zhang
第四章：python图像处理入门
"""
import matplotlib.pyplot as plt

"""
1、OpenCV读取修改像素
OpenCV中读取图像的像素值可以直接通过遍历图像的位置实现，如果是灰度图像则返回其灰度值，如果是彩色图像则韩慧蓝色（b）、绿色（g）、红色（r）三分量值。
查看图像中某个位置像素的方法：
灰度图：返回值 = 图像[位置参数]
如：test = img[88,42]
彩色图像：返回值 = 图像[位置参数，0或1或2]，获取BGR三个通道的像素
如:blue = img[88,142,0]
    green = img[88,142,1]
    red = img[88,142,2]
"""
#----------读取图像，并修改其中像素-------
"""
import cv2
import numpy as np

img = cv2.imread("tuxiangchuli.jpg")
#print(img,type(img))
cv2.imshow("1",img)
#img2 = img[88:120,142:500] # 读取img的88-120行、142-500列
#print("********",img2)
img[88:120,142:500] = [255,255,255] # img的88-120行、142-500列设置为白色
#print(img)
#img3 = img[88:120,142:500]
#print("----------",img3)
cv2.imshow("2",img)
cv2.waitKey()
cv2.destroyAllWindows()
"""


"""
2、OpenCV创建复制保存图像
创建图像：
需要使用numpy库函数实现
如：img = np.zeros（img.shape，np.uint8）
其中，img.shape为原始图像的形状，np.uint8表示类型
复制图像：
img2 = img.copy（） 
"""

#-------创建图像和复制图像-------
"""
import cv2
import numpy as np

img = cv2.imread("tuxiangchuli.jpg")
emptyimage = np.zeros((img.shape), np.uint8)
emptyimage2 = img.copy()
cv2.imshow("1",img)
cv2.imshow("2",emptyimage)
cv2.imshow("3",emptyimage2)
cv2.waitKey()
cv2.destroyAllWindows()
"""

"""
3、在opencv中，输出图像到文件使用的函数为imwrite（）。
retval = cv2.imwrite(filename,img[,params])
其中，filename为保存的路径及文件夹名字
img表示图像矩阵
params表示特定格式保存的参数编码，默认值为空。
对于JPEG图片，该参数（cv2.IMWRITE_JPEG_QUALITY）表示图像的质量，用0-100的整数表示，默认为95.
对于PNG图片，该参数（CV2.IMWRITE_PNG_COMPRESSION）表示压缩级别，从0-9，压缩级别逐渐增高，推向尺寸逐渐变小，默认级别为3.
对于ppm pgm pbm图像，该参数（CV2.IMWRITE_PXM_BINARY）表示一个二进制格式的标志。
注意，该类型为Long、必须转换成int。

"""

#-----------使用cv2.imwrite（）函数--------
"""
import cv2
import  numpy as np

img = cv2.imread("tuxiangchuli.jpg")
cv2.imshow("1",img)
cv2.imwrite("xiugai1.jpg",img,[int(cv2.IMWRITE_JPEG_QUALITY),5])
cv2.imwrite("xiugai.jpg",img,[int(cv2.IMWRITE_JPEG_QUALITY),100])
cv2.imwrite("xiugai3.png",img,[int(cv2.IMWRITE_PNG_COMPRESSION),0])
cv2.imwrite("xiugai4.png",img,[int(cv2.IMWRITE_PNG_COMPRESSION),9])
img1 = cv2.imread("xiugai1.jpg")
cv2.imshow("2",img1)
img2 = cv2.imread("xiugai.jpg")
cv2.imshow("3",img2)
img = cv2.imread("xiugai3.png")
cv2.imshow("4",img)
img3 = cv2.imread("xiugai4.png")
cv2.imshow("4",img3)
cv2.waitKey()
cv2.destroyAllWindows()
"""


"""
4、图像属性：形状（shape）、像素大小（size）、图像类型（dtype）
shape：通过shape关键字获取图像的形状，返回包含行数、列数、通道数的元组。其中灰度图像返回行数和列数，彩色图像函数行数、列数和通道数。
size:通过size关键字获取图像的像素大小，其中灰度图像返回行数*列数，彩色图像返回行数*列数*通道数
dtype：通过detype关键字获取图像的数据类型，通常返回uint8
"""
#----------获取图像属性及通道-------------
"""
import cv2

img = cv2.imread("tuxiangchuli.jpg")
print("形状",img.shape,"像素大小",img.size,"类型",img.dtype)

"""

"""
5、图像通道处理
OpenCV通过split（）函数和merge（）函数实现对图像通道的处理。图像通道的处理包括通道分离和通道合并。
OpenCV读取的彩色图像由蓝色、绿色、红色三原色组成，每一种颜色可以认为是一个通道分量。
b、g、r三通道显示的图片都是灰度图。每个通道的像素值量化到0-255.当着三通道融合后就变成了彩色图。
"""
"""
split()：用于将一个多通道数组分离成三个单通道，其函数原型如下所示。
mv = split（m，[mv]）
其中，m表示输入的多通道数组，mv表示输出的数组或vector容量。
"""
#----------split（）函数将彩色图分成单独三通道------------
"""
import cv2
img = cv2.imread("tuxiangchuli.jpg")
b0 = img[:,:,0]
g0 = img[:,:,1]
r0 = img[:,:,2]
#print("b = \n",b,"\ng = ",g,"\nr = ", r)

b,g,r = cv2.split(img)  # 获取所有通道
b1 = cv2.split(img)[0]  # 获取蓝通道
g1 = cv2.split(img)[1]  # 获取绿通道
r1 = cv2.split(img)[2]  # 获取红通道
####b0和b和b1都相同
if (b == b0).all and (g == g0).all and (r == r0).all:  
    cv2.imwrite("B.jpg",b,[int(cv2.IMWRITE_JPEG_QUALITY),100])  # 虽然结果显示灰度图，但保存结果仍然是彩色三通道图像，每个通道像素值相同。
    cv2.imwrite("G.jpg",g,[int(cv2.IMWRITE_JPEG_QUALITY),100])
    cv2.imwrite("R.jpg",r,[int(cv2.IMWRITE_JPEG_QUALITY),100])

"""

"""
merge()函数，该函数㐊split（）函数的逆向操作，将多个数组合成一个通道的数组，从而实现图像通道的合并。
dst = cv2.merge（mv[，dst]）
其中，mv表示输入的需要合并的数组，所有矩阵必须由相同的大小和深度。
dst表示输出的具有与mv[0]相同大小和深度的数组。
"""
#------------使用merge（）将三通道合并-----------
"""
import cv2
img = cv2.imread("tuxiangchuli.jpg")
b,g,r = cv2.split(img)
print(b.shape,g.shape,r.shape)
bgr = cv2.merge([b,g,r])
print(bgr.shape)
cv2.imwrite("BGR.jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY),100])
"""

#----------------使用merge（）提取图像的不同颜色，若提取b通道颜色，则把g、r通道颜色设置为0-------
"""
import cv2
import numpy as np

img = cv2.imread("tuxiangchuli.jpg")
rows,cols,chn = img.shape
b1,g1,r1 = cv2.split(img)
D0 = np.zeros((rows,cols),dtype=img.dtype)
imgB = cv2.merge([b1,D0,D0])  # 注意：merge（B,G,R），括号中的通道位置不能调动。
imgG = cv2.merge([D0,g1,D0])
imgR = cv2.merge([D0,D0,r1])
cv2.imwrite("imgB.JPG",imgB,[int(cv2.IMWRITE_JPEG_QUALITY),100]) # 显示出蓝通道图像,其他通道值为0.
cv2.imwrite("imgG.JPG",imgG,[int(cv2.IMWRITE_JPEG_QUALITY),100]) # 显示出绿通道图像
cv2.imwrite("imgR.JPG",imgR,[int(cv2.IMWRITE_JPEG_QUALITY),100]) # 显示出红通道图像

"""

"""
6、图像算术与逻辑运算
包括：图像加法运算、减法、与运算、或运算、异或运算、非运算
加法运算：det = cv2.add（src1,src2[,dst[,mask[,dtype]]]）
其中，src1表示第一幅图像的像素矩阵。
src2表示第二幅图像的像素矩阵。
dst表示输出的图像，必须与输入图像具有相同的大小和通道数。
mask表示可选操作掩码（8位单通道数组），用于指定要更改的输出数组的元素。
dtype表示输出数组的可选深度。
注意：当两幅图像的像素值相加结果相较于等于255时，输出图像直接赋值该结果，若相加结果大于255，则输出图像赋值255.

"""

#----------------图像加法运算----------------
"""
import cv2
import numpy as np

img = cv2.imread("tuxiangchuli.jpg")
m = np.ones(img.shape, dtype="uint8")*200 # 图像个像素+200
result = cv2.add(img,m)
cv2.imwrite("xiangjia.JPG",result,[int(cv2.IMWRITE_JPEG_QUALITY),100])
"""

"""
减法运算：
dst = cv2.subtract(src1,src2[,det[,mask[,dtype]]]) 
其中，src1表示第一幅图像的像素矩阵
src2表示第二幅图像的像素矩阵,src1-src2
dst表示输出的图像，必须与输入图像具有相同的大小和通道数
"""
#-----------图像减法--------------
"""
import cv2
import numpy as np

img = cv2.imread("tuxiangchuli.jpg")
m = np.ones(img.shape, dtype="uint8")*100
retult = cv2.subtract(img, m)
cv2.imwrite("jianfa.jpg",retult,[int(cv2.IMWRITE_JPEG_QUALITY),100])
"""

"""
图像与运算
图像的与运算是指两幅图像（灰度图或彩色图）的每个像素值进行二进制与操作，实现图像裁剪。
dst = cv2.bitwise_and(src1,src2[,det[,mask]])
"""

#--------------图像与运算-----------
"""
import cv2
import numpy as np

img = cv2.imread("tuxiangchuli.jpg")
rows,cols = img.shape[:2]
m = np.zeros(img.shape, dtype="uint8")
cv2.circle(m,(int(rows/2), int(cols/2)),500,(255,255,255),-1)
retult = cv2.bitwise_and(img,m)
cv2.imwrite("yu.jpg",retult,[int(cv2.IMWRITE_JPEG_QUALITY),100])

"""


"""
图像或运算
图像的或运算是指两幅图像的每个像素值进行二进制或操作,实现图像裁剪。
dst = bitwise_or(src1,src2[,det[,mask]])
"""

#----------图像或运算-----------
"""
import cv2
import numpy as np

img = cv2.imread("tuxiangchuli.jpg")
rows,cols = img.shape[:2]
m = np.zeros(img.shape,dtype="uint8")
cv2.circle(m,(int(rows/2),int(cols/2)),500,(255,255,255),-1)
retult = cv2.bitwise_or(img,m)
cv2.imwrite("huo.JPG",retult,[int(cv2.IMWRITE_JPEG_QUALITY),100])
"""

"""
图像异或运算
如果a，b两个值不相同，则异或结果为1，否则为0.
图像的异或运算是指两幅图像的每个像素值进行二进制异或操作，实现图像裁剪。
dst =  bitwise_xor(src1,src2[,det[,mask]])
"""

#----------图像异或运算--------
"""
import cv2
import numpy as np

img = cv2.imread("tuxiangchuli.jpg")
rows,cols = img.shape[:2]
m = np.zeros(img.shape,dtype="uint8")
cv2.circle(m,(int(rows/2),int(cols/2)),500,(255,255,255),-1)
retult = cv2.bitwise_xor(img,m)
cv2.imwrite("yihuo.JPG",retult,[int(cv2.IMWRITE_JPEG_QUALITY),100])
"""

"""
图像非运算
图像非运算就是图像的像素反色处理，它将原始图像的黑色像素点转换为白色像素点，白色像素点转换为黑色像素点。
dst =  bitwise_not(src1,src2[,det[,mask]])
"""
#-----------图像非运算--------------
"""
import cv2


img = cv2.imread("tuxiangchuli.jpg")
result = cv2.bitwise_not(img)
cv2.imwrite("fei.jpg",result,[int(cv2.IMWRITE_JPEG_QUALITY),100])
"""

"""
7、图像融合处理
图像融合通常是将两幅或两幅以上的图像信息融合到一幅图像上，融合的图像含有更多的信息，更方便人们观察或计算机处理。如将两张清晰度不同的画面内容相同的图像，进行融合，可以得到较高清晰度的图像。
图像融合是在图像加法的基础上增加了系数和亮度调节量，与图像加法的主要区别如下：
图像加法：目标图像=图像1+图像2
图像融合;图像1*系数+图像2*系数+亮度调节量
在OpenCV中图像融合主要调用addWeighted()函数实现，需要注意的是两幅图像的像素大小必须一致，参数gamma不能省略.该函数只能融合两张图像
dst = cv2.addWeighted(src1,alpha,src2,beta,gamma)g
"""

#-----------图像融合------------
"""
import cv2

img1 = cv2.imread("imgB.JPG")
img2 = cv2.imread("imgG.JPG")
img3 = cv2.imread("imgR.JPG")
img4 = cv2.imread("ronghe.JPG")
result = cv2.addWeighted(img3,1,img4,1,0)
cv2.imwrite("ronghe2.JPG",result,[int(cv2.IMWRITE_JPEG_QUALITY),100])

"""

"""
8、获取图像ROI区域
ROI表示感兴趣区域，是指从被处理图像以方框、原型、椭圆、不规则多边形等方式勾勒出要处理的区域。可以通过各种算子和函数求得ROI区域。
广泛应用于热点地图、人脸识别、图像分割等领域。
"""

#-----------通过像素矩阵直接扩区ROI区域-------------
"""
import cv2
img = cv2.imread("tuxiangchuli.jpg")
ROI = img[1500:2000,123:2000]
cv2.imshow("roi",ROI)
#-------将提取的roi区域融合到其他图像上------------
img2 = cv2.imread("fei.jpg")
img2[1500:2000,123:2000] = ROI
cv2.imshow("tihuan",img2)
cv2.imwrite("tihuan.jpg",img2,[int(cv2.IMWRITE_JPEG_QUALITY),100])
cv2.waitKey()
cv2.destroyAllWindows()
"""

"""
9、图像类别转换
在入场生活中，我们看到的大多数彩色图像都是RGB类型，但是在图像处理过程中，尝尝需要用到灰度图像、二值图像、HSV、HIS等。
最常用的类型转换有以下三类：
OpenCV提供了cv.cvtColor（）函数来实现类型转换功能。
dst = cv2.cvtcolor(src,code[,dst[,dstCn]])
其中，src表示输入图像，需要进行颜色空间变换的原图像
code表示转换的代码或标识
dst表示输出图像，其大小和深度与src一致。
dstCn表示目标图像通道数，其值为0时，有src和code决定。
注意：RGB是指红、绿、蓝，即一幅图像是由这三个通道构成；Gray表示只有灰度值一个通道；HSV包含色调、饱和度和明亮三个通道。
"""

"""
import cv2
img = cv2.imread("tuxiangchuli.jpg") # 原始图像时BGR
# ---------------BGR-GRAY----------------
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imwrite("GRAY.JPG",gray,[int(cv2.IMWRITE_JPEG_QUALITY),100])
#-------------BGR-HSV------------------------------
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
cv2.imwrite("HSV.JPG",hsv,[int(cv2.IMWRITE_JPEG_QUALITY),100])
#------------BGR-RGB------------
rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
cv2.imwrite("RGB.JPG",rgb,[int(cv2.IMWRITE_JPEG_QUALITY),100])
#------------BGR-YCRCB----------
ycrcb = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
cv2.imwrite("YCRCB.JPG",ycrcb,[int(cv2.IMWRITE_JPEG_QUALITY),100])
#------------BGR-HLS-------------
hls = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
cv2.imwrite("HLS.JPG",hls,[int(cv2.IMWRITE_JPEG_QUALITY),100])
#----------BGR-XYZ-------------
xyz = cv2.cvtColor(img,cv2.COLOR_BGR2XYZ)
cv2.imwrite("XYZ.JPG",xyz,[int(cv2.IMWRITE_JPEG_QUALITY),100])
#---------BGR-LAB--------------
lab = cv2.cvtColor(img,cv2.COLOR_BGR2Lab)
cv2.imwrite("LAB.JPG",lab,[int(cv2.IMWRITE_JPEG_QUALITY),100])
#---------BGR-YUV--------------
yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
cv2.imwrite("YUV.JPG",yuv,[int(cv2.IMWRITE_JPEG_QUALITY),100])

title = ['bgr','gray','hsv','rgb','ycrcb','hls','xyz','lab','yuv']
images = [img,gray,hsv,rgb,ycrcb,hls,xyz,lab,yuv]
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(images[i])
    plt.title(title[i])
    plt.xticks([]),plt.yticks([])
plt.show()
"""