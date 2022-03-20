#-*-conding:utf-8-*-
"""
Creaded on 2022年2月15日16:44:18
@Author Yukuan Zhang

第十一章 python图像平滑
"""
"""
1、在图像产生、传输和应用过程中，通常会导致图像数据丢失或被噪声干扰，从而降低图像的质量。
这就需要通过图像平滑方法来消除这些噪声并保留图像的边缘轮廓和线条清晰度，本章将详细介绍五种图像平滑的滤波算法，
包括均值滤波、方框滤波、高斯滤波、中值滤波和双边滤波。
"""
"""
2、图像平滑是一项简单且使用频率很高的图像处理方法，可以用来压制、弱化或消除图像中的细节、突变、边缘和噪声，最常见的是用
来减少图像上的噪声。
"""
"""
3、噪声滤除方法有很多，从设计方法上分为线性滤波算法和非线性滤波算法，
在图像处理中，对邻域中的像素的计算为线性运算时，如利用窗口函数进行平滑加权求和运算，或者某种卷积运算，都可以称为线性滤波。
在数字信号处理和数字图像处理的早期研究中，线性滤波器是噪声抑制处理的主要手段，如均值滤波、方框滤波、高斯滤波等。
线性滤波算法对高斯型噪声有较好的滤波效果，而当信号频谱与噪声频谱混叠时或当信号中含有非叠加型噪声时（如由系统非线性引起的噪声或存在非高斯噪声等），
线性滤波器的处理结果就很令人不满意。
非线性滤波器利用原始图像与模板之间的一种逻辑关系得到结果，如中值滤波、双边滤波等。非线性滤波技术从某种程度上弥补了线性滤波
方法的不足，由于它能够在滤除噪声的同时较好地保持图像信号的高频细节，从而得到了广泛的应用。
"""
"""
4、均值滤波
均值滤波是最简单的一种线性滤波算法，它是指在原始图像上对目标像素给一个模板，
该模板包括其周围的邻近像素（以目标像素为中心的周围的8个像素，构成一个绿波模板，即去掉目标像素本身），再用模板中的全体像素的平均值来代替
原来的像素值。
均值滤波算法比较简单，计算速度较快，对周期性的干扰噪声有很好的的抑制作用，但是它不能很好地保护图像的细节，在图像去噪的同时，
也破坏看图像的细节部分，从而使图像变得模糊。
python调用opencv中的cv2.blur（）函数实现均值滤波处理，其函数原型如下所示，输出的dst图像与输入图像src具有相同的大小和类型。
dst = cv2.blur(src,ksize[,dst[,anchor[,borderType]]])
其中，src表示输入图像
ksize表示模糊内核大小，以（宽度，高度）的形式呈现，常见模糊内核为（3,3），（5,5）
anchor表示锚点，即被平滑的那个点，其默认值point（-1，-1）表示位于内核的中央，可省略。
bordertype表示边框模式，用于推断图像外部像素的某种边界模式，可省略。

如果均值滤波后图像中噪声仍然存在，则可以增加模糊内核的大小，但是处理后的图像会逐渐变得更模糊。
"""
"""
5、方框滤波
方框滤波利用卷积运算对图像邻域的像素值进行平均处理，从而实现消除噪声。
调用opencv中的cv2.boxFilter（）实现方框滤波处理，其函数原型如下所示：
dst = cv2.boxFilter(src,depth,ksize[,dst[,anchor[,normalize[,borderType]]]])
其中，src表示输入图像
depth表示输出图像深度，通常设置为“-1”，表示与原图深度一直
ksize表示模糊内核大小，以（宽度，高度）的形式呈现，常见模糊内核为（3,3），（5,5）
anchor表示锚点，即被平滑的那个点，其默认值point（-1，-1）表示位于内核的中央，可省略。
normalize表示是否对目标图像进行归一化处理，默认值为true
bordertype表示边框模式，用于推断图像外部像素的某种边界模式，可省略。

参数normalize表示是否对目标图像进行归一化处理。
当normalize为true时，需要执行归一化处理，方框滤波就变成了均值滤波。其中，归一化就是要把处理的像素值都缩放到一个范围内，
以便统一处理和直观量化。
当normalize为false时，表示非归一化的方框滤波，不进行均值化处理，实际上就是求周围像素的和。但此时很容易发生溢出，多个像素值相加
后的像素值大于255，溢出后的像素值均设置为255，即白色。
"""
"""
6、高斯滤波
为了克服局部平均法造成的图像模糊的弊端，又提出了一些保持边缘细节的局部平滑算法，图像高斯滤波（高斯平滑），就是这样一种算法，
它是应用邻域平均思想对图像进行平滑的一种线性平滑滤波，对于抑制服从正泰分布的噪声非常有效，使用于消除高斯噪声，广泛应用于图像处理的减噪过程。，
图像高斯滤波为图像不同位置的像素值赋予了不同的权重，距离中心点像素越近的点权重越大，距离中心点像素越远的点权重越小。
它与方框滤波和均值滤波不同，它对邻域内的像素进行平均时，为不同位置的像素赋予不同的权值。通俗地讲，高斯滤波就是对整幅图像进行加权平均
的过程，每一个像素点的值，都由本身和邻域内的其他像素值（权值不同）经过加权平均后得到。
调用opencv中的cv2.GaussianBiur（）实现方框滤波处理，其函数原型如下所示：
dst = cv2.GaussianBiur(src,ksize，sigmax[,dst[,sigmay[,borderType]]])
其中，src表示输入图像
ksize表示高斯滤波器模板大小，以（宽度，高度）的形式呈现，常见模糊内核为（3,3），（5,5）
sigmax表示高斯核函数在x方向的高斯内核标准差
sigmay表示高斯核函数在y方向的高斯内核标砖茶。
bordertype表示边框模式，用于推断图像外部像素的某种边界模式，可省略。
"""
"""
7、中值滤波
中值滤波通过计算每一个像素点某邻域范围内所有像素点灰度值的中值，来替换该像素点的灰度值，从而让周围的像素值更接近真实情况，消除孤立的噪声。
中值滤波对脉冲噪声有良好的滤除作用，特别是在滤除噪声的同时，能够保护图像的边缘和细节，使之不被模糊处理，这些优良特性是线性滤波方法所不具有的，从而使其
常常应用于消除图像中的椒盐噪声。
使用cv2.medianBlur（src,ksize[,dst]）
src表示待处理的输入图像
ksize表示内核大小，其值必须是大于1的计数，如3,5,7等
"""
"""
8、双边滤波
双边滤波结合了图像的空间邻近度和像素值相似度（即空间域和值域）的一种折中处理，从而达到保边去噪的目的。双边滤波的优势是
能够做到边缘的保护。
dst = cv2.bilateralFilter(src,s,sigmacolor,sigmaspace[,dst[,bordertype]])
src表示待输入的图像
d表示在过滤期间使用的每个像素邻域的直径。如果这个值设为非正数，则 它会由sigmaspace计算得出。
sigmacolor表示颜色空间的标准方差，该值越大，表明像素邻域内较远的颜色会混合在一起，从而产生更大面积的半相等颜色区域。
sigmaspace表示坐标空间的标准方差。
"""

# --------------------代码实现---------------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("tuxiangchuli.jpg")
img = cv2.resize(img,None,fx=0.2,fy=0.2)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# 均值滤波
result1 = cv2.blur(img,(3,3))
# 方框滤波  不进行归一化处理
result2 = cv2.boxFilter(img,-1,(3,3),normalize=0)
# 方框滤波  进行归一化处理
result3 = cv2.boxFilter(img,-1,(3,3),normalize=1)
# 高斯滤波
result3 = cv2.GaussianBlur(img,(7,7),0)
# 中值滤波
result4 = cv2.medianBlur(img,3)
# 双边滤波
result5 = cv2.bilateralFilter(img,15,150,150)

plt.rcParams['font.sans-serif']=['SimHei']

results = [img,result1,result2,result3,result4,result5]
titles = ['原图','均值滤波','方框滤波-不归一化','方框滤波-归一化','高斯滤波','中值滤波','双边滤波']
for i in range(6):
    plt.subplot(3,2,i+1)
    plt.imshow(results[i],'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()