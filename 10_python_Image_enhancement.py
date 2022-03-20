#-*-conding:utf-8-*-
"""
creaded on 2022年2月12日17:04:17
@anthor Yukuan Zhang

第10章 python图像增强
"""
"""
1、图像增强是将原来不清晰的图像变清晰或强调某些兴趣特征，扩大图像中不同物体特征之间的差别，抑制不感兴趣的特征，改善图像质量，丰富图像信息，
从而加强图像的识别和处理，满足某些特殊分析的需要。
"""
"""
2、图像增强是指按照某种特定的需求，突出图像中有用的信息，去除或者削弱无用的信息。
"""
"""
3、图像增强的目的是使处理后的图像更适合人眼的视觉特性或易于机器识别。
在医学成像、遥感成像、人物摄行、等领域，图像增强技术都有着广泛的应用，图像增强同时可以作为目标识别、目标跟踪、特征点匹配、
图像融合、超分辨率重构等图像处理算法的预处理算法。
"""
"""
4、由于场景条件的影响，很多在高动态范围场景、昏暗环境或特殊光线条件下拍摄的图像视觉效果不佳，需要进行后期增强处理压缩、拉伸动态范围或提取
一致色感才能满足现实和印刷的要求。
"""
"""
6、图像增强通常划分为以下几类，其中最重要的是图像平滑和图像锐化处理。
图像增强：空间域:点运算（灰度变换、直方图修正法）、区域运算（平滑、锐化）
        频率域:高通滤波、低通滤波、同态滤波增强
        彩色增强：假彩色增强、伪彩色增强、彩色变换增强
        代数运算：
"""
"""
7、直方图均衡化
直方图均衡化是图像灰度变化的一个重要处理，广泛应用于图像增强领域。
它是指通过某种灰度映射将原始图像的像素点均匀地分布在每一个灰度级上，其结果将产生一幅灰度级分布概率均衡的图像。
直方图均衡化的中心思想是吧原始图像的灰度直方图从比较集中的某个灰度区间转变为全范围均匀分布的灰度区间，通过该处理，增加了
像素灰度值的动态范围，从而达到增强图像整体对比度的效果。
其核心步骤：
统计直方图中每个灰度级出现的次数
计算累计归一化直方图
重新计算像素点的像素值
"""
"""
8、使用opencv的cv2.equalizeHist()函数实现直方图均衡化。
det = cv2.equalizeHist(src)
"""
#---------------------对灰度图进行全局直方图均衡化处理---------------
"""
import cv2
import matplotlib.pyplot as plt


img = cv2.imread("tuxiangchuli.jpg")
img = cv2.resize(img,None,fx=0.2,fy=0.2)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_gray_hist = cv2.calcHist([img_gray],[0],None,[256],[0,255])
img_gray_eq = cv2.equalizeHist(img_gray)
img_gray_eq_hist = cv2.calcHist(img_gray_eq,[0],None,[256],[0,255])

plt.rcParams['font.sans-serif']=['SimHei']
plt.figure("直方图均衡化",figsize=(8,6))
plt.subplot(2,2,1),plt.imshow(img_gray,'gray'),plt.axis('off'),plt.title("原图")
plt.subplot(2,2,2),plt.plot(img_gray_hist),plt.xlabel("x"),plt.ylabel("y"),plt.title("原图直方图")
plt.subplot(2,2,3),plt.imshow(img_gray_eq,'gray'),plt.axis('off'),plt.title("均衡化")
plt.subplot(2,2,4),plt.plot(img_gray_eq_hist),plt.xlabel("x"),plt.ylabel("y"),plt.title("均衡化后的直方图")
plt.savefig("直方图均衡化")
plt.show()
"""

"""
9、若需要对彩色图像进行全局均衡化处理，则需要分解rgb三色通道，分别进行处理再进行通道合并。
"""
#------------对彩色图像进行全局直方图均衡化处理------------
"""
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("tuxiangchuli.jpg")
img = cv2.resize(img,None,fx=0.2,fy=0.2)
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# 分解三通道
r,g,b = cv2.split(img_rgb)
r_en = cv2.equalizeHist(r)
g_en = cv2.equalizeHist(g)
b_en = cv2.equalizeHist(b)
# 合并三通道
result = cv2.merge((r_en,g_en,b_en))
cv2.imshow("input",img)
cv2.imshow("output",result)
cv2.waitKey()
cv2.destroyAllWindows()
# 画直方图
plt.figure("Hist")
plt.hist(r_en.ravel(),bins=256,density=True,facecolor='r',edgecolor='r')
plt.hist(b_en.ravel(),bins=256,density=True,facecolor='b',edgecolor='b')
plt.hist(g_en.ravel(),bins=256,density=True,facecolor='g',edgecolor='g')
plt.xlabel("x")
plt.ylabel("y")
plt.show()
"""
"""
10、局部直方图均衡化
在opencv中调用函数cv2.createCLAHE()实现对比度受限的局部直方图均衡化。它将
整个图像分成许多小块，那么对每个小块进行均衡化。
其函数原型如下：
retval = cv2.createCLAHE([,clipLimit[,tileGridSize]])
cliplimit表示对比度的大小
tileGridSize表示每次处理块的大小
"""
#------------------灰度图局部直方图均衡化------------------
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("tuxiangchuli.jpg")
img = cv2.resize(img,None,fx=0.2,fy=0.2)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 局部直方图均衡化处理
ckahe = cv2.createCLAHE(clipLimit=2,tileGridSize=(10,10))
# 将灰度图像和局部直方图相关联把直方图均衡化应用到灰度图
result = ckahe.apply(img_gray)
# 显示图像
plt.rcParams['font.sans-serif']=['SimHei']
plt.figure("局部直方图均衡化",figsize=(8,6))
plt.subplot(2,2,1),plt.axis('off'),plt.imshow(img_gray,'gray'),plt.title("原图")
plt.subplot(2,2,2),plt.hist(img_gray.ravel(),bins=256),plt.xlabel("x"),plt.ylabel("y"),plt.title("直方图")
plt.subplot(2,2,3),plt.axis('off'),plt.imshow(result,'gray'),plt.title("局部直方图均衡化后")
plt.subplot(2,2,4),plt.hist(result.ravel(),bins=256),plt.xlabel("x"),plt.ylabel("y"),plt.title("均衡化后的直方图")
plt.show()
"""
#------------------灰度图局部直方图均衡化------------------
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("tuxiangchuli.jpg")
img = cv2.resize(img,None,fx=0.2,fy=0.2)
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# 局部直方图均衡化处理
ckahe = cv2.createCLAHE(clipLimit=2,tileGridSize=(10,10))
# 分解三通道
r,g,b = cv2.split(img_rgb)
# 将灰度图像和局部直方图相关联把直方图均衡化应用到灰度图
b_ = ckahe.apply(b)
g_ = ckahe.apply(g)
r_ = ckahe.apply(r)
# 融合三通道图像
result = cv2.merge((r_,g_,b_))
# 显示图像
plt.rcParams['font.sans-serif']=['SimHei']
plt.figure("局部直方图均衡化",figsize=(8,6))
plt.subplot(2,2,1),plt.axis('off'),plt.imshow(img_rgb,'gray'),plt.title("原图")
plt.subplot(2,2,2),plt.hist(img_rgb.ravel(),bins=256),plt.xlabel("x"),plt.ylabel("y"),plt.title("直方图")
plt.subplot(2,2,3),plt.axis('off'),plt.imshow(result,'gray'),plt.title("局部直方图均衡化后")
plt.subplot(2,2,4),plt.hist(result.ravel(),bins=256),plt.xlabel("x"),plt.ylabel("y"),plt.title("均衡化后的直方图")
plt.savefig("彩色局部直方图均衡化")
plt.show()
"""
"""
11、自动色彩均衡化
Retinex算法是代表性的图像增强算法，它根据人的视网膜和大脑皮质模拟对物体颜色的波长光线反射能力而形成，对复杂环境下的一维条形码具有一定
范围内的动态压缩，对图像边缘有着一定自适应的增强。
"""
"""
12、自动色彩均衡（ACE）算法是在Retinex算法的理论上提出的，它通过计算图像目标像素点和周围像素点的明暗程度及其关系来对最终的像素值进行校正，
实现图像的对比度调整，产生类似人体视网膜的色彩恒常性和亮度恒常性的均衡，具有很好的图像增强效果。
ACE算法包括两个步骤：
一是对图像进行色彩和空域调整，完成图像的色差校正，得到空域重构图像：二是对校正后的图像进行动态扩展。

"""
"""
13、ACE算法能够进行图像去雾处理，增强图像细节
由于opencv中暂时没有ACE算法包，下面是实现彩色直方图均衡化的代码,代码有BUG
"""

#--------------------ACE算法-------------------------

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# 线性拉伸处理
#去掉最大最小0.5%的像素值线性拉伸至【0,1】

def stretchImage(data,s=0.005,bins=2000):
    ht = np.histogram(data,bins)
    d = np.cumsum(ht[0]) / float(data.size)
    lmin =0
    lmax=bins-1
    while lmin<bins:
        if d[lmin]>=s:
            break
        lmin+=1
    while lmax >= 0:
        if d[lmax]<=1-s:
            break
            lmax-=1
    return  np.clip((data-ht[1][lmin])/(ht[1][lmax]-ht[1][lmin]),0,1)
# 根据半径计算权重参数矩阵
g_para = {}
def getpara(radius=5):
    global g_para
    m = g_para.get(radius,None)
    if m is not None:
        return  m
    size = radius*2+1
    m = np.zeros((size,size))
    for h in range(-radius,radius+1):
        for w in range(-radius,radius+1):
            if h==0 and w==0:
                continue
            m[radius+h,radius+w]=1.0/math.sqrt(h**2+w**2)
    m/=m.sum()
    g_para[radius]=m
    return
#常规ACE实现
def zmIce(I,radio=4,radius=300):
    para = getpara(radius)
    height,width = I.shape
    zh,zw = [0]*radius + list(range(height))+[height-1]*radius,[0]*radius+list(range(width))+[width-1]*radius
    Z = I[np.ix_(zh,zw)]
    res = np.zeros(I.shape)
    for h in range(int(radius*2+1)):
        for w in range(int(radius*2+1)):
            if para[h][w]==0:
                continue
            res+=(para[h][w]*np.clip((I-Z[h:h+height,w:w+width])*radio,-1,1))
    return res
#单通道ACE快速增强实现
def zmIceFast(I,ratio,radius):
    height,width = I.shape[:2]
    if min(height,width)<=2:
        return np.zeros(I.shape)+0.5
    Rs = cv2.resize(I,((width+1)//2,(height+1)//2))
    Rf = zmIceFast(Rs,ratio,radius)  #递归调用
    Rf = cv2.resize(Rf,(width,height))
    Rs = cv2.resize(Rs,(width,height))
    return Rf+zmIce(I,ratio,radius)-zmIce(Rs,ratio,radius)
#RGB三通道分别增强，ratio是对比度增强因子，radius是卷积模板半径
def zmIceColor(I,ratio=4,radius=3):
    res = np.zeros(I.shape)
    for k in range(3):
        res[:,:,k] = stretchImage(zmIceFast(I[:,:,k],ratio,radius))
    return  res
#主函数

if __name__=='__main__':
    img = cv2.imread("tuxiangchuli.jpg")
    img = cv2.resize(img,None,fx=0.2,fy=0.2)
    res = zmIceColor(img/255.0)*255
    cv2.imshow("ddd",np.uint8(res))
    cv2.waitKey()
    cv2.destroyAllWindows()
