#-*-conding:utf-8-*-
"""
Creaded on 2022年2月21日19:05:06
@Anthor Yukuan Zhang

第十二章： python图像分割
2022年2月21日我写了一篇数字图像处理的结课论文，课题名称为：图像分割算法和基于深度学习的语义分割算法研究，探讨了基于阈值的图像分割、基于边缘检测的图像分割、基于纹理背景的图像分割
和基于K-Means聚类的区域分割、基于均值飘逸算法的图像分割、基于分水岭算法的图像分割和基于深度学习的语义分割。
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

"""
1、基于阈值的图像分割见第七章
"""
"""
1、基于边缘检测的图像分割
图像中的相邻区域之间的像素集合共同构成了图像的边缘。基于边缘检测的图像分割方法是通过确定图像中的边缘轮廓像素，然后将这些像素连接起来
构建区域边界的过程。由于沿着图像边缘走向的像素值变化比较平缓，而沿着垂直与边缘走向的像素值变化比较大，所以通常会采用一阶导数和二阶导数来描述与检测边缘。
下面的代码是对比常用的微分算子，如Roberts、Prewitt、Sobel、Laplacian、Scharr、Canny、LOG等。
"""
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("tuxiangchuli.jpg")
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
binary = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

# 阈值处理
ret,binary = cv2.threshold(binary,127,255,cv2.THRESH_BINARY)

# Roberts算子
kernelx = np.array([[-1,0],[0,1]],dtype=int)
kernely = np.array([[0,-1],[1,0]],dtype=int)
x = cv2.filter2D(binary,cv2.CV_16S,kernelx)
y = cv2.filter2D(binary,cv2.CV_16S,kernely)
absx = cv2.convertScaleAbs(x)
absy = cv2.convertScaleAbs(y)
roberts = cv2.addWeighted(absx,0.5,absy,0.5,0)
# sobel算子
x = cv2.Sobel(binary,cv2.CV_16S,1,0)
y = cv2.Sobel(binary,cv2.CV_16S,0,1)
absx = cv2.convertScaleAbs(x)
absy = cv2.convertScaleAbs(y)
sobel = cv2.addWeighted(absx,0.5,absy,0.5,0)
# prewitt算子
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]],dtype=int)
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]],dtype=int)
x = cv2.filter2D(binary,cv2.CV_16S,kernelx)
y = cv2.filter2D(binary,cv2.CV_16S,kernely)
absx = cv2.convertScaleAbs(x)
absy = cv2.convertScaleAbs(y)
prewitt = cv2.addWeighted(absx,0.5,absy,0.5,0)
# laplacian算子
dst = cv2.Laplacian(binary,cv2.CV_16S,ksize=3)
laplacian = cv2.convertScaleAbs(dst)
# scharr算子
x = cv2.Scharr(binary,cv2.CV_32F,1,0)
y = cv2.Scharr(binary,cv2.CV_32F,0,1)
absx = cv2.convertScaleAbs(x)
absy = cv2.convertScaleAbs(y)
scharr = cv2.addWeighted(absx,0.5,absy,0.5,0)
# canny算子
gaussianblur = cv2.GaussianBlur(binary,(3,3),0)
canny = cv2.Canny(gaussianblur,50,150)
# log算子
gaussianblur = cv2.GaussianBlur(binary,(3,3),0)
dst = cv2.Laplacian(gaussianblur,cv2.CV_16S,ksize=3)
log = cv2.convertScaleAbs(dst)
# 使用findcontours联合drawcontours
ret,binary = cv2.threshold(binary,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
contours,hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_rgb,contours,-1,(0,255,0),1)

# 展示结果
titles = ["Roberts","Sobel","Presitt","Laplacian","Scharr","Canny","LOG","FinDraw"]
images = [roberts,sobel,prewitt,laplacian,scharr,canny,log,img_rgb]
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.savefig("bianyuan")
plt.show()
"""
"""
2、基于K-Means聚类的区域分割
k-Means聚类是最常用的聚类算法，起源于信号处理，其目标是将数据点划分为K个类簇，找到每个簇的中心并使其度量最小化。该算法的最大优点是简单、便于理解，运算速度较快，
缺点是只能应用于连续型数据，并且要在聚类前指定聚集的类簇数。
下面是K-Means聚类算法的分析流程，步骤如下：
第一步，确定K值，即将数据集聚集成K个类簇或小组
第二步，从数据集中随机选择K个数据点作为质心（centroid）或数据中心。
第三步，分别计算每个点到每个质心之间的距离，并将每个点划分到距离最近质心的小组，跟定了那个质心。
第四步，当每个质心都聚集了一些点后，重新定义算法选出新的质心。
第五步，比较新的质心和老的质心，如果新质心和老质心之间的距离小于某一个阈值，则表示重新计算的质心位置变化不大，收敛稳定，则认为聚类已经达到了期望的结果，算法终止。
第六步，如果新的质心和老的质心变化很大，即距离大于阈值，则继续迭代执行第三步到第五步，直到算法终止。
在图像处理中，通过K-Means聚类算法可以实现图像分割、图像聚类、图像识别等操作，本实验主要用来进行图像颜色分割。假设存在一张100像素×100像素的灰度图像，它由10000个RGB灰度级组成，
通过K-Means可以将这些像素点聚类成K个簇，然后使用每个簇内的质心来替换簇内所有的像素点，这样就能实现在不改变分辨率的情况下量化压缩图像颜色，实现图像颜色层级分割。

调用opencv中的kmeans（）函数实现：
 kmeans(data: Any, 表示聚类数据，最好是np.float32类型的N维点集。
           K: Any,表示聚类类簇数
           bestLabels: Any,表示输出的整数数组，用于存储每个样本的聚类标签索引
           criteria: Any,表示算法终止条件，即最大迭代次数或所需要的精度。在某些迭代中，一旦每个簇中心的移动小于criteria.epsilon，算法就会停止。
           attempts: Any,表示重复实验kmeans算法的次数，算法返回产生最佳紧凑性的标签。
           flags: Any,表示初始中心的选择,cv2.KMEANS_RANDOM_CENTERS 或 cv2.KMEANS_PP_CENTERS

           centers: Any = None) -> None表示集群中心的输出矩阵，每个集群中心为一行数据。

下面是使用该方法对灰度图像颜色进行图像分割处理，需要注意，在进行kmeans聚类操作之前，需要将rgb像素点转换为一维的数组，再将各形式的颜色聚集在一起，形成最终的颜色分割。
"""
#--------------------kmeans 对灰度图进行分割-----------------

"""
import cv2
import numpy as np

img = cv2.imread("tuxiangchuli.jpg")
img = cv2.resize(img,None,fx=0.2,fy=0.2)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
rows,cols = img_gray.shape # 图像的宽高0

# 图像二维数组转换为一维,类型为float32
data = np.float32(img_gray.reshape((rows * cols,1)))

#定义终止条件
criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
#设置初始中心的选择
flags = cv2.KMEANS_RANDOM_CENTERS
# kmeans聚类 ，聚集成4类
compactness,labels,centers = cv2.kmeans(data,4,None,criteria,10,flags) # labels是将一幅图片分成了4类，用0,1,2,3表示不同的类，生成的元素个为图像的长*宽，二维的。centers为最终每个类的中心点像素值


# 生成最终图像
dst = labels.reshape((rows,cols))
plt.rcParams['font.sans-serif']=['SimHei']
plt.figure("K-means",figsize=(8,6))
plt.subplot(1,2,1),plt.imshow(img_gray,'gray'),plt.axis('off'),plt.title("原图")
plt.subplot(1,2,2),plt.imshow(dst,'gray'),plt.axis('off'),plt.title("k-means")
plt.show()
"""


#-------------kmeans对彩色图像分割---------------------
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("tuxiangchuli.jpg")
img = cv2.resize(img,None,fx=0.1,fy=0.1)
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# 图像二维转换为一维
data = img.reshape((-1,3))
data = np.float32(data)
#定义中心
criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
# 设置标签
flags = cv2.KMEANS_RANDOM_CENTERS
# 聚集成2类
compactness,labels2,centers2 = cv2.kmeans(data,2,None,criteria,10,flags)
# 聚集成4类
compactness,labels4,centers4 = cv2.kmeans(data,4,None,criteria,10,flags)
# 聚集成8类
compactness,labels8,centers8 = cv2.kmeans(data,8,None,criteria,10,flags)
# 聚集成16类
compactness,labels64,centers64 = cv2.kmeans(data,64,None,criteria,10,flags)
# 图像转换回uint8二维类型
centers2 = np.uint8(centers2)
res = centers2[labels2.flatten()]
dst2 = res.reshape((img.shape))

centers4 = np.uint8(centers4)
res = centers4[labels4.flatten()] # labels.flatten()表示将二维的labels降为一维，centers[labels.flatten()]表示将centers依据labels扩展元素个数
dst4 = res.reshape((img.shape))

centers8 = np.uint8(centers8)
res = centers8[labels8.flatten()]
dst8 = res.reshape((img.shape))

centers64 = np.uint8(centers64)
res = centers64[labels64.flatten()]
dst64 = res.reshape((img.shape))

# 图像转换为rgb显示
dst2 = cv2.cvtColor(dst2,cv2.COLOR_BGR2RGB)
dst4 = cv2.cvtColor(dst4,cv2.COLOR_BGR2RGB)
dst8 = cv2.cvtColor(dst8,cv2.COLOR_BGR2RGB)
dst16 = cv2.cvtColor(dst64,cv2.COLOR_BGR2RGB)

plt.rcParams['font.sans-serif'] = ['SimHei']
titles = ["K=2","K=4","K=8","K=64"]
imgs = [dst2,dst4,dst8,dst64]
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(imgs[i],'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.savefig("K聚类")
plt.show()
"""

"""
3、基于均值飘逸算法的图像分割
均值飘逸（mean shift）算法是一种通用的聚类算法，最早是FUkunaga于1975年提出的。它是一种无参估计算法，沿着概率梯度的上升方向寻找分布的峰值。
均值飘逸算法先算出当前点的偏移均值，移动该点到其偏移均值，然后以此为新的起始点，继续移动，直到满足一定的条件结束。
图像分割中可以利用均值飘逸算法的特性，实现彩色图像分割。在OpenCV中提供的函数为pyrMeanShiftFiltering（），
该函数严格来说并不是图像分割，而是图像在色彩层面的平滑滤波，可以中和色彩分布相近的颜色，平滑色彩细节，侵蚀掉面积较小的颜色区域。
均值飘逸函数pyrMeanShiftFiltering（）的执行过程如下：
第一步，构建迭代空间。以输入图像上任一点P0为圆心，建立以sp为物理空间半径，sr为色彩空间半径的球形空间，物理空间上坐标为x和y，
色彩空间上坐标为RGB或HSV，构成一个空间球体。其中x和y表示图像的长和宽，色彩空间R,G,B在0-255。
第二步，求迭代空间的向量并移动迭代空间球体重新计算向量，直至收敛。在上一步构建的球形空间中，求出所有点相对于中心点的色彩向量之和，
移动迭代空间的中心点到该向量的终点，并再次计算该球形空间中所有点的向量之和，如此迭代，直到在最后一个空间球体中所求得向量和的终点就是该空间球体
的中心点Pn，迭代结束。
第三步，更新输出图像上对应的初始原点P0的色彩值为本轮迭代的终点Pn的色彩值，完成一个点的色彩均值飘逸。
第四步，对输入图像上的其他点，依次执行以上三个步骤，直至遍历完所有点，整个均值偏移色彩滤波完成。

使用cv2.pyrMeanShiftFiltering()
pyrMeanShiftFiltering(src: Any, 输入图像，八位三通道的彩色图像
                          sp: Any, 表示定义飘移物理空间半径的大小
                          sr: Any,表示定义飘移彩色色彩空间半径的大小
                          dst: Any = None,表示输出图像
                          maxLevel: Any = None, 定义金字塔的最大成熟
                          termcrit: Any = None) -> None表示定义的飘移迭代终止条件
"""

# ----------------------------------均值飘移分割-------------------------------------
"""
import cv2

img = cv2.imread("tuxiangchuli.jpg")
img = cv2.resize(img,None,fx=0.2,fy=0.2)
spa = 50  # 空间窗口大小
colo = 50 # 色彩窗口大小
maxp = 2 # 金字塔层数

dst = cv2.pyrMeanShiftFiltering(img,spa,colo,maxp)
cv2.imwrite("piaoyi.jpg",dst)
cv2.imshow("jg",dst)
cv2.waitKey()
cv2.destroyAllWindows()
"""
"""
4、基于纹理背景的图像分割
本节主要介绍基于图像纹理信息（颜色）、边界信息（反差）和背景信息的图像分割算法。
在opencv中Grabcut算法能够有效地利用纹理信息和边界信息分割背景，提取图像目标物体。
 mask,bgdmodel,fgdmodel=grabCut(img: Any,表示输入图像，为8位三通道图像
            mask: Any,表示蒙版图像，输入|输出的8位单通道掩码，确定前景区域、背景区域、不确定区域
            rect: Any,表示前景对象的矩形坐标，其基本格式为（x,y,w,h），分别为左上角坐标和宽度，高度
            bgdModel: Any,表示后台模型使用的数组，通常设置大小为（1,65）,np.float64
            fgdModel: Any,表示前台模型使用的数组，通常设置大小为（1,65）,np.float64
            iterCount: Any,表示算法运行的迭代次数
            mode: Any = None) -> None

实现方法：https://zhuanlan.zhihu.com/p/85813603
首先通过调用np.zeros（）函数创建mask、fgbmodel、bdgmodel，然后定义rect矩形范围，调用grabcut（）函数实现图像分割。
由于该方法会修改掩码，像素会被标记为不同的标志来指明它们是背景或前景。最后将所有的0和2像素点赋值为0（背景），而所有的1和3像素点赋值为1（前景）
"""
#--------------纹理背景分割--------------------
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("tuxiangchuli.jpg")
img = cv2.resize(img,None,fx=0.2,fy=0.2)
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# 创建mask、fgbmodel、bdgmodel
mask = np.zeros(img.shape[:2],dtype='uint8')
#mask = np.zeros(img.shape[:2],dtype="uint8")
fbgmodel = np.zeros((1,65),dtype='float64')
bdgmodel = np.zeros((1,65),dtype='float64')

#矩形坐标
r = cv2.selectROI("input",img_rgb,False) # selectroi调用之后，会生成一个窗口，用户可以在上面画出感兴趣的目标矩形，返回（x,y,w,h）
print("提取的感兴趣的待分割目标的坐标=",r)
roi = img_rgb[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] # 根据人工提取的目标坐标，将目标从原始图中抠出来
rect = r # 包括前景的矩形，格式为(x,y,w,h)

#图像分割
cv2.grabCut(img_rgb,mask,rect,bdgmodel,fbgmodel,5,cv2.GC_INIT_WITH_RECT) # 生成的mask值包括0,1,2,3
print(mask.min(),mask.max())
#设置掩码0,和2作为背景，1和3作为前景
mask2 = np.where((mask==0)|(mask==2),0,1).astype('uint8')  # 将mask中所有是0或2的元素变成0，其他的变成1
print(mask2.min(),mask2.max())

#使用模板来获取前景区域
img_mask = img_rgb * mask2[:,:,np.newaxis] # np.newaxis增加一维

plt.rcParams['font.sans-serif']=['SimHei']
plt.figure("基于纹理分割",figsize=(8,6))
plt.subplot(1,2,1),plt.imshow(img_rgb),plt.axis('off'),plt.title("原图")
plt.subplot(1,2,2),plt.imshow(img_mask),plt.axis('off'),plt.title("分割图"),plt.colorbar()

plt.show()
"""
"""
基于分水岭算法的图像分割
图像分水岭算法是将图像的边缘轮廓转换为山脉，将均匀区域转换为山谷，从而提升分割效果的算法。分水岭算法是基于拓扑理论的数学形态学的分割方法，灰度图像根据灰度值把像素之间的关系看成山峰和山谷
的关系，高亮度（灰度值高）的地方是山峰，低亮度的地方是山谷。然后给每个孤立的山谷（局部最小值）不同颜色的水（label），当水涨起来时，根据周围的山峰（梯度），不同的山谷也就是不同颜色的像素点开始合并，为了避免这个现象，
可以在水要合并的地方建立障碍，直到所有山峰都被淹没。所创建的障碍就是分割结果，这就是分水岭的原理。

分水岭算法的计算过程是一个迭代标注过程，主要包括排序和淹没两个步骤。由于图像会存在噪声或缺失等问题，该方法会造成过度分割。
opencv提供了watershed（）函数实现图像分水岭算法，并且能够指定需要合并的点，其函数原型如下：
markers = watershed(image: Any, 表示输入图像，需要为8位三通道的彩色图像
              markers: Any) -> None 表示用于存储函数调佣之后的运算结果，输入/输出32位单通道图像的标记结构，输出结果需与输入图像的尺寸和类型一致。

实现步骤：
1、通过图像灰度化和阈值化处理提取图像灰度轮廓，采用OTSU二值化处理获取硬币的轮廓
2、通过形态学开运算过滤掉小的白色噪声。
3、创建标记变量
4、调用watershed（）函数实现分水岭图像分割
"""
# ----------------分水岭算法---------------------
"""
import cv2

img = cv2.imread("tuxiangchuli.jpg")
img = cv2.resize(img,None,fx=0.2,fy=0.2)
# 灰度化
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 阈值化
ret,thresh = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# 开运算消除噪声
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)
# 图像膨胀操作确定背景区域
sure_bg = cv2.dilate(opening,kernel,iterations=3)
#距离运算确定前景区域
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret,sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
#寻找未知区域
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
#标记变量
ret,markers = cv2.connectedComponents(sure_fg)
#所有标签加一，以确定背景不是0而是1
markers= markers+1
#用0标记未知区域
markers[unknown==255]=0
# 分水岭算法实现图像分割
markers = cv2.watershed(img,markers)
img[markers==-1]=[255,0,0]
plt.rcParams['font.sans-serif']=['SimHei']
plt.figure("分水岭分割算法")
plt.imshow(img,'gray')
plt.axis('off')
plt.show()
"""

"""
案例：文字区域定位及提取
步骤：
1、读取文字原始图像，并利用中值滤波算法消除图像噪声，同时保留图像边缘细节。
2、通过图形灰度转换将中值滤波处理后彩色图像转换为灰度图像
3、采用sobel算子锐化突出文字图像的边缘细节，改善图像的对比度，提取文字轮廓
4、经过二值化处理提取图像中的文字区域，将图像的背景和文字区域分离
5、将阈值化处理后的图像进行膨胀处理和腐蚀处理，突出图像轮廓的同时过滤掉图像的细节
6、采用findcontours函数寻找文字轮廓，定位并提取目标文字，然后调用drawcontours函数绘制相关轮廓，输出最终图像
"""
# ----------------文字提取---------------------

import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("wenzi.png")
img_ = cv2.resize(img,None,fx=1.2,fy=1.2)
print(img.shape)
print(img_.shape)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# 中值滤波去除噪声
median = cv2.medianBlur(img,3)
#灰度图
img_gray = cv2.cvtColor(median,cv2.COLOR_BGR2GRAY)
# sobel算子锐化
img_sobel_x = cv2.Sobel(img_gray,cv2.CV_8U,1,0,ksize=3)
img_sobel_y = cv2.Sobel(img_gray,cv2.CV_8U,0,1,ksize=3)
absx = cv2.convertScaleAbs(img_sobel_x)
absy = cv2.convertScaleAbs(img_sobel_y)
img_sobel = cv2.addWeighted(absx,0.5,absy,0.5,0)
# 二值化
ret,bianry = cv2.threshold(img_sobel,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)

# 膨胀 腐蚀
# 设置膨胀和腐蚀操作的核函数
element1 = cv2.getStructuringElement(cv2.MORPH_RECT,(30,9))
element2 = cv2.getStructuringElement(cv2.MORPH_RECT,(24,6))
#膨胀突出轮廓
dilatipn = cv2.dilate(bianry,element2,iterations=1)
#腐蚀去掉细节
erosion = cv2.erode(dilatipn,element1,iterations=1)

#查找文字轮廓
region = []
contours,hierarchy = cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#筛选面积
for i in range(len(contours)):
    # 遍历所有轮廓
    cont = contours[i]
    #计算轮廓面积
    area = cv2.contourArea(cont)
    #寻找最小矩形
    rect = cv2.minAreaRect(cont)
    # 轮廓的四个点坐标
    box = cv2.boxPoints(rect)
    box = np.uint0(box)
    #计算高和宽
    height = abs(box[0][1]-box[2][1])
    width = abs(box[0][0]-box[2][0])
    #过滤太细矩形
    if(height>width*1.5):
        continue
    region.append(box)
# 定位的文字用绿线绘制轮廓
for box in region:
    print(box)
    cv2.drawContours(img,[box],0,(0,255,0),2)
#显示图像

imgs = [median,img_gray,img_sobel,bianry,dilatipn,erosion,img]
titles = ["去噪","灰度化","sobel算子锐化","二值化","膨胀","腐蚀","结果"]
plt.rcParams['font.sans-serif']=['SimHei']
plt.figure("提取文字案例",figsize=(21,6))
for i in range(7):
    plt.subplot(2,4,i+1)
    plt.imshow(imgs[i],'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()