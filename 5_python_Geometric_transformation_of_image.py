#-*-coding:utf-8-*-
"""
Creaed on 2022年1月11日21:16:45
@author Yukuan Zhang
第五章： python 图像几何变换
"""

"""
1、图像几何变换：是在不改变图像内容的情况下，对图像像素进行空间几何变换的处理方式。它将一幅图像中的坐标位置映射到另一幅图像的新坐标位置，
其实质是改变像素的空间位置，估算新空间位置上的像素。
图像的几何变换不改变图像的像素值，只是在图像平面上进行像素的重新安排。
几何变换通常作为图像处理应用的预处理步骤，是图像归一化的核心工作之一。
图像的几何变换主要包括：图形平移变换、图像缩放变换、图像旋转变换、图像镜像变换、图像放射变换、图像透视变换等。
"""

"""
2、对于数字图像而言，像素的坐标是离散型非负整数，但是在进行变换的过程中有可能产生浮点坐标值。这在图像处理中是一个无效的坐标。
为了解决这个问题需要用到插值算法。常见的插值有：最近邻插值法、双线性插值法、双立方插值法等。

1）最近邻插值：浮点坐标的像素值等于距离该点最近的输入图像的最近的输入图像的像素值。
2）双线性插值：双线性插值的主要思想是计算出浮点坐标像素近似值。
  一个浮点坐标必定会被四个整数坐标包围，将这四个整数坐标的像素值按照一定的比例混合就可以求出浮点坐标的像素值。
3）双立方插值：双立方插值是一种更加复杂的插值方式。
能创造出比双线性插值更平滑的图像边缘。在图像处理中，双立方插值计算设计周围16个像素点，插值后的坐标点是原图中邻近16个像素点的权重卷积之和。

"""

"""
3图像平滑变换
图像平滑变换试讲图像中的所有像素点按照给定的平移量进行水平或者垂直方向上的移动。

图像平移首先定义平移矩阵M，然后调用warpAffine（）函数实现平移，
核心函数如下：
M = np.float32([[1,0,x],[0,1,y]])
其中，M表示为平移矩阵，x表示水平偏移量，y表示垂直平移量。
shifted = cv2.warpAffine(src,M,dsize[,dst[,flage[,borderMode[,borderValue]]]])
其中，src表示原始图像
M表示平移矩阵
dsize表示变换后的输出图像的尺寸大小
dst表示输出图像，其大小为dsize，类型与src相同。
flags表示插值方法的组合和可选值。
borderMode表示像素外推法，当bordermode=BORDER_TRANSPARENT时，表示目标图像中的像素不会修改原图像中的“异常值”。
bordervalue用于边界不变的情况，默认情况下为0.
"""

#---------图像平移：垂直向下平移50像素，水平向右平移100像素-------------
"""
import cv2
import numpy as np
img = cv2.imread("tuxiangchuli.jpg")
M = np.float32([[1,0,-100],[0,1,-50]])  # 正数：水平向右，垂直向下平移。负数：水平向左，垂直向上平移
rows,cols = img.shape[:2]
resulet = cv2.warpAffine(img,M,(cols,rows))
cv2.imshow("pingyi", resulet)
cv2.imwrite("pingyi.jpg",resulet,[int(cv2.IMWRITE_JPEG_QUALITY),100])
cv2.waitKey()
cv2.destroyAllWindows()

"""
#-------图像上下左右平移----------
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
imgs = cv2.imread("tuxiangchuli.jpg")
img = cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)
M1 = np.float32([[1,0,0],[0,1,100]]) # 垂直向下平移100
M2 = np.float32([[1,0,0],[0,1,-100]]) # 垂直向上平移100
M3 = np.float32([[1,0,100],[0,1,0]]) # 水平向右平移100
M4 = np.float32([[1,0,-100],[0,1,0]]) # 水平向左平移100

img1 = cv2.warpAffine(img,M1,(img.shape[1],img.shape[0]))
img2 = cv2.warpAffine(img,M2,(img.shape[1],img.shape[0]))
img3 = cv2.warpAffine(img,M3,(img.shape[1],img.shape[0]))
img4 = cv2.warpAffine(img,M4,(img.shape[1],img.shape[0]))

titles = ['xia','shang','you','zuo']
images = [img1,img2,img3,img4]
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()
"""

"""
4、图像缩放变换

图像缩放变换是指对数字图像的大小进行调整的过程。
result = cv2.resize(src,dsize[,result[,fx[,fy[,interpolation]]]])
其中，src表示原始图像。
dsize表示图像缩放的大小。
result表示输出图像的结果
fx表示图像x轴方向缩放大小的倍数。
fy表示图像y轴方向缩放大小的倍数。
interpolation表示变换方法。CV_INTER_NN表示最近邻插值，CV_INTER_LINEAR表示双线性插值（缺省使用），
CV_INTER_AREA表示使用像素关系重采样，当图像缩小时，该方法可以避免波纹出现，当图像放大时CV_INTER_CUBIC表示立方插值。

常用的缩放方法：
result = cv2.resize(src,size)
result = cv2.resize(src,None,fx=0.5,fy=0.5)表示将x轴y轴个缩小50%
"""

#-----------缩放------------------
"""
import cv2
import numpy as np

img = cv2.imread("tuxiangchuli.jpg")
rols,wols = img.shape[:2]
print(rols,wols)
result = cv2.resize(img,(int(wols/4),int(rols/4)))
result1 = cv2.resize(img,None,fx=0.2,fy=0.2)
print(result1.shape[:2])
cv2.imshow("suoxiao",result)
cv2.imshow("suoxiao1",result1)
cv2.waitKey()
cv2.destroyAllWindows()

"""

"""
5、图像旋转变换

图像旋转是图形以某一点为中心旋转一定的角度，形成一幅心的图像的过程。
图像旋转变换会有一个旋转中心，这个旋转中心一般为图像的中心，旋转之后图像的大小一般会发生改变。
图像旋转通过调用以下函数：
M = cv2.getRotationMatrix2D(center,angle,scale)
其中，center表示旋转中心点，通常设置为（cols/2，rows/2）
angle表示旋转角度，正值表示逆时针旋转，坐标原点被定为左上角。
scale表示比例因子

rotated = cv2.warpAffine(src,M,(cols,rows))
src表示原始图像，
M表示旋转参数
(cols,rows)表示变换后的图像的宽和高或列数和行数
"""

#--------------图像旋转---------------
"""
import cv2
img = cv2.imread("tuxiangchuli.jpg")
img = cv2.resize(img,None,fx=0.2,fy=0.2)
rows,cols = img.shape[:2]
M = cv2.getRotationMatrix2D((int(cols/2),int(rows/2)),-20,1) # -20表示顺时针旋转20度，若是20则表示逆时针旋转20度
result = cv2.warpAffine(img,M,(cols,rows))
cv2.imshow("xuanzhuan",result)
cv2.waitKey()
cv2.destroyAllWindows()
"""

"""
6、图像镜像：
水平镜像：通常是以原始图像的垂直中轴线为中心，将图像分为左右两部分进行对称变换。
垂直镜像：通常是以原始图像的水平中轴线为中心，将图像划分为上下两部分对称变换。

dst = cv2.flip(src,flipcode)
其中，src表示原始图像
flipcode表示翻转方向，如果flipcode=0，则以X轴镜像，>0，以Y轴镜像，<0，以X,Y镜像

"""

#----------图像镜像-------------
"""
import cv2
import matplotlib.pyplot as plt


img = cv2.imread("tuxiangchuli.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img = cv2.resize(img,None,fx=0.2,fy=0.2)
relstX = cv2.flip(img,0)
relstY = cv2.flip(img,1)
relstXY = cv2.flip(img,-1)

titiles = ['yuantu','x','y','xy']
imgs = [img,relstX,relstY,relstXY]

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(imgs[i])
    plt.title(titiles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()
"""

"""
图像放射变换
图像放射变换又称为图像放射映射，是指在几何中，一个向量空间进行一次线性变换并接上一个平移，变换为另一个向量空间。

核心函数吐下：
pos1 = np.float32([[],[],[]])
pos2 = np.float32([[],[],[]])
M = cv2.getffineTransform(pos1,pos2) 其中，pos1,2为变换前后的三个点对应关系
result = cv2.warpAffine(src,M,(cols,rows))
"""

#--------------放射变化----------------
"""
import cv2
import numpy as np

img = cv2.imread("tuxiangchuli.jpg")
img = cv2.resize(img,None,fx=0.2,fy=0.2)
pos1 = np.float32([[100,50],[25,66],[88,96]])
pos2 = np.float32([[200,50],[50,23],[44,66]])
M = cv2.getAffineTransform(pos1,pos2)
result = cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))
cv2.imshow("xuanzhuan",result)
cv2.waitKey()
cv2.destroyAllWindows()
"""

"""
图像透视：
图像透视变换的本质是将图像投影到一个新的视平面，

核心函数吐下：
pos1 = np.float32([[],[],[],[]])
pos2 = np.float32([[],[],[],[]])
M = cv2.getffineTransform(pos1,pos2) 其中，pos1,2为变换前后的四个点对应关系
result = cv2.warpPerspective(src,M,(cols,rows))
"""

#-------------透视变换--------------
"""
import cv2
import numpy as np

img = cv2.imread("tuxiangchuli.jpg")
img = cv2.resize(img,None,fx=0.2,fy=0.2)
print(img.shape)
pos1 = np.float32([[120,132],[120,200],[200,132],[200,132]])
pos2 = np.float32([[0,0],[200,0],[0,200],[200,200]])
M = cv2.getAffineTransform(pos1,pos2)
result = cv2.warpPerspective(img,M,(img.shape[1],img.shape[0]))
cv2.imshow("toushi",result)
cv2.waitKey()
cv2.destroyAllWindows()
"""