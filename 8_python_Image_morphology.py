#-*-conding:utf-8-*-
"""
Creaded on 2022年2月9日11:16:49
@anthor Yukuan Zhang

第8章 python 图像形态学处理
"""
"""
1、数学形态学的应用可以简化图像数据，保持它们基本的形状特征，并除去不相干的结构。
数学形态学的算法有天然的并行实现结构，主要针对的是二值图像（0或1）。在图像处理方面，二值形态学经常应用于图像分割、细化、抽取骨架、边缘提取、形状分析
、角点检测、分水岭算法等。
"""
"""
2、常见的图像形态学运算包括腐蚀、膨胀、开运算、闭运算、梯度运算、顶帽运算和底帽运算等。
"""
"""
3、morphologyEx（）函数，能利用基本的膨胀和腐蚀技术，来执行更加高级的形态学变换，如开闭运算、形态学梯度、顶帽、黑帽等，也可以实现
最基本的图像膨胀和腐蚀。
cv2.morphologyEx(src,model,kernel)
其中，sec表示原始图像
model表示图像进行形态学处理，包括以下几类：
cv2.MORPH_OPEN:开运算
cv2.MORPH_CLOSE:闭运算
cv2.MORPH_GRADIENT:形态学梯度
cv2.MORPH_TOPHAT:顶帽运算
cv2.MORPH_BLACKHAT:黑帽运算
kernel表示卷积核，可以用numpy.ones()函数构建。
"""

"""
4、图像腐蚀和膨胀是两种基本的形态学运算，主要用来寻找图像中的极小区域和极大区域。
图像被腐蚀处理后，将去除噪声，但同时会压缩图像，而图像膨胀操作可以去除噪声，并保持原有形状。
"""
"""
5、图像腐蚀类似于“领域被蚕食”，它将图像中的高亮区域或白色部分进行缩减细化。
设A,B为集合，A被B的腐蚀，记作A-B，表示图像A用卷积模板B来进行腐蚀处理，通过模板B与图像A进行卷积计算，得出B覆盖区域的像素点最小值，
并用这个最小值来替代参考点的像素值。
图像腐蚀主要包括二值图像和卷积核两个输入对象，卷积核是腐蚀中的关键数组，采用Numpy库可以生成。
卷积核的中心点逐个像素扫描原始图像，被扫描到的原始图像中的像素点，只有当卷积核对应的元素值均为1时，其值才为1，否则将其像素值修改为0。

"""
"""
6、主要调用opencv的erode（）函数实现图像腐蚀。
cv2.erode(src,kernel,iterations)
src表示原始图像
kernel表示卷积核
iterations表示迭代次数，默认值为1，表示进行一次腐蚀操作。
"""
"""
cv2.imread(filename, flags)
参数：
filepath：读入imge的完整路径
flags：标志位，{cv2.IMREAD_COLOR，cv2.IMREAD_GRAYSCALE，cv2.IMREAD_UNCHANGED}
cv2.IMREAD_COLOR：默认参数，读入一副彩色图片，忽略alpha通道，可用1作为实参替代
cv2.IMREAD_GRAYSCALE：读入灰度图片，可用0作为实参替代
cv2.IMREAD_UNCHANGED：顾名思义，读入完整图片，包括alpha通道，可用-1作为实参替代
PS：alpha通道，又称A通道，是一个8位的灰度通道，该通道用256级灰度来记录图像中的透明度复信息，定义透明、不透明和半透明区域，其中黑表示全透明，白表示不透明，灰表示半透明
"""
#------------------图像腐蚀------------------
"""import cv2
import numpy as np

img = cv2.imread("tuxiangchuli.jpg",cv2.IMREAD_UNCHANGED)  
img = cv2.resize(img,None,fx=0.2,fy=0.2)
kernel = np.ones((3,3),dtype="uint8")
img_erode = cv2.erode(img,kernel,1000)
cv2.imshow("yuantu",img)
cv2.imshow("fushi",img_erode)
cv2.waitKey()
cv2.destroyAllWindows()
"""
"""
7、图像膨胀
图像膨胀是腐蚀操作的逆操作，类似于“领域扩张”，它将图像中的高亮区域或白色部分进行扩张。
"""
"""
8、设A,B为集合，A被B的膨胀记为A⊕B，其中⊕为膨胀算子。表示用B来对图像A进行膨胀处理，其中B是一个卷积模板，其形状可以为正方形或圆形，
通过模板B与图像A进行卷积计算，扫描图像中的每一个像素点，用模板元素与二值图像元素做异或运算，如果都为0或1，那么目标像素点为0，否则为1.
从而计算B覆盖区域的像素点最大值，并用改制替换参考点的像素值实现图像膨胀。
"""
"""
9、调用opencv的dilate（）函数实现图像腐蚀。
cv2.dilate(src,kernel,iterations)
"""
#----------图像膨胀------------
"""import cv2
import numpy as np

img = cv2.imread("tuxiangchuli.jpg",cv2.IMREAD_UNCHANGED)
img = cv2.resize(img,None,fx=0.2,fy=0.2)
kernel = np.ones((5,5),dtype="uint8")
img_erode = cv2.dilate(img,kernel,1)
cv2.imshow("yuantu",img)
cv2.imshow("pengzhang",img_erode)
cv2.waitKey()
cv2.destroyAllWindows()"""

"""
10、图像开运算
开运算一般能够平滑图像的轮廓，削弱狭窄部分，去掉较细的突出。
闭运算也能平滑图像的轮廓，与开运算相反，它一般融合窄的缺口和细长的弯口，去掉小洞，填补轮廓上的缝隙。
"""
"""
11、图像开运算是图像依次经过腐蚀、膨胀处理的过程，图像被腐蚀后将去除噪声，但同时也压缩了图像，然后对腐蚀过得图像进行膨胀处理，可以在保留原有图像的基础上去除噪声。
    设A是原始图像，B是结构元素图像，则集合A被结构元素B开运算，记为A°B，表示A被B开运算就是A被B腐蚀后的结果再被B膨胀。
"""
"""
12、图像闭运算就是图像依次经过膨胀、腐蚀处理的过程，先膨胀后腐蚀有助于过滤前景物体内部的小孔或物体上的小黑点。
    设A是原始图像，B是结构元素图像，则集合A被结构元素B闭运算，记为A.B，表示A被B闭运算就是A被B膨胀后的结果再被B腐蚀。
"""
"""
13、图像梯度运算是图像膨胀处理减去图像腐蚀处理后的结果，从而得到图像的轮廓。
"""
"""
14、图像顶帽运算是用原始图像减去图像开运算后的结果，常用于解决由于光照不均匀图像分割出错的问题。
图像顶帽运算是用一个结构元通过开运算从一幅图像中删除物体。顶帽运算用于暗背景上的亮物体，它的一个重要用途是校正不均匀光照的影响。
"""
"""
15、图像底帽运算又称为图像黑帽运算，它是用图像闭运算操作减去原始图像后的结果，从而获取图像内部的小孔或前景色中黑点，也常用于解决
由于光照不均匀图像分割出错的问题。
底帽运算是用一个结构元通过闭运算从一幅图像中删除物体，常用于校正不均匀光照问题。
"""
"""
16、使用函数morphologyEx(),它是形态学扩展的一组函数，其函数原型如下：
cv2.morphologyEx(src,dist，kernel)
其中dist表示运算算法，当取
cv2.MORPH_OPEN表示图像进行开运算处理
cv2.MORPH_CLOSE表示图像进行闭运算处理
cv2.MORPH_GRADIENT表示图像进行梯度运算处理
cv2.MORPH_TOPHAT表示图像进行顶帽运算处理
cv2.MORPH_BLACKHAT表示图像进行底（黑）帽运算处理
"""

#------------------开、闭、梯度、顶帽、黑帽运算处理------------------------
"""
import cv2
import numpy as np

img = cv2.imread("tuxiangchuli.jpg",cv2.IMREAD_UNCHANGED)
img = cv2.resize(img,None,fx=0.2,fy=0.2)
kernel = np.ones((5,5),dtype="uint8")
kai = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
bi = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
tidu = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)
ding = cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel)
hei = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel)
cv2.imshow("kai",kai)
cv2.imshow("bi",bi)
cv2.imshow("tidu",tidu)
cv2.imshow("ding",ding)
cv2.imshow("hei",hei)
cv2.waitKey()
cv2.destroyAllWindows()
"""
"""
17、为什么通过顶帽运算能够消除光照不均匀的效果呢？通常可以利用灰度三维图来解释该算法。灰度三维图主要调用Axes3D包实现，对原图绘制灰度三维图的代码如下：
"""
#-------------------------绘制灰度三维图------------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator,FormatStrFormatter

# 读取图像

img = cv2.imread("tuxiangchuli.jpg",cv2.IMREAD_UNCHANGED)
img = cv2.resize(img,None,fx=0.2,fy=0.2)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
kernel = np.ones((5,5),dtype="uint8")
baimao = cv2.morphologyEx(img_gray,cv2.MORPH_TOPHAT,kernel)
imgd1 = np.array(img_gray)# 原图数据
imgd2 = np.array(baimao) # 顶帽图数据

# 准备数据
sp = img.shape
h = int(sp[0])
w = int(sp[1])

# 绘图初始处理
fig = plt.figure(figsize=(16,12))
ax = fig.gca(projection="3d")

x = np.arange(0,w,1)
y = np.arange(0,h,1)
x,y = np.meshgrid(x,y)
z = imgd2
surf = ax.plot_surface(x,y,z,cmap=cm.coolwarm)

# 自定义z轴
ax.set_zlim(-10,255)
ax.zaxis.set_major_locator(LinearLocator(10)) # 设置z轴网格线的疏密
# 将z的value字符串转为float并保留两位小数
ax.zaxis.set_major_formatter(FormatStrFormatter('%0.02f'))

# 设置坐标轴的label和标题
ax.set_xlabel('x',size=15)
ax.set_ylabel('y',size=15)
ax.set_zlabel('z',size=15)
ax.set_title("surface plot",weight='bold',size=20)

# 添加右侧的色卡条
fig.colorbar(surf,shrink=0.6,aspect=8)
plt.show()