#-*-conding utf-8-*-
"""
Creaded on 2022年2月25日14:07:20
@Anthor Yukuan Zhang

第十二章 python图像锐化及边缘检测
"""
import cv2

"""
1、图像在传输过程中会受一些外界因素造成图像模糊和有噪声，此时可以通过图像锐化和边缘检测，加强图像的高频部分，锐化突出图像的边缘细节，改善图像的对比度，
使模糊的图像变得更清晰。
"""
"""
2、图像锐化和边缘检测主要包括一阶微分锐化和二阶微分锐化。
"""
"""
3、一般来说，图像的能量主要几种在低频部分，噪声所在的频段主要在高频段，同时图像边缘信息主要几种在高频部分。
这就导致原始图像在平滑处理之后，图像边缘和图像轮廓模糊。为了减少这类不利的影响，就需要利用图像锐化技术，使图像的边缘变得清晰。
"""
"""
4、图像锐化处理的目的是使图像的边缘、轮廓线以及图像的细节变得清晰，经过平滑的图像变得模糊的根本原因是图像受到平均或积分运算，因此可以对其进行逆运算。
微分运算是求信号的变化率，具有较强高频分量作用。 从频率域来考虑，图像模糊的实质是其高频分量被衰减，因此可以用高通滤波器来使图像清晰。 
"""
"""
5、能够进行锐化处理的图像必须有较高的信噪比，否则锐化后图像信噪比反而更低，从而使得噪声增加比信号还要多，因此一般是先去除或减轻噪声后在进行锐化处理。
这时需要开展图像锐化和边缘检测处理，加强原图像的高频部分，锐化突出图像的边缘细节，改善图像的对比度，使模糊的图像变得更清晰。
"""
"""
6、图像锐化和边缘提取技术可以消除图像中的噪声，提取图像信息中用来表征图像的一些变量，为图像识别提供基础。
通过使用灰度差分法对图像的边缘、轮廓进行处理并凸显。图像锐化的方法分为高通滤波和空域微分法。
"""
"""
7、一阶微分算子
一阶微分算子一般借助空域微分算子通过卷积完成，但实际上数字图像处理中求导是利用差分近似微分来进行的,如对于数字图像f(x,y)，df/dx =f(x+1,y)-f(x,y),df/dx =f(x,y+1)-f(x,y)。
梯度对应一阶导数，梯度算子是一阶导数算子。对于数字图像f（x，y），导数可以用差分来近似，则梯度可以表示为：▽f = （f(i+1,j)-f(i,j),f(i,j+1)-f(i,j)）

在实际中常用区域模板卷积来近似计算，对水平方向和垂直方向各用一个模板，再通过两个模板组合起来构成一个梯度算子。
根据模板的大小，其中元素值不同，可以提出多种模板，构成不同的检测算子，后面将对各种算子进行详细介绍。
由梯度的计算可知，在图像灰度变化较大的边缘区域其梯度值大，在灰度变化平缓的区域梯度值较小，而在灰度均匀的区域其梯度值为零。
根据得到的梯度值来返回像素值，如将梯度值大的像素设置成白色，梯度值小的设置为黑色，这样就可以将边缘提取出来，或者是加强梯度值大的像素灰度值就可以突出细节，从而达到锐化目的。
"""
"""
8、二阶微分算子
二阶微分算子是求图像灰度变化导数的导数，对图像中灰度变化强烈的地方很敏感，从而可以突出图像的纹理结构。当图像灰度变化剧烈时，进行一阶微分则会形成一个局部的机极值，对图像进行二阶微分则会形成一个
过零点，并且在零点两边产生一个波峰和波谷，设定一个阈值检测到这个过零点。
这样做的好吃有两个：一是二阶微分关心的是图像灰度的突变而不强调灰度缓慢变化的区域，对边缘的定位能力更强；
二是Laplacian算子是各向同性的，即具有旋转不变性，在一阶微分中，是用|dx|+|dy|来近似一个点的梯度，当图像旋转一个角度时，这个值就会变化，但对于Lapkacuab算子来说，不管图像怎么旋转，得到的相应值
是一样的。
要想确定过零点以p为中心的一个3*3邻域，p点为过零点意味着至少有两个相对的邻域像素的符号不同。有几种要检测的情况：左右，上下，两个对角。如果g（x，y）的值与一个阈值比较，那么不仅要求相对邻域的符号不同，
数值差的绝对值也要超过这个阈值，这时p称为过零点像素。
"""
"""
9、二阶微分在恒定灰度区域的微分值为零，在灰度台阶或斜坡起点处微分值非零，沿着斜坡的微分值为零。一阶微分算子获得的边界是比较粗略的边界，反映的边界信息较少，但是所反映的边界比较清晰；二阶微分算子
获得的边界是比较细致的边界，反映的边界信息包括许多细节信息，但是所反映的边界不是太清晰。
"""
"""
10、边缘检测算法主要是基于图像增强的一阶和二阶导数，但导数通常对噪声很敏感，因此需要采用滤波器来过滤噪声，并调用图像增强或阈值化算法进行处理，最后再进行边缘检测。
高斯滤波调用cv2.GaussianBlur（srt，（3,3），0）
阈值化处理调佣 ret，binary = cv2.threshold（）
"""
"""
10、Roberts算子
roberts算子又称为交叉微分算法，它是基于交叉差分的梯度算法，通过局部差分计算检测边缘线条。常用来处理具有陡峭的低噪声图像。当图像边缘接近于+45°或-45°时，该算法处理效果更理想，其缺点是对边缘的定位不太准确，提取的边缘线条较粗。
在python中Roberts算子主要通过Numpy定义模板，再调用OpenCV的filter2D（）函数实现边缘提取。该函数主要是利用内核实现对图像的卷积运算，其函数原型如下图所示。
dst = cv2.filter2D(src,ddepth,kernel[,det[,anchor[,delta[,borderType]]]])
1)src 表示输入图像
2）ddepth表示目标图像所需的深度
3）kernel表示卷积核，一个单通道浮点型矩阵
4）dst表示输出的边缘图像，其大小和通道数与输入图像相同
5）anchor表示内核的基准点，其默认值为（-1，-1），位于中心位置
6）delta表示在储存目标图像前可选的添加到像素的值，默认值为0
7）bordertype表示边框模式
在进行Roberts算子处理之后，还需要调用cv2.convertscaleabs（）函数计算绝对值，并将图像转换为8位图进行显示。其函数原型如下：
dst = cv2.converScaleAbs(src[,dst[,alpha[,beta]]])
src表示原数组
dst表示输出数组，深度为8位
alpha表示比例因子
beta表示原数组元素按比例缩放后添加的值
最后调用addweighted（）函数计算水平方向和垂直方向的Roberts算子。
"""
#-------------------------------------------------------------Roberts算子---------------------------------
"""
import cv2
import numpy as np

img = cv2.imread("tuxiangchuli.jpg")
img = cv2.resize(img,None,fx=0.2,fy=0.2)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 高斯滤波
img_gray = cv2.GaussianBlur(img_gray,(3,3),0)
# 阈值化处理
ret,img_gray = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)
# 设置Roberts算子 参数
kernelx = np.array([[-1,0],[0,1]],dtype=int) # 水平方向卷积核（模板）
kernely = np.array([[0,-1],[1,0]],dtype=int) # 垂直方向卷积核（模板）
# 使用cv2.filter2D（）进行边缘提取
x = cv2.filter2D(img_gray,cv2.CV_16S,kernelx) # 水平方向边缘提取
y = cv2.filter2D(img_gray,cv2.CV_16S,kernely) # 垂直方向边缘提取
# 使用cv2.convertscaleabs（）函数计算绝对值并将图像转换为8位图进行显示
absx = cv2.convertScaleAbs(x) # 水平方向的边缘检测结果
absy = cv2.convertScaleAbs(y)# 垂直方向的边缘检测结果
# 使用addweighted()函数计算水平方向和垂直方向的Roberts算子
Roberts = cv2.addWeighted(absx,0.5,absy,0.5,0)

cv2.imshow("x - 1",absx)
cv2.imshow("y - 2",absy)
cv2.imshow("x-y-3",Roberts)
cv2.waitKey()
cv2.destroyAllWindows()

"""
"""
11、prewitt算子
prewitt算子是一种图像边缘检测的微分算子，其利用特定区域内像素灰度值产生的差分实现边缘检测。由于Prewitt算子采用3*3模板对区域内的像素值进行计算，而Roberts算子的模板是2*2，故prewitt算子
的边缘检测结果在水平方向和垂直方向的检测结果都比Roberts算子更加明显。prewitt算子适合用来识别噪声较多，灰度渐变的图像。
prewitt算子的实现与Roberts算子一样，只是卷积核（模板）不一样。
"""
# -----------------------------prewitt算子---------------------

"""
import cv2
import numpy as np

img = cv2.imread("tuxiangchuli.jpg")
img = cv2.resize(img,None,fx=0.2,fy=0.2)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 高斯滤波
img_gray = cv2.GaussianBlur(img_gray,(3,3),0)
# 阈值化处理
ret,img_gray = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)
kernelx = np.array([[1,1,1,],[0,0,0,],[-1,-1,-1]],dtype=int)
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]],dtype=int)
x = cv2.filter2D(img_gray,cv2.CV_16S,kernelx)
y = cv2.filter2D(img_gray,cv2.CV_16S,kernely)
absx = cv2.convertScaleAbs(x)
absy = cv2.convertScaleAbs(y)
Prewitt = cv2.addWeighted(absx,0.5,absy,0.5,0)
cv2.imshow("x",absx)
cv2.imshow("y",absy)
cv2.imshow("x-y",Prewitt)
cv2.waitKey()
cv2.destroyAllWindows()
"""
"""
12、sobel算子
sobel算子是一种用于边缘检测的离散微分算子，它结合了高斯平滑和微分求导。
该算子用于计算图像明暗程度近似值，根据图像边缘旁边明暗程度把该区域内超过某个数的特定点记为边缘。
sobel算子在prewitt算子的基础上增加了权重的概念，认为相邻点的距离对当前像素点的影响是不同的，距离越近的像素点对当前像素的影响越大，从而实现图像锐化并突出边缘轮廓。
sobel算子的边缘定位更准确，常用于噪声较多，灰度渐变的图像。
sobel算子根据像素点上下，左右邻点灰度加权差，在边缘处达到极值这一现象检测边缘。对噪声具有平滑作用，提供较为精确的边缘方向信息。因为sobel算子结合了高斯平滑和
微分求导（分化），因此结果会具有更多的抗噪性，当对精度要求不是很高时，sobel算子是一种较为常用的边缘检测方法。

使用cv2.Sobel（）函数
dst = cv2.Sobel(dst,ddepth,dx,dy[,dst[,ksize[,scale[,delta[,bordertype]]]]])
ddepth表示目标图像所需的深度，针对不同的输入图像，输出目标图像有不同的深度。
dx表示x方向上的差分阶数，取值1或0
dy表示y方向上的差分阶数，取值1或0
dst表示输出的边缘图，其大小和通道数与输入图像相同
ksize表示sobel算子的大小，其值必须是正数和奇数。
scale表示缩放导数的比例常数，默认情况下没有伸缩系数
delta表示将结果存入目标图像之前，添加到结果中的可选增量值
bordertype表示边框模式

在进行sobel算子处理之后，还需要调用cv2.convertscaleabs（）函数计算绝对值，并将图像转换为8位图像进行显示。
最后调用addweighted（）函数计算水平方向和垂直方向的Roberts算子。

"""
#-----------------sobel算子---------------------
"""
import cv2

img = cv2.imread("tuxiangchuli.jpg")
img = cv2.resize(img,None,fx=0.2,fy=0.2)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 高斯滤波
img_gray = cv2.GaussianBlur(img_gray,(3,3),0)
# 阈值化处理
ret,img_gray = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)
# sobel算子
x = cv2.Sobel(img_gray,cv2.CV_16S,1,0) # 水平方向sobel算子 x方向的导数
y = cv2.Sobel(img_gray,cv2.CV_16S,0,1)# 垂直方向sobel算子 y方向的导数

absx = cv2.convertScaleAbs(x)
absy = cv2.convertScaleAbs(y)
Sobel = cv2.addWeighted(absx,0.5,absy,0.5,0)

cv2.imshow("x",absx)
cv2.imshow("y",absy)
cv2.imshow("x-y",Sobel)
cv2.waitKey()
cv2.destroyAllWindows()
"""
"""
13、Laplacian算子
laplacian算子是n维欧几里得空间中的一个二阶微分算子，常用于图像增强邻域和边缘提取。他通过灰度差分计算邻域内的像素，基本流程是：判断图像中心像素灰度值与它周围其他像素的灰度值，如果中心像素的灰度值更高，则提升中心
像素的灰度值，反之降低中心像素的灰度值，从而实现图像锐化操作。
在算法实现过程中，laplacian算子通过对邻域中心像素的四方向或八方向求梯度，再将梯度相加起来判断中心像素灰度与邻域内其他像素灰度的关系，最后通过梯度运算的结果对像素灰度进行调整。
使用cv2.Laplacian(dst,ddepth[,dst[,ksize[,scale[,delta[,bordertype]]]]])
ksize表示用于计算二阶导数的滤波器的孔径大小，其值必须是正数或奇数，且默认值为1.
当ksize=1时，Laplacian（）函数采用3*3的孔径（四邻域模板）进行变换处理。
"""

# - --------------------laplacian算子------------------------
"""
import cv2

img = cv2.imread("tuxiangchuli.jpg")
img = cv2.resize(img,None,fx=0.2,fy=0.2)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 高斯滤波
img_gray = cv2.GaussianBlur(img_gray,(3,3),0)
# 阈值化处理
ret,img_gray = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)
# laplacian算子
dst = cv2.Laplacian(img_gray,cv2.CV_16S,ksize=3)
Laplacian = cv2.convertScaleAbs(dst)

cv2.imshow("Laplacian",Laplacian)
cv2.waitKey()
cv2.destroyAllWindows()
"""

"""
14、综合分析，Laplacian算子对噪声比较敏感，由于其算法可能会出现双像素边界，常用来判断边缘像素位于图像的明区或暗区，很少用于边缘检测。
Robert算子对陡峭的低噪声图像效果较好，尤其是边缘正负45°的图像，但定位准确率较差，Prewitt算子对灰度渐变的图像边缘提取效果较好，而没有考虑相邻点的距离对当前像素点的影响。
sobel算子考虑了总和因素，对噪声较多的图像处理效果较好。
"""
"""
15、scharr算子
由于sobel算子在计算相对较小的核的时候，其近似计算导数的精度比较低，如一个3*3的sobel算子，当梯度角度接近水平或垂直方向时，其不精确性就更加明显。scharr算子同sobel算子的速度一样块，
但是准确率更高，尤其是计算较小核的情景，所以利用3*3滤波器实现图像边缘提取更推荐使用scahrr算子。
实现过程与sobel算子一样，只是卷积核大小（模板）不一样。
"""
#-------------------scharr算子--------------------
"""
import cv2

img = cv2.imread("tuxiangchuli.jpg")
img = cv2.resize(img,None,fx=0.2,fy=0.2)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

x = cv2.Scharr(img_gray,cv2.CV_16S,1,0)
y = cv2.Scharr(img_gray,cv2.CV_16S,0,1)
absx = cv2.convertScaleAbs(x)
absy = cv2.convertScaleAbs(y)
Scharr = cv2.addWeighted(absx,0.5,absy,0.5,0)
cv2.imshow("Scharr",Scharr)
cv2.waitKey()
cv2.destroyAllWindows()
"""
"""
16、Canny算子
Canny算子是一种广泛应用于边缘检测的标准算法，其目标是找到一个最优的边缘检测解或寻找一幅图像中灰度强度变化最强的位置。
实现步骤：
使用高斯平滑去除噪声
按照sobel滤波器步骤计算梯度幅值和方向，寻找图像的强度梯度。
通过非极大值抑制过滤掉非边缘像素，将模糊的边界变得清晰。
利用双阈值方法来确定潜在的边界
利用滞后技术来跟踪边界
调用cv2.Canny（）函数
 Canny(image: Any,
          threshold1: Any, 第一个滞后性阈值
          threshold2: Any, 第二个滞后性阈值
          edges: Any = None, 输出的边缘图
          apertureSize: Any = None, 应用sobel算子的孔径大小，其默认值为3
          L2gradient: Any = None) -> None 一个计算图像梯度幅值的标识，默认值为false
其中，
"""

#-----------------canny算子------------------
"""
import cv2

img = cv2.imread("tuxiangchuli.jpg")
img = cv2.resize(img,None,fx=0.2,fy=0.2)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 高斯滤波降噪
img_gray = cv2.GaussianBlur(img_gray,(1,1,),0)
# canny算子
canny = cv2.Canny(img_gray,50,150)

cv2.imshow("canny",canny)
cv2.waitKey()
cv2.destroyAllWindows()
"""

"""
17、log算子
log算子根据图像的信噪比来检测边缘的最优滤波器。
该算法首先对图像做高斯滤波，然后再求其Laplacian二阶导数，根据二阶导数的过零点来检测图像的边界，即通过检测滤波结果零交叉来获得图像或物体的边缘。
log算子综合考虑了对噪声的抑制和对边缘的检测两个方面，并且把高斯平滑滤波器和laplacian锐化滤波器结合起来，先平滑掉噪声，再进行边缘检测，所以效果会更好。
该算子与视觉胜利中的数学模型相似，因此在图像处理领域中得到了广泛的应用，它具有抗干扰能力强，边界定位精度高，边缘连续性好，能有效提取对比度弱的边界等特点。
"""

#-----------------log算子-------------------
"""
import cv2

img = cv2.imread("tuxiangchuli.jpg")
img = cv2.resize(img,None,fx=0.2,fy=0.2)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 高斯滤波降噪
img_gray = cv2.GaussianBlur(img_gray,(1,1,),0)
# laplacian算子做边缘检测
dst = cv2.Laplacian(img_gray,cv2.CV_16S,ksize=3)
# log算子
log = cv2.convertScaleAbs(dst)

cv2.imshow("canny",log)
cv2.waitKey()
cv2.destroyAllWindows()
"""

"""
18、在opencv中还可以通过cv2.findContours（）函数从二值图像中寻找轮廓，其函数原型如下：
 findContours(image: Any,
                 mode: Any, 表示轮廓检索模式
                 method: Any, 表示轮廓的近似方法
                 contours: Any = None, 表示检测到的轮廓
                 hierarchy: Any = None,表示输出变量，包含图像的拓扑信息，作为轮廓数量的表示，它包含了许多元素
                 offset: Any = None) -> None 表示每个轮廓点的可选偏移量

在使用findcontours（）函数检测图像边缘轮廓后，通常需要和drawcontours（）函数联合使用，接着绘制检测到的轮廓。
drawContours(image: Any,表示目标图像，即所要绘制轮廓的背景图像
                 contours: Any,表示所有的输入轮廓，每个轮廓存储为一个点向量
                 contourIdx: Any,表示轮廓绘制的指示变量，如果为负数表示绘制所有轮廓
                 color: Any,表示绘制轮廓的颜色
                 thickness: Any = None,表示绘制轮廓线条的粗细程度，默认值为1
                 lineType: Any = None,表示线条类型，默认值为8，可选线包括8（8连通线型）、4（4连通线型）、cv——aa（抗锯齿线型）
                 hierarchy: Any = None,表示可选的层次结构信息
                 maxLevel: Any = None,表示用于绘制轮廓的最大等级
                 offset: Any = None) -> None表示每个轮廓点的可选偏移量
"""
#-----------------------使用findcontours检测轮廓-------------------------

"""
import cv2

img = cv2.imread("tuxiangchuli.jpg")
img = cv2.resize(img,None,fx=0.2,fy=0.2)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 高斯滤波
img_gray = cv2.GaussianBlur(img_gray,(1,1),0)
#阈值化处理
ret,binary = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#边缘检测
contours,hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# 轮廓绘制
cv2.drawContours(img,contours,-1,(255,0,0),1)
cv2.imshow("lunkuo",img)
cv2.waitKey()
cv2.destroyAllWindows()
"""
