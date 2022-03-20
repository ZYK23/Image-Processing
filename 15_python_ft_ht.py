#-*-conding:utf-8-*-
"""
Creaded on 2022年2月27日16:57:33
@Anthor YKuan Zhang

第15章 python傅里叶变换与霍夫变换
"""
import cv2
import numpy as np

"""
1、在数字图像处理中，有两个经典的变换被广泛应用---傅里叶变换和霍夫变换。其中，傅里叶变换主要是将时间域上的信号转变为频率域上的信号，用来进行图像除噪、图像增强等处理。
霍夫变换主要用来辨别找出物体的特征，进行特征检测、图像分析、数位影像等处理
"""
"""
2、傅里叶变换可以应用于图像处理中，经过对图像进行变换得到其频谱图。从频谱图中频率高低来表征图像中灰度变化的剧烈程度。图像中的边缘信号和噪声信号往往是高频信号，而图像变化频繁的图像轮廓及背景等信号往往是低频信号。这时
可以有针对性地对图像进行相关操作，如图像除噪、图像增强和锐化等。
fft的结果为复数，绝对值是振幅
有两种方法能够实现傅里叶变换，通过numpy或opencv可以实现,
"""
"""
3、numpy实现傅里叶变换
首先调用np.fft.fft2（）实现快速傅里叶变换，再调用np.fft.shift（）将fft输出中的直流分量移动到频谱中央，将中心位置转移到中间。
"""
"""
4、numpy实现傅里叶逆变换
傅里叶逆变换是将频谱图像转换为原始图像的过程。通过傅里叶变换转换为频谱图，并对高频（边界）和低频（细节）部分进行处理，接着需要通过傅里叶逆变换恢复原始效果图。
频域上对图像的处理，会反映在逆变换图像上，从而更好地进行图像处理。
"""
#------------------------fft-ifft---------------------------------

"""
import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread("tuxiangchuli.jpg")
img = cv2.resize(img,None,fx=0.2,fy=0.2)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# fft
f = np.fft.fft2(img_gray)
#默认结果中心点位置是在左上角，需要转移到中间位置
fshift = np.fft.fftshift(f)

#fft结果为复数，绝对值为振幅
#将复数变化成实数，取对数的目的是将数据变化到0-255
fimg = np.log(np.abs(fshift))

#ifft
# 将频谱图像的中心低频部分移动至左上角
ishift = np.fft.ifftshift(fshift)
iimg = np.fft.ifft2(ishift)
# 将复数转换为0-255
iimg = np.abs(iimg)


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.subplot(1,3,1),plt.imshow(img_gray,'gray'),plt.title("原图"),plt.axis('off')
plt.subplot(1,3,2),plt.imshow(fimg,'gray'),plt.title("fft"),plt.axis('off')
plt.subplot(1,3,3),plt.imshow(iimg,'gray'),plt.title("ifft"),plt.axis('off')
plt.show()
"""
"""
5、观察结果可以得知，频谱图中，越靠近中心位置频率越低，越亮（灰度值越高）的位置代表该频率的信号振幅越大。
需要注意，傅里叶变换得到低频、高频信息，针对低频和高频处理能够实现不同的目的，同时傅里叶过程是可逆的，图像经过傅里叶变换、傅里叶逆变换能够恢复原始图像。
"""

"""
6、opencv实现傅里叶变换
opencv中相应的函数是cv2.dft（），它和用numpy输出的结果一样，但是是双通道的。第一个通道是结果的实数部分，第二个通道是结果的虚数部分，并且输入图像要首先转换成np.float32格式。
注意由于输出的频谱结果是一个复数，需要调用cv2.magnitude（）函数将傅里叶变换的双通道结果转换为0-255.
"""
"""
7、opencv实现傅里叶逆变换
cv2.idft（）实现傅里叶逆变换，其返回结果取决于原始图像的类型和大小，原始图像可以为实数或复数。
"""
#-------------------------dft-idft-----------------------
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("tuxiangchuli.jpg")
img = cv2.resize(img,None,fx=0.2,fy=0.2)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#傅里叶变换
dft = cv2.dft(np.float32(img_gray),flags=cv2.DFT_COMPLEX_OUTPUT)
# 将频谱低频从左上角移至中心位置
dftshift = np.fft.fftshift(dft)
# 频谱图像双通道复数转换为0-255区间
result = 20*np.log(cv2.magnitude(dftshift[:,:,0],dftshift[:,:,1]))
# 傅里叶逆变换
ishift = np.fft.ifftshift(dftshift)
iimg = cv2.idft(ishift)
result2 = cv2.magnitude(iimg[:,:,0],iimg[:,:,1])

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.subplot(1,3,1),plt.imshow(img_gray,'gray'),plt.title("原图"),plt.axis('off')
plt.subplot(1,3,2),plt.imshow(result,'gray'),plt.title("fft"),plt.axis('off')
plt.subplot(1,3,3),plt.imshow(result2,'gray'),plt.title("ifft"),plt.axis('off')
plt.show()
"""

"""
7、基于傅里叶变换的高通滤波和低通滤波
傅里叶变换的目的并不是观察图像的频谱分布（至少不是最终目的），更多情况下是对频率进行过滤，通过修改频谱以实现图像增强、图像去噪、边缘检测、特征提取、压缩加密等。
过滤：低通、高通、带通
高通滤波器：常用于增强尖锐的细节，但是图像的对比度会降低。该滤波器将检测图像的某个区域，根据像素与周围像素的差值来提升像素的亮度。可以使用高通滤波器提取图像的边缘。
rows,cols = img.shape
crow,ccol = int(rows/2),int(cols/2)
mask = np.ones((rows,cols,2),dtype="uint8")
mask[crow-30:crow+30,ccol-30:ccol+30] = 0
低通滤波器：当一个像素与周围像素的差值小于一个特定值时，平滑该像素的亮度，常用于去噪和模糊化处理。
低通滤波器模板为
rows,cols = img.shape
crow,ccol = int(rows/2),int(cols/2)
mask = np.zeros((rows,cols,2),dtype="uint8")
mask[crow-30:crow+30,ccol-30:ccol+30] = 1
"""
# --------------高通和低通-------------------
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("tuxiangchuli.jpg")
img = cv2.resize(img,None,fx=0.2,fy=0.2)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# 傅里叶变化
f = np.fft.fft2(img_gray)
fshift = np.fft.fftshift(f)
result = np.log(np.abs(fshift))
#设置高通和低通滤波器模板
rows,cols = img_gray.shape
crow,ccol = int(rows/2),int(cols/2)
gao_mask = np.ones((rows,cols),dtype="uint8")
gao_mask[crow-30:crow+30,ccol-30:ccol+30] = 0

di_mask = np.zeros((rows,cols),dtype="uint8")
di_mask[crow-30:crow+30,ccol-30:ccol+30] = 1

# 处理
gao_fshift = fshift * gao_mask
gao_mask = np.abs(gao_fshift)
di_fshift = fshift * di_mask
di_mask = np.abs(di_fshift)
#逆变换
gao_ishift = np.fft.ifftshift(gao_fshift)
gao_iimg = np.fft.ifft2(gao_ishift)
gao_iimg = np.abs(gao_iimg)

di_ishift = np.fft.ifftshift(di_fshift)
di_iimg = np.fft.ifft2(di_ishift)
di_iimg = np.abs(di_iimg)


plt.rcParams['font.sans-serif']=['SimHei']
plt.figure()
imgs = [img_gray,result,gao_mask,gao_iimg,di_mask,di_iimg]
titles = ["原图","fft","高通滤波器","高通","低通滤波器","低通"]
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(imgs[i],'gray')
    plt.title(titles[i])
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
plt.show()
"""
"""
8、图像霍夫变换
霍夫变换是一种特征检测，广泛应用于图像分析、计算机视觉以及数位影响处理。
景点的霍夫变换是检测图片中的直线，之后，霍夫变换不仅能识别直线，也能识别任何形状，常见的有圆形、椭圆形。
霍夫变换是一种特征提取技术，用来辨别找出物件中的特征，其目的是通过投票程序在特定类型的形状内找到对象的不完美实例。这个投票程序是在一个参数空间中进行的，在这个参数空间中，候选对象被当做
所谓的累加器空间中的局部最大值来获得，累加器空间是由计算霍夫变换的算法明确地构建的。霍夫变换主要优点是能容忍特征边界描述的间隙，并且相对不受图像噪声的影响。
"""
"""
9、图像霍夫线变换操作
在opencv中，霍夫变换分为霍夫线变换和霍夫圆变换，其中霍夫线变换支持三种不同方法：标准霍夫变换、多尺度霍夫变化核累计概率霍夫变换。
1、标准霍夫变换主要由cv2.HoughLines()函数实现
2、多尺度霍夫变换是标准霍夫变换在多尺度下的变换，可以通过HoughLines（）函数实现。
3、累计概率霍夫变换是标准霍夫变换的改进，他能在一定范围内进行霍夫变换，计算单独线段的方向及范围，从而减少计算量，缩短计算时间，可以通过cv2.HoughLinesP()函数实现。
HoughLines(image: Any,
               rho: Any,
               theta: Any,
               threshold: Any,
               lines: Any = None,
               srn: Any = None,
               stn: Any = None,
               min_theta: Any = None,
               max_theta: Any = None) -> None
HoughLinesP(image: Any,
                rho: Any,
                theta: Any,
                threshold: Any,
                lines: Any = None,
                minLineLength: Any = None,
                maxLineGap: Any = None) -> None
               
"""
#---------------houghlines-------------------

"""
import cv2
import numpy as np

img = cv2.imread("xian.png")
#img = cv2.resize(img,None,fx=0.2,fy=0.2)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#转换为二值图像
edges = cv2.Canny(img_gray,50,150)
cv2.imshow("1",edges)

#霍夫变换检测直线
lines = cv2.HoughLines(edges,1,np.pi / 180,160)

# 三维转换为二维
line = lines[:,0,:]

# 将检测到的线在极坐标中绘制
for rho,theta in line[:]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    print(x0,y0)
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    print(x1,y1,x2,y2)
    # 绘制直线
    cv2.line(img,(x1,y1),(x2,y2),(255,0,0),1)

cv2.imshow("2", img)
cv2.waitKey()
cv2.destroyAllWindows()
"""
#-------------------houghlinesp--------------------------
"""
import cv2
import numpy as np

img = cv2.imread("xian.png")
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#转换为二值图像
edges = cv2.Canny(img_gray,50,150)
cv2.imshow("1",edges)

#概率霍夫变换检测直线
rho = 1 # 表示以像素为单位的累加器的距离精度
thera = np.pi / 180 #表示以弧度为单位的累加器的角度精度
threshold = 30 # 表示累加平面的阈值参数，识别某部分为图像中的一条直线时它在累加平面中必须达到的值，大于该值的线段才能被检测返回
min = 60 # 表示最低线段的长度，比这个设定参数短的线段不能显示出来
max = 10 # 表示允许将同一行点与点之间连接起来的最大距离
lines = cv2.HoughLinesP(edges,rho,thera,threshold,min,max)
print(lines)
# 三维转换为二维
line = lines[:,0,:]

# 将检测到的线在极坐标中绘制
for x1,y1,x2,y2 in line[:]:
    # 绘制直线
    cv2.line(img,(x1,y1),(x2,y2),(255,0,0),1)

cv2.imshow("2", img)
cv2.waitKey()
cv2.destroyAllWindows()
"""
"""
10、图像霍夫圆变换操作
霍夫圆变换的原理与霍夫线变换很类似，一个圆的确定需要三个参数，通过三层循环实现，接着寻找参数空间累加器的
最大（或大于某一阈值）值。随着数据量的增大，圆的检测将比直线更耗时，所以一般使用霍夫梯度法减少计算量。在opencv中，
提供了cv2.houghcircles()函数检测圆，其含函数原型如下所示。
 HoughCircles(image: Any,
                 method: Any,
                 dp: Any,
                 minDist: Any,
                 circles: Any = None,
                 param1: Any = None,
                 param2: Any = None,
                 minRadius: Any = None,
                 maxRadius: Any = None) -> None
"""

#------------------检测圆-------------------
import cv2
import numpy as np

img = cv2.imread("yuan.png")
img = cv2.resize(img,None,fx=0.2,fy=0.2)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#转换为二值图像
er = cv2.Canny(img_gray,50,150)
cv2.imshow("1",er)

# 霍夫变换检测圆
circles1 = cv2.HoughCircles(er,cv2.HOUGH_GRADIENT,1,20,param1=100,param2=50,minRadius=110,maxRadius=180)
# 通过不断调整param1=100,param2=50,minRadius=110,maxRadius=180达到最优点
print(circles1)
# 三维转二维
circles1 = circles1[0,:,:]
print(circles1)

# 四舍五入取整
circles1 = np.uint16(np.around(circles1))
print(circles1)

# 绘制圆
for i in circles1[:]:
    cv2.circle(img,(i[0],i[1]),i[2],(255,0,0),1) #画圆
    cv2.circle(img,(i[0],i[1]),1,(255,0,255),10) #画圆

cv2.imshow("2",img)
cv2.waitKey()
cv2.destroyAllWindows()