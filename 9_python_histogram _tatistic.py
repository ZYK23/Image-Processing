#-*-conding:utf-8-*-
"""
creaded on 2022年2月10日09:48:06
@anthor Yukuan Zhang

第九章 python直方图统计
"""
"""
1、图像直方图是灰度级分布的函数，是对图像中灰度级分布的统计。
灰度直方图是将数字图像中的所有像素，按照灰度值的大小，统计其出现的频率并绘制相关图形。
"""
"""
2、灰度直方图是灰度级的函数，描述的是图像中每种灰度级像素的个数，反映图像中每种灰度出现的频率。
"""
"""
3、为了让图像各灰度级的出现频数形成固定标准的形式，可以通过归一化方法对图像直方图进行处理，将待处理的原始图像转换成相应的标准形式。
"""
"""
4、直方图广泛应用于计算机视觉领域，在使用边缘和颜色确定物体边界时，通过直方图能更好地选择边界阈值，进行阈值化处理。同时，直方图对物体与背景有较强对比的景物的分隔特别有用，可以应用于检测视频中场景的变换及图像中的兴趣点，简单物体的面积和
综合光密度也可以通过图像的直方图计算而得。
"""
"""
5、使用matplotlib.pyplot库中的hist（）函数绘制直方图，如下：
n,bins,patches = plt.hist(arr,bins=50,density=False,facecolor='green',alpha=0.75)
其中，arr表示需要计算直方图的一维数组。
bins表示直方图显示的柱数，可选项，默认值为10
density表示是否将得到的直方图进行向量归一化处理，默认值为false
facecolor表示直方图的颜色
alpha表示透明度
n表示返回值，表示直方图向量
bins为返回值，表示各个bin的区间范围。
patches为返回值，表示返回每个bin里面包含的数据，是一个列表。
"""
"""
6、cv2.imread（）读取图像时，返回的是二维数组。而hist（）函数的数据必须是一维数组，通常需要通过方法ravel（）拉直图像。
"""
#----------------灰度图原始图绘制直方图----------------------
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("tuxiangchuli.jpg",cv2.IMREAD_GRAYSCALE)

plt.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
plt.rcParams['axes.unicode_minus']=False     # 正常显示负号
# 绘制直方图
plt.hist(img.ravel(),256)
plt.xlabel("x轴")
plt.ylabel("y轴")
plt.show()

"""
#------------------彩色三通道图-----绘制直方图--显示在一个窗口上---------------------
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("tuxiangchuli.jpg")
b,g,r = cv2.split(img)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure("直方图",figsize=(8,6))

#蓝色分量直方图
plt.hist(b.ravel(),bins = 256,density=True,facecolor='b',edgecolor='b')
#绿色分量直方图
plt.hist(g.ravel(),bins = 256,density=True,facecolor='g',edgecolor='g')
#红色分量直方图
plt.hist(r.ravel(),bins=256,density=True,facecolor='r',edgecolor='r')

plt.xlabel("X")
plt.ylabel("Y")
plt.show()
"""
#-------------彩色三通道直方图绘制，分别显示三个通道的直方图--------------
"""
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("tuxiangchuli.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
r,g,b = cv2.split(img)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure("直方图",figsize=(8,6))
plt.subplot(2,2,1)
plt.imshow(img)
plt.title("原图",y=-0.2)
#蓝色分量直方图
plt.subplot(2,2,2)
plt.hist(b.ravel(),bins = 256,density=True,facecolor='b',edgecolor='b')
plt.xlabel("x",x=0.1)
plt.ylabel("y")
plt.title("蓝色分量直方图",y=-0.2)
#绿色分量直方图
plt.subplot(2,2,3)
plt.hist(g.ravel(),bins = 256,density=True,facecolor='g',edgecolor='g')
plt.xlabel("x",x=0.1)
plt.ylabel("y")
plt.title("绿色分量直方图",y=-0.2)
#红色分量直方图
plt.subplot(2,2,4)
plt.hist(r.ravel(),bins=256,density=True,facecolor='r',edgecolor='r')
plt.xlabel("X",x=0.1)
plt.ylabel("Y")
plt.title("红色分量直方图",y=-0.2)
plt.show()
"""

"""
6、使用opencv库中的calcHist（）函数计算直方图，计算完成之后再只用opencv中的绘图函数绘制直方图。
绘图函数如：rectangle（）、line（）。
如下：
hist = cv2.calcHist(images,channels,mask,histSize,ranges,accumulate)
其中，hist表示直方图的，返回一个二维数组
images表示输入的原始图像
channels表示指定通道，通道编号需要使用中括号，输入图像时灰度图像时，它的值为[0]、彩色图像为[0]、[1]、[2]，分别表示蓝色、绿色、红色
mask表示可选的操作掩码。如果要统计整幅图像的直方图，则该值为None，如果要统计图像的某一部分直方图，需要掩摸来计算。
histSize表示灰度级的个数，需要使用中括号，如[256]
ranges表示像素值范围，如[0,255]
accumulate表示累计叠加标识，默认为false，如果被设置为true，则直方图在开始分配时不会被清零。该参数允许从多个对象中计算单个直方图，或者用于实时更新直方图，过个直方图的累计结果用于对一组图像的直方图计算。
"""
#-----------------绘制直方图---------
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("tuxiangchuli.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
hist = cv2.calcHist(img,[0],None,[256],[0,255])
hist1 = cv2.calcHist(img,[1],None,[256],[0,255])
hist2 = cv2.calcHist(img,[2],None,[256],[0,255])

print("直方图大小=",hist.size)
print("直方图形状=",hist.shape)
print("每个灰度级像素频数=",hist)

plt.rcParams['font.sans-serif'] = ['SimHei']

# 显示原始图像和绘制的直方图
plt.figure("直方图",(8,6))
plt.subplot(1,2,1)
plt.imshow(img)
plt.axis('off') #不显示横纵值
plt.title("原图",y=-0.2)
plt.subplot(1,2,2)
plt.plot(hist2,color='b')
plt.plot(hist1,color='g')
plt.plot(hist,color='r')
plt.xlabel("x",x=0.1)
plt.ylabel("y")
plt.title("三通道直方图",y=-0.1)
plt.savefig("三通道图像直方图.jpg")
plt.show()
"""

"""
7、如果要统计图像的某一部分直方图，就需要使用掩模来进行计算。
掩模：假设将要统计的部分设置为白色，其余部分设置为黑色，然后使用该掩模进行直方图绘制。
"""

#----------------掩模直方图绘制----------------
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("tuxiangchuli.jpg")
img = cv2.resize(img,None,fx=0.2,fy=0.2)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

#设置掩模
mask = np.zeros(img.shape[:2],dtype="uint8")
mask[100:300,100:300] = 255
masked = cv2.bitwise_and(img,img,mask=mask)

# 图像直方图
hist_img = cv2.calcHist([img],[0],None,[256],[0,256])
hist_mask_img = cv2.calcHist([img],[0],mask,[256],[0,256])

plt.figure("掩摸直方图",figsize=(8,6))
plt.rcParams['font.sans-serif']=['SimHei']
plt.subplot(2,2,1)
plt.imshow(img,'gray')
plt.axis('off')
plt.title("原图",y=-0.2)
plt.subplot(2,2,2)
plt.imshow(mask,'gray')
plt.axis('off')
plt.title("掩模",y=-0.2)
plt.subplot(2,2,3)
plt.imshow(masked,'gray')
plt.axis('off')
plt.title("图像掩模处理",y=-0.2)
plt.subplot(2,2,4)
plt.plot(hist_img,color='r')
plt.plot(hist_mask_img,color='g')
plt.xlabel("x",x=0.1)
plt.ylabel("y")
plt.title("直方图曲线",y=-0.2)
plt.savefig("掩模直方图.jpg")
plt.show()
"""

"""
8、图像灰度变换直方图对比
"""
#-------------------灰度变换直方图对比--------------
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("tuxiangchuli.jpg")
img = cv2.resize(img,None,fx=0.2,fy=0.2)
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

h = img.shape[0]
w = img.shape[1]
up_img = np.zeros((h,w),dtype="uint8")
down_img = np.zeros((h,w),dtype="uint8")
down_cnt_img = np.zeros((h,w),dtype="uint8")
invert_img = np.zeros((h,w),dtype="uint8")
log_img = np.zeros((h,w),dtype="uint8")

img_hist = cv2.calcHist([img],[0],None,[256],[0,255])
# 灰度上移50
for i in range(h):
    for j in range(w):
        if (int(img[i,j]+50) > 255):
            up_img[i,j] = 255
        else:
            up_img[i,j] = np.uint8(int(img[i,j]+50))
up_img_hist = cv2.calcHist([up_img],[0],None,[256],[0,255])
# 灰度下移50
for i in range(h):
    for j in range(w):
        down_img[i,j] = np.uint8(int(img[i,j]-50))
down_img_hise = cv2.calcHist([down_img],[0],None,[256],[0,255])
# 降低对比度（灰度减弱）
for i in range(h):
    for j in range(w):
        down_cnt_img[i,j] = np.uint8(int(img[i,j]*0.8))
down_cnt_img_hist = cv2.calcHist([down_cnt_img],[0],None,[256],[0,255])
# 反色变换
for i in range(h):
    for j in range(w):
        invert_img[i,j] = np.uint8(int(255-img[i,j]))
invert_img_hist = cv2.calcHist([invert_img],[0],None,[256],[0,255])
# 对数变换
for i in range(h):
    for j in range(w):
        log_img[i,j] = np.uint8(int(42*np.log(1.0+img[i,j])))
log_img_hist = cv2.calcHist([log_img],[0],None,[256],[0,255])
# 二进制阈值化处理
r,b = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
b_hist = cv2.calcHist([b],[0],None,[256],[0,255])
# 展示结果
plt.rcParams['font.sans-serif']=['SimHei']
plt.figure("灰度变换直方图对比",figsize=(15,8))
plt.subplot(4,4,1),plt.imshow(img,'gray'),plt.axis('off'),plt.title("原图",y=-0.2)
plt.subplot(4,4,2),plt.plot(img_hist),plt.xlabel("x",x=0.1),plt.ylabel("y")
plt.subplot(4,4,3),plt.imshow(up_img,'gray'),plt.axis('off'),plt.title("灰度上移50",y=-0.2)
plt.subplot(4,4,4),plt.plot(up_img_hist),plt.xlabel("x",x=0.1),plt.ylabel("y")
plt.subplot(4,4,5),plt.imshow(down_cnt_img,'gray'),plt.axis('off'),plt.title("降低对比度",y=-0.2)
plt.subplot(4,4,6),plt.plot(down_cnt_img_hist),plt.xlabel("x",x=0.1),plt.ylabel("y")
plt.subplot(4,4,7),plt.imshow(invert_img,'gray'),plt.axis('off'),plt.title("反色变换",y=-0.2)
plt.subplot(4,4,8),plt.plot(invert_img_hist),plt.xlabel("x",x=0.1),plt.ylabel("y")
plt.subplot(4,4,9),plt.imshow(log_img,'gray'),plt.axis('off'),plt.title("对数变换",y=-0.2)
plt.subplot(4,4,10),plt.plot(log_img_hist),plt.xlabel("x",x=0.1),plt.ylabel("y")
plt.subplot(4,4,11),plt.imshow(b,'gray'),plt.axis('off'),plt.title("二进制阈值变换",y=-0.2)
plt.subplot(4,4,12),plt.plot(b_hist),plt.xlabel("x",x=0.1),plt.ylabel("y")
plt.subplot(4,4,13),plt.imshow(down_img,'gray'),plt.axis('off'),plt.title("灰度下移50变换",y=-0.2)
plt.subplot(4,4,14),plt.plot(down_img_hise),plt.xlabel("x",x=0.1),plt.ylabel("y")
plt.savefig("灰度变换直方图对比.jpg")
plt.show()
"""
"""
9、
颜色常用三种基本特性：
色调H：是光波混合中与主波长有关的属性，色调表示观察者接收的主要颜色。这样，当我们说一个物体是红色、橘黄色、黄色时，是指它的色调。
饱和度S：与一定色调的纯度有关，纯光谱色是完全饱和的，随着白光的加入饱和度逐渐减少。
亮度V：如果就只有亮度一个维量的变化，表达的是强度。
"""
"""
10、图像H-S直方图
为了刻画图像中颜色的直观特性，常常需要分析图像的HSV空间下的直方图特性。
HSV空间是由色调、饱和度、明度构成的，因此在进行直方图计算时，需要先将原RGB图像转化为HSV颜色空间图像，
然后将对应的H和S通道进行单元划分，在其二维空间上计算相对应的直方图，再计算直方图空间上的最大值并归一化绘制响应的直方图信息，
从而形成色调、饱和度直方图（H-S直方图）。该直方图通常应用在目标检测、特征分析以及目标特征跟踪等场景。
"""
"""
11、由于H和S分量与人感受颜色的方式是紧密相连的，V分量与图像的彩色信息无关，这些特点使得HSV模型非常适合借助人的视觉系统来感知彩色特性的
图像处理算法。
"""
"""
12、使用matplotlib.pyplot库中的imshow（）函数来绘制具有不同颜色映射的2D直方图
"""
#-------------------绘制H-S直方图-------------------
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("tuxiangchuli.jpg")
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
hsv_hist = cv2.calcHist([img_hsv],[0,1],None,[180,256],[0,180,0,256])
plt.rcParams['font.sans-serif']=['SimHei']
plt.figure("H-S直方图",figsize=(8,6))
plt.subplot(2,2,1),plt.imshow(img_rgb,'gray'),plt.axis('off'),plt.title("原图")
plt.subplot(2,2,2),plt.imshow(img_hsv,'gray'),plt.axis('off'),plt.title("HSV图")
plt.subplot(2,2,3),plt.imshow(hsv_hist,'gray',interpolation='nearest'),plt.axis('on'),plt.title("H-S直方图"),plt.xlabel("x"),plt.ylabel("y")
plt.savefig("H-S直方图")
plt.show()
"""

"""
13、直方图判断黑夜白天
常见的方法是通过计算图像的灰度平均值、灰度中值、灰度标准差，再与自定义的阈值进行对比，从而判断是黑夜还是白天。
灰度平均值：该值等于图像中所有像素灰度值之和除以图像的像素个数。
灰度中值：对图像中所有像素灰度值进行排序，然后获取所有像素最中间的值。
灰度标准差：又称为均方差，是离均差平方的算术平均数的平方根。标准差能反映一个数据集的离散程度，是总体各单位标准值与平均数离差平方的
算术平均数的平方根。如果一幅图像看起来灰蒙蒙的，那灰度标准差就小，如果一幅图看起来很鲜艳，那对比度就很大，标准差也大。
"""

#-----------计算灰度平均值、灰度中值、灰度标准差---------------------
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 函数：获取图像的灰度平均值
def fun_mean(img):
    sum_img=0
    h,w = img.shape[:2]
    for i in range(h):
        for j in range(w):
            sum_img = sum_img+int(img[i,j])
    mean = sum_img/(h*w)
    return mean

# 函数：获取中位数
def fun_median(img):
    data = img.ravel()
    length = len(data)
    data.sort()
    if(length%2)==1:
        z = length//2
        y=data[z]
    else:
        y=(int(data[length//2])+int(data[length//2-1]))/2
    return y

img = cv2.imread("tuxiangchuli.jpg")
img = cv2.resize(img,None,fx=0.2,fy=0.2)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
mean = fun_mean(img_gray)
median = fun_median(img_gray)
std = np.std(img_gray.ravel(),ddof=1)
print("灰度平均值=",mean)
print("灰度中值=",median)
print("灰度标准差=",std)
"""

"""
14、判断黑夜白天 法2
读取原始图像，转换为灰度图，并获取图像的所有像素值
设置灰度阈值并计算该阈值一下的像素个数。
设置比例参数，对比该参数与该阈值的像素占比，如果地狱参数则预测为白天，否则为黑夜。
"""
#------------判断黑夜白天---------

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 函数：判断黑夜白天的
def fun_jedge(img):
    h,w = img.shape[:2]
    hei = 0
    parameter = ["图像是黑天","图像是白天"]
    for i in range(h):
        for j in range(w):
            if img[i,j] < 50:
                hei+=1
    if ((hei*1.0) / (h*w)) >= 0.8:
        return parameter[0]
    else:
        return parameter[1]


img = cv2.imread("tuxiangchuli.jpg")
img = cv2.resize(img,None,fx=0.2,fy=0.2)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
result = fun_jedge(img_gray)
hist = cv2.calcHist([img_gray],[0],None,[256],[0,255])
plt.rcParams['font.sans-serif']=['SimHei']
plt.figure("直方图",figsize=(8,6))
plt.subplot(1,2,1)
plt.imshow(img_gray,'gray')
plt.axis('off')
plt.title(f"原图-{result[3:]}")
plt.subplot(1,2,2)
plt.plot(hist)
plt.title(f"{result}的直方图")
plt.xlabel("x",x=0.1)
plt.ylabel("y")
plt.show()