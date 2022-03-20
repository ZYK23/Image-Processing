#-*-coding:utf-8-*-
"""
Creaded on 2022年1月13日22:13:14
@author Yukuan Zhang
第六章：python图像量化及采样处理
"""

"""
数字化幅度值称为量化，数字化坐标值称为采样。
图像量化处理：
量化是指将图像像素点对应亮度的连续变化区间转换为单个特定值的过程，即将原始灰度图像的空间坐标幅度值离散化。量化等级越多，
图像层次越丰富，灰度分辨率越高，图像的质量也越好。相反，量化等级越少，图像层次欠丰富，灰度分辨率越低，会出现图像轮廓分层的现象，降低图像的质量。
"""

"""
1、量化方法
图像量化处理的核心流程是建立一幅临时图像，然后循环遍历原始图像中所有像素点，判断每个像素点应该属于的量化等级，最后将临时图像显示。

"""

#------------量化：等级为2/4/8------------
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("tuxiangchuli.jpg")
img = cv2.resize(img,None,fx=0.2,fy=0.2)
rols,wols = img.shape[:2]
new_img1 = np.zeros((rols,wols,3),dtype="uint8")
new_img2 = np.zeros((rols,wols,3),dtype="uint8")
new_img3 = np.zeros((rols,wols,3),dtype="uint8")

# 图像量化等级为2的量化处理

for i in range(rols):
    for j in range(wols):
        for k in range(3): # 对应BGR分量
            if img[i,j][k] < 128:
                gray = 0
            else:
                gray = 128
            new_img1[i,j][k] = np.uint8(gray)

# 量化等级为4的量化处理

for i in range(rols):
    for j in range(wols):
        for k in range(3):
            if img[i,j][k] < 64:
                gray = 0
            elif img[i,j][k] <128:
                gray = 64
            elif img[i,j][k] < 192:
                gray = 128
            else:
                gray = 192
            new_img2[i,j][k] = np.uint8(gray)

#量化等级为8的处理

for i in range(rols):
    for j in range(wols):
        for k in range(3):
            if img[i,j][k] < 32:
                gray = 0
            elif img[i,j][k] < 64:
                gray = 32
            elif img[i,j][k] < 96:
                gray = 64
            elif img[i,j][k] < 128:
                gray = 96
            elif img[i,j][k] < 160:
                gray = 128
            elif img[i,j][k] < 192:
                gray = 160
            elif img[i,j][k] < 224:
                gray = 192
            else:
                gray = 224
            new_img3[i,j][k] = np.uint8(gray)

# 用来正确显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']

titles = ['原始图像','量化-L2','量化-L4','量化-L8']
imgs = [img,new_img1,new_img2,new_img3]

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(imgs[i])
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()
"""

"""
2、图像采样处理
图像采样处理是将一幅连续图像在空间上分隔成M*N的网格，每个网格用一个亮度值或灰度值来表示。
图像采样的间隔越大，所得图像像素数越少，空间分辨率越低，图像质量越差，甚至出现马赛克效应
数字图像处理的质量很大程度上取决于量化和采样中所采用的的灰度级和样本数。

具体处理方法：
其核心流程是建立一幅临时图像，设置需要的采样区域，然后循环遍历原始图像中所有像素点，采样区域内的像素点赋值相同，最终实现图像采样处理。
"""

#---------采样区域为16*16---------------
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("tuxiangchuli.jpg")
img = cv2.resize(img,None,fx=0.2,fy=0.2)

height = img.shape[0]
weight = img.shape[1]

new_img = np.zeros((height,weight,3),dtype="uint8")

# 设置采样区域 16*16
numheight = height/16
numweight = weight/16
print(height,numheight)
print(weight,numweight)

# 图像循环采样16*16区域
for i in range(16):
    y = int(i * numheight) # 获取y坐标
    for j in range(16):
        x = int(j * numweight) # 获取x坐标

        b = img[y,x][0] # 获取填充颜色，采样区域左上角像素点
        g = img[y,x][1]
        r = img[y,x][2]

        #循环采样区域进行量化

        for n in range(int(numheight)):
            for m in range(int(numweight)):

                new_img[y+n,x+m][0] = np.uint8(b) # 获取的采样区域坐上角像素值赋值给整个采样区域
                new_img[y+n,x+m][1] = np.uint8(g)
                new_img[y+n,x+m][2] = np.uint8(r)
cv2.imshow("16*16caiyang",new_img)
cv2.waitKey()
cv2.destroyAllWindows()

"""

"""
上面介绍的代码是对整幅图像进行采样处理，那么图和对图像的局部区域进行马赛克处理呢？下面的代码就实现了该功能。当单击鼠标时，他能够给鼠标拖动的区域打上马赛克，按下s
键可以保存图像至本地。
"""
# ---------局部马赛克--------------
"""
import cv2

img = cv2.imread("tuxiangchuli.jpg")
img = cv2.resize(img,None,fx=0.2,fy=0.2)

# 图像局部采样操作

def drawMask(x,y,size=10):

    m = int(x/size * size)
    n = int(y/size * size)

    for i in range(size):
        for j in range(size):
            img[m+i][n+j] = img[m][n]


# 设置单击鼠标开启
en = False
# 鼠标事件
def draw(event,x,y,flags,param):
    global en # 将变量en设置为全局变量
    # 单击鼠标开启en键
    if event == cv2.EVENT_LBUTTONDOWN: # 判断鼠标是否点击
        en = True
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_LBUTTONDOWN:
        # 调用函数打马赛克
        if en:
            drawMask(y,x)
        elif event == cv2.EVENT_LBUTTONUP:
            en = False

# 打开对话框

cv2.namedWindow("image")
# 调用draw函数设置鼠标操作
cv2.setMouseCallback("image",draw)

# 循环处理
while(1):
    cv2.imshow("image",img)
    # 按esc键退出
    if cv2.waitKey(10) & 0xFF==27:
        break
    # 按 s 键保存图片
    elif cv2.waitKey(10) & 0xFF==115:
        cv2.imwrite("masaike.png",img)

cv2.destroyAllWindows()
"""

"""
3、图像金字塔
图像金字塔是指由一组图像且不同分辨率的子图集合，它是图像多尺度表达的一种，以多分辨率来解释图像的结构，主要用于图像的分隔或压缩。
一幅图像的金字塔是一系列以金字塔形状排列的分辨率逐步降低，且来源于同一幅原始图像的图像集合。
生成图像金字塔主要包括两种方式，向下采样和向上采样。
"""
"""
4、下采样：
图像金字塔可以通过梯次向下采样获得，直到达到某个终止条件停止采样，在向下采样中，层级越高，图像越小，分辨率月底。
在图像下采样中，使用最多的是高斯金字塔。
它将图像进行高斯核卷积，并删除原图中所有的偶数行和列，最终缩小图像。其中，高斯核卷积运算就是对整幅图像进行加权平均的过程，每一个像素点的值，都由其本身和
邻域内的其他像素值（权值不同）经过加权平均后得到。
高斯核卷积让临近中心的像素点具有更高的重要度，对周围像素计算加权平均。

向下采样使用的函数为：
cv2.pyrDown（src,[,dst[,dstsize[,borderType]]]）
其中，src表示输入图像
dst表示输出图像，与输入图像具有一样的尺寸和类型
dstsize表示输出图像的大小，默认值为size（）
bordertype表示像素外推法
"""

#-----------------图像下采样--------------
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("tuxiangchuli.jpg")

r = cv2.pyrDown(img)
r1 = cv2.pyrDown(r)
r2 = cv2.pyrDown(r1)
# 设置正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
titles = ['原图','下采样-1','下采样-2','下采样-3']
images = [img,r,r1,r2]
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
plt.show()
"""

"""
5、图像上采样：
图像上采样是由小图像不断放大图像的过程，它将图像在每个方向上扩大为原图像的两倍，新增的行和列用0来填充，并使用
与下采样相同的卷积核乘以4，再与放大后的图像进行卷积运算，以获得新增像素的新值。
上采样与下采样不是互逆的操作。

上采样使用函数cv2.pyrUp（src,[,dst[,dstsize[,borderType]]]）

"""

#---------------上采样-----------------
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("tuxiangchuli.jpg")
img = cv2.resize(img,None,fx=0.1,fy=0.1)
r = cv2.pyrUp(img)
r1 = cv2.pyrUp(r)
r2 = cv2.pyrUp(r1)

plt.rcParams['font.sans-serif'] = ['SimHei']

titles = ['原图','一次上采样','二次上采样','三次上采样']
images= [img,r,r1,r2]

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
plt.show()
"""