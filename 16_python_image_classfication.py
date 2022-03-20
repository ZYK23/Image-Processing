#-*-conding:utf-8-*-
"""
Creaded on 2022年2月28日09:35:01
@Author Yk Zhang

第十六章 python图像分类
"""
"""
1、本章介绍常见的图像分类算法：贝叶斯图像分类算法、基于KNN算法的图像分类和基于SVM算法的图像分类
"""
"""
2、图像份额里是对图像内容进行分类，它利用计算机对图像进行定量分析，把图像或图像中的区域划分为若干个类别，以代替人的视觉盘判断。
图像分类的传统方法是特征描述及检测，这类传统方法对于一些简单的图像分类可能是有效的，但由于实际情况非常复杂，传统的分类方法不堪重负。现在
广泛使用机器学习和深度学习的方法来处理图像分类问题，其主要任务是给定一堆输入图片，将其指派到一个已知的混合类别中的某个标签。
"""
"""
3、图像分类首先是输入一堆图像的像素值数组，然后给它分配一个分类标签，通过训练学习来建立算法模型，最后使用该模型进行图像分类预测。
"""
"""
4、常见的分类算法：朴素贝叶斯分类器、决策树、K最近邻分类算法、支持向量机、神经网络和基于规则的分类算法等，同时还有用于组合单一类方法的集成学习算法，如Bagging和Boosting等。
"""
"""
5、朴素贝叶斯分类算法 https://blog.csdn.net/u013850277/article/details/83996358
利用贝叶斯定理来预测一个未知类别的样本属于各个类别的可能性，选择其中可能性最大的一个类别作为该样本的最终类别。在朴素贝叶斯分类模型中，它将为每一个类别的特征向量建立服从正态分布的函数，给定训练数据，
算法将会估计每一个类别的向量均值和方差矩阵，然后根据这些进行预测。
朴素贝叶斯份分类模型的正式定义如下：
1、设x = {a1,a2,...,am}为一个待分类项，而每个a为x的一个特征属性
2、有类别集合C={y1,y2,...,yn}
3、计算P(y1|x),P(y2|x),P(yn|x)
4、如果P(yk|x) = max{P(y1|x),P(y2|x),P(yn|x)},则x∈yk
该算法的特点为：如果没有很多数据，该模型会比很多复杂的模型获得更好的性能，因为复杂的模型用了太多假设，以致产生欠拟合。
"""
"""
6、KNN分类算法
K最近邻分类算法是一种基于实例的分类方法，是数据挖掘分类技术中最简单常用的方法之一。该算法的核心思想如下。一个样本x与样本集中的k个最相邻的样本
中的大多数属于某一个类别Ylabel，那么该样本x也属于类别ylabel，并具有这个类别样本的特性。简而言之，一个样本与数据集中的k个最相邻样本中的大多数的类别相同。
由其思想可以看出，KNN是通过测量不同特征值之间的距离进行分类，而且在决策样本类别时，只参考样本周围K个邻居样本的所属类别。因此，
比较适合处理样本集存在较多重叠的场景，主要用于预测分析、文本分类、降维等处理。
该算法在建立训练集时，就要确定训练数据及其对应的类别标签；然后把待分类的测试数据与训练数据一次进行特征比较，从训练集中挑选出最相邻的k个数据，着k个数据中投票最多的分类，即为新样本的类别。
KNN分类算法的流程如下：
1、计算测试数据与各个训练数据之间的距离
2、对距离从小到大进行排序
3、选取距离最小的k个点
4、确定前k个点类别的出现频率
5、出现频率最高的类别作为预测分类

该算法的特点为：简单有效，但因为需要存储所有的训练集，占用很大内存，速度相对较慢，使用该方法前通常需要进行降维处理。
"""

"""
7、SVM分类算法
支持向量机（SVM）其基本模型定义为特征空间上间隔最大的线性分类器，学习策略是间隔最大化，最终转换为一个凸二次规划问题的求解。
SVM分类算法基于核函数把特征向量映射到高威空间，建立一个线性判别函数，解最优在某种意义上是两类中距离分割面最近的特征向量和分割面的距离最大化。离分割面最近的特征向量称为
支持向量，即其他向量不影响分割面。
该算法的特点为：当数据集比较小时，SVM的效果非常好。同时，SVM分类算法较好地解决了非线性、高维数、局部极小点等问题，维数大于样本数时仍然有效。
"""
"""
8、随机森林分类算法
随机森林是用随机的方式建立一个森林，在森林里有很多决策树，并且每一颗决策树之间是没有关联的。
当有一个新样本出现时，通过森林中的每一颗决策树分别进行判断，看看这个样本属于哪一类，然后用投票的方式，决定哪一类被选择的多，并作为最终的分类结果。

随机森林中的每一个决策树“种植”和“生长”主要包括以下四个步骤：
1、假设训练集中的样本个数为N，通过有重置的重复多次抽样获取这N个样本，抽样结果将作为生成决策树的训练集。
2、如果有M个输入变量，每个节点都将随机选择m（m<M）个特定的变量，然后运用这m个变量来确定最佳的分裂点。在决策树的生成过程中，m值是保持不变的。
3、每棵决策树都最大可能地进行生长而不尽心剪枝
4、通过对所有的决策树进行加总来预测新的数据（在分类时采用多数投票，在回归时采用平均）。
该算法的特点为：在分类和回归分析中都表现良好；对高维数据的处理能力强，可以处理成千上万个输入变量，也是一个非常不错的降维方法；能够输出特征的重要程度，能够有效地处理默认值。
"""
"""
9、神经网络分类算法
神经网络是对非线性可分数据的分类方法，通常包括输入层、隐藏层、输出层。其中与输入直接相连的称为隐藏层，与输出直接相连的称为输出层。
神经网络算法的特点是有比较多的局部最优值，可通过多次随机设定初始值并运行梯度下降算法获得最优值。
图像分类中使用最广泛的是BP神经网络和卷积神经网络。
BP神经网络是一种多层的前馈神经网络，其主要特点为：信号是前向传播的，而误差是反向传播的。BP神经网络的过程主要分为两个阶段法，第一阶段是
信号的前向传播，第二阶段是误差反向传播，依次调节隐藏层到输出层的权重和偏置。
卷积神经网络（CNN）是一类包含卷积计算且具有深度结构的前馈神经网络，是深度学习的代表算法之一。
"""

#-------------------------------------------代码实现----------------------------------------------------
"""
10、主要使用Scikit-Learn包进行python图像分类处理。该扩展包适用于python数据挖掘和数据分析的经典、实用扩展包。该包中
的机器学习模型是非常丰富的，包括线性回归、决策树、SVM、K-Means、KNN、PCA等
"""
#----------------------------------------朴素贝叶斯分类---------------------------------------------------------
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report

#-------------------------------------------------------------------------------------------
#第一步 切分训练集和测试集
#--------------------------------------------------------------------------------------------

x = [] # 定义图像名称   x与y一一对应
y = [] # 定义图像分类类标
z = [] # 定义图像像素
for i in range(len(os.listdir("photo"))):
    # 遍历文件夹，读取图像
    for f in os.listdir("photo/%s"%i): # 等价于photo/i
        # 获取图像名称
        x.append("photo//"+str(i)+"//"+str(f))
        y.append(i)
x = np.array(x)
y = np.array(y)

# 随机率为100% 选取其中30%作为测试集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)
#----------------------------------------
# 第二步 图像读取及转换为像素直方图
#----------------------------------------

# 训练集
xx_train_fic= []
for i in x_train:
    image = cv2.imread(i)
    img = cv2.resize(image,(256,256),interpolation=cv2.INTER_CUBIC)
    hist = cv2.calcHist([img],[0,1],None,[256,256],[0,255,0,255])
    xx_train_fic.append((hist/255).flatten()) # 直方图除以255，作为归一化处理

# 测试集
xx_test_fic= []
for i in x_test:
    image = cv2.imread(i)
    img = cv2.resize(image,(256,256),interpolation=cv2.INTER_CUBIC)
    hist = cv2.calcHist([img],[0,1],None,[256,256],[0,255,0,255])
    xx_test_fic.append((hist/255).flatten()) # 二维直方图转换为一维 除以255，作为归一化处理

#--------------------------------------------
# 基于朴素贝叶斯的图像分类处理 https://codeantenna.com/a/8K86ZSOZoc  https://blog.aimoon.top/4mlmethd/
#-------------------------------------------
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB().fit(xx_train_fic,y_train) # 训练
predictions_labels = clf.predict(xx_test_fic)

print("----------------------基于朴素贝叶斯分类结果---------------------------------------------------")
# 结果
print("预测结果",predictions_labels)
# 评估
print("评估",classification_report(y_test,predictions_labels))

#--------------------------------------------
# 基于KNN算法的图像分类处理
#-------------------------------------------

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=11).fit(xx_train_fic,y_train)
predictions_labels = clf.predict(xx_test_fic)

print("----------------------基于KNN算法分类结果---------------------------------------------------")
# 结果
print("预测结果",predictions_labels)
# 评估
print("评估",classification_report(y_test,predictions_labels))

"""
使用KNN算法进行图像分类实验，最后算法评价的准确率、召回率和F值略差于朴素贝叶斯的图像分类算法
"""
"""
下面是基于BP神经网络算法的图像分类代码，主要通过自定义的神经网络实现图像分类。
它的基本思想为：首先计算每一层的状态和激活值，直到最后一层（即信号是前向传播的）；
然后计算每一层的误差，误差的计算过程是从最后一层向前推进的(反向传播)；
最后更新参数（目标是误差变小），迭代前面两个步骤，直到满足停止准则，如相邻两次迭代的误差的差别很小。
"""
#--------------------------------------------
# 基于神经网络的图像分类处理
#-------------------------------------------
import datetime

starttime = datetime.datetime.now()

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import os
import cv2

X = []
Y = []

for i in range(0, 10):
    # 遍历文件夹，读取图片
    for f in os.listdir("./photo/%s" % i):
        # 打开一张图片并灰度化
        Images = cv2.imread("./photo/%s/%s" % (i, f))
        image = cv2.resize(Images, (256, 256), interpolation=cv2.INTER_CUBIC)
        hist = cv2.calcHist([image], [0, 1], None, [256, 256], [0.0, 255.0, 0.0, 255.0])
        X.append((hist / 255).flatten())
        Y.append(i)
X = np.array(X)
Y = np.array(Y)
# 切分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

from sklearn.preprocessing import LabelBinarizer
import random


def logistic(x):
    return 1 / (1 + np.exp(-x))


def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))


class NeuralNetwork:

    def predict(self, x):
        for b, w in zip(self.biases, self.weights):
            # 计算权重相加再加上偏向的结果
            z = np.dot(x, w) + b
            # 计算输出值
            x = self.activation(z)
        return self.classes_[np.argmax(x, axis=1)]


class BP(NeuralNetwork):

    def __init__(self, layers, batch):

        self.layers = layers
        self.batch = batch
        self.activation = logistic
        self.activation_deriv = logistic_derivative

        self.num_layers = len(layers)
        self.biases = [np.random.randn(x) for x in layers[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(layers[:-1], layers[1:])]

    def fit(self, X, y, learning_rate=0.1, epochs=1):

        labelbin = LabelBinarizer()
        y = labelbin.fit_transform(y)
        self.classes_ = labelbin.classes_
        training_data = [(x, y) for x, y in zip(X, y)]
        n = len(training_data)
        for k in range(epochs):
            # 每次迭代都循环一次训练
            # 搅乱训练集，让其排序顺序发生变化
            random.shuffle(training_data)
            batches = [training_data[k:k + self.batch] for k in range(0, n, self.batch)]
            # 批量梯度下降
            for mini_batch in batches:
                x = []
                y = []
                for a, b in mini_batch:
                    x.append(a)
                    y.append(b)
                activations = [np.array(x)]
                # 向前一层一层的走
                for b, w in zip(self.biases, self.weights):
                    # 计算激活函数的参数,计算公式：权重.dot(输入)+偏向
                    z = np.dot(activations[-1], w) + b
                    # 计算输出值
                    output = self.activation(z)
                    # 将本次输出放进输入列表，后面更新权重的时候备用
                    activations.append(output)
                # 计算误差值
                error = activations[-1] - np.array(y)
                # 计算输出层误差率
                deltas = [error * self.activation_deriv(activations[-1])]

                # 循环计算隐藏层的误差率,从倒数第2层开始
                for l in range(self.num_layers - 2, 0, -1):
                    deltas.append(self.activation_deriv(activations[l]) * np.dot(deltas[-1], self.weights[l].T))

                # 将各层误差率顺序颠倒，准备逐层更新权重和偏向
                deltas.reverse()
                # 更新权重和偏向
                for j in range(self.num_layers - 1):
                    # 权重的增长量，计算公式，增长量 = 学习率 * (错误率.dot(输出值)),单个训练数据的误差
                    delta = learning_rate / self.batch * (
                        (np.atleast_2d(activations[j].sum(axis=0)).T).dot(np.atleast_2d(deltas[j].sum(axis=0))))
                    # 更新权重
                    self.weights[j] -= delta
                    # 偏向增加量，计算公式：学习率 * 错误率
                    delta = learning_rate / self.batch * deltas[j].sum(axis=0)
                    # 更新偏向
                    self.biases[j] -= delta
        return self


clf0 = BP([X_train.shape[1], 10], 10).fit(X_train, y_train, epochs=10)
predictions_labels = clf0.predict(X_test)
print(confusion_matrix(y_test, predictions_labels))
print(classification_report(y_test, predictions_labels))
endtime = datetime.datetime.now()
print(endtime - starttime)
