import numpy as np
import matplotlib.pyplot as plt


def plotBestFit(weights, data_set_path):
    """分类结果可视化, 专门用于对testSet.txt这个数据的可视化"""
    dataMat, labelMat = Logistic.loadDataSet(data_set_path)
    dataArr = np.array(dataMat)
    n = np.shape(dataMat)[0]
    xcord1, ycord1, xcord2, ycord2 = [], [], [], []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3, 3, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

class Logistic:
    """逻辑回归相关函数"""

    @staticmethod
    def loadDataSet(data_path):
        """读数据，在这是专门用于读取testSet.txt数据的"""
        dataMat = []
        labelMat = []
        fr = open(data_path)
        for line in fr.readlines():
            lineArr = line.strip().split()
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            labelMat.append(int(lineArr[2]))
        return dataMat, labelMat

    @staticmethod
    def sigmoid(inx):
        """sigmoid函数"""
        return 1.0 / (1 + np.exp(-inx))

    @staticmethod
    def grandAscent(dataMatIn, classLabels):
        """梯度上升算法"""
        dataMatrrix = np.mat(dataMatIn)
        labelMat = np.mat(classLabels).transpose()
        m, n = np.shape(dataMatrrix)
        alpha = 0.001
        maxCycles = 500
        weights = np.ones((n, 1))
        for k in range(maxCycles):
            h = Logistic.sigmoid(dataMatrrix * weights)
            error = (labelMat - h)
            weights = weights + alpha * dataMatrrix.transpose() * error
        return weights

    @staticmethod
    def stocGradAscent0(dataMatrix, classLabels):
        """随机梯度上升"""
        m, n = np.shape(dataMatrix)
        alpha = 0.01
        weights = np.ones(n)
        # 循环进行200次
        for i in range(200):
            for i in range(m):
                h = Logistic.sigmoid(sum(dataMatrix[i] * weights))
                error = h - classLabels[i]
                weights = weights - alpha * error * np.array(dataMatrix[i])
        return weights

    @staticmethod
    def stocGradAscent1(dataMatrix, classLabels, numIter=150):
        """随机梯度上升的优化版本

        1、增加了随机样本的选择
        2、学习率变成了一个动态变化的值
        """
        m, n = np.shape(dataMatrix)
        weights = np.ones(n)
        for i in range(numIter):
            dataIndex = list(range(m))
            for j in range(m):
                alpha = 4 / (1.0 + i + j) + 0.01
                randIndex = int(np.random.uniform(0, len(dataIndex)))
                h = Logistic.sigmoid(sum(dataMatrix[randIndex] * weights))
                error = h - classLabels[randIndex]
                weights = weights - alpha * error * np.array(dataMatrix[randIndex])
                del dataIndex[randIndex]
        return weights

    @staticmethod
    def clsssifyVector(inX, weights):
        """分类结果判断"""
        prob = Logistic.sigmoid(sum(inX * weights))
        return 0 if prob < 0.5 else 1


"""
# 该部分使用testSet.txt数据集来进行训练和预测
if __name__ == '__main__':
    dataMat, labelMat = Logistic.loadDataSet('testSet.txt')
    plotBestFit(np.array([1, 1, 1]), 'testSet.txt')
    # 梯度上升
    # weights = grandAscent(dataMat, labelMat)
    # 随机梯度上升
    weights = Logistic.stocGradAscent1(dataMat, labelMat)
    plotBestFit(weights.getA(), 'testSet.txt')
"""

# 使用马的数据集进行logistic
if __name__ == '__main__':
    def coliCTest():
        frTrain = open("horseColicTraining.txt")
        frTest = open("horseColicTest.txt")
        trainingSet = []
        trainingSetLabels = []
        for line in frTrain.readlines():
            currLine = line.strip().split('\t')
            trainingSet.append(list(map(lambda x: float(x), currLine[:21])))
            trainingSetLabels.append(float(currLine[-1]))
        weights = Logistic.stocGradAscent1(np.array(trainingSet), trainingSetLabels, 500)
        errorCount = 0
        numTestVec = 0.0
        for line in frTest.readlines():
            numTestVec += 1
            currLine = line.strip().split('\t')
            lineArr = list(map(lambda x: float(x), currLine[:21]))
            if int(Logistic.clsssifyVector(np.array(lineArr), weights)) != int(currLine[21]):
                errorCount += 1
        errorRate = float(errorCount) / numTestVec
        print(f"errorRate is {errorRate}")
        return errorRate

    numTests = 10
    errorSum = 0
    for k in range(numTests):
        errorSum += coliCTest()
    print(f"Total errorRate is {errorSum/float(numTests)}")