import numpy as np


def show_point(dataMat, labelMat, alpha, b):
    # 画图查看
    labelMat = np.mat(labelMat).transpose()
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xcord1, ycord1, xcord2, ycord2 = [], [], [], []
    for index, line in enumerate(dataMat):
        if labelMat[index] == 1:
            xcord1.append(line[0])
            ycord1.append(line[1])
        else:
            xcord2.append(line[0])
            ycord2.append(line[1])
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    w = np.sum(np.multiply(np.mat(dataMat), np.multiply(alpha, np.mat(labelMat))), axis=0)
    x = np.arange(-1, 13, 0.1)
    y = (-b.A[0][0] - np.array(w)[0][0] * x) / np.array(w)[0][1]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


# 简化版的svm，使用简化的smo算法进行参数更新
class SVM:
    """SVM的相关函数."""

    @staticmethod
    def loadDataSet(fileName):
        """加载数据."""
        dataMat = []
        labelMat = []
        fr = open(fileName)
        for line in fr.readlines():
            lineArr = line.strip().split('\t')
            dataMat.append([float(lineArr[0]), float(lineArr[1])])
            labelMat.append(float(lineArr[2]))
        return dataMat, labelMat

    @staticmethod
    def selectJrand(i, m):
        """i是第一个选择的alpha的下标，m是所有alpha的个数，
        然后随机的从m个数值中选择一个不等于i的变量（命名为j），和i一起进行优化"""
        j = i
        while j == i:
            j = int(np.random.uniform(0, m))
        return j

    @staticmethod
    def clipAlpha(aj, H, L):
        """对上下界进行截断.

        主要是用于：在smo优化的过程中，由于有限制条件 0<= a <=C，
        因此在更新a的时候，需要据此对上下界进行截断。
        """
        if aj > H:
            aj = H
        if aj < L:
            aj = L
        return aj

    @staticmethod
    def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
        """简易的smo算法."""
        dataMatrix = np.mat(dataMatIn)
        labelMat = np.mat(classLabels).transpose()
        b = 0
        m, n = np.shape(dataMatrix)
        alphas = np.mat(np.zeros((m, 1)))
        iter = 0
        while iter < maxIter:
            alphaPairsChanged = 0
            for i in range(m):
                fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
                Ei = fXi - float(labelMat[i])
                if (fXi < 1 and alphas[i] == 0) or (fXi > 1 and alphas[i] == C) or (0 < alphas[i] < C and fXi != 1):
                    # if (labelMat[i] * Ei < -toler and alphas[i] < C) or (
                    #         labelMat[i] * Ei > toler and alphas[i] > 0):
                    j = SVM().selectJrand(i, m)
                    fXj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                    Ej = fXj - float(labelMat[j])
                    alphaIold = alphas[i].copy()
                    alphaJold = alphas[j].copy()
                    if labelMat[i] != labelMat[j]:
                        L = max(0, alphas[j] - alphas[i])
                        H = min(C, C + alphas[j] - alphas[i])
                    else:
                        L = max(0, alphas[j] + alphas[i] - C)
                        H = min(C, alphas[i] + alphas[j])
                    if L == H:
                        print("L ==H")
                        continue
                    eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - \
                          dataMatrix[j, :] * dataMatrix[j, :].T
                    if eta >= 0:
                        print("eta>=0")
                        continue
                    alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                    alphas[j] = SVM().clipAlpha(alphas[j], H, L)
                    if abs(alphas[j] - alphaJold) < 0.00001:
                        print("j not moving enough")
                        continue
                    alphas[i] += labelMat[j] * labelMat[i] * \
                                 (alphaJold - alphas[j])
                    b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * \
                         dataMatrix[i, :] * dataMatrix[i, :].T - \
                         labelMat[j] * (alphas[j] - alphaJold) * \
                         dataMatrix[i, :] * dataMatrix[j, :].T
                    b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * \
                         dataMatrix[i, :] * dataMatrix[j, :].T - \
                         labelMat[j] * (alphas[j] - alphaJold) * \
                         dataMatrix[j, :] * dataMatrix[j, :].T
                    if alphas[i] > 0 and alphas[i] < C:
                        b = b1
                    elif alphas[j] > 0 and alphas[j] < C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2.0
                    alphaPairsChanged += 1
                    print(f"iter: {iter}, i: {i}, pairs changed {alphaPairsChanged}")
            if alphaPairsChanged == 0:
                iter += 1
            else:
                iter = 0
            print(f"iteration number: {iter}")
        return b, alphas


# 完整的svm，使用真正的smo算法进行优化
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        """ 完整的smo算法-初始化参数"""
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))


class WholeSVM:

    @staticmethod
    def clipAlpha(aj, H, L):
        """对上下界进行截断.

        主要是用于：在smo优化的过程中，由于有限制条件 0<= a <=C，
        因此在更新a的时候，需要据此对上下界进行截断。
        """
        if aj > H:
            aj = H
        if aj < L:
            aj = L
        return aj

    @staticmethod
    def calcEk(oS, k):
        """计算Ek"""
        fXk = float(np.multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b
        Ek = fXk - float(oS.labelMat[k])
        return Ek

    @staticmethod
    def selectJ(i, oS: optStruct, Ei):
        """选择合适的J，使得Ei-Ej的绝对差值最大."""
        maxK = -1
        maxDeltaE = 0
        Ej = 0
        oS.eCache[i] = [1, Ei]
        validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
        if len(validEcacheList) > 1:
            for k in validEcacheList:
                if k == i:
                    continue
                Ek = WholeSVM.calcEk(oS, k)
                deltaE = abs(Ei - Ek)
                if deltaE > maxDeltaE:
                    maxK = k
                    maxDeltaE = deltaE
                    Ej = Ek
            return maxK, Ej
        else:
            j = WholeSVM.selectJrand(i, oS.m)
            Ej = WholeSVM.calcEk(oS, i)
        return j, Ej

    @staticmethod
    def selectJrand(i, m):
        """i是第一个选择的alpha的下标，m是所有alpha的个数，
        然后随机的从m个数值中选择一个不等于i的变量（命名为j），和i一起进行优化"""
        j = i
        while j == i:
            j = int(np.random.uniform(0, m))
        return j

    @staticmethod
    def updateEk(oS, k):
        """更新Ek."""
        Ek = WholeSVM.calcEk(oS, k)
        oS.eCache[k] = [1, Ek]

    @staticmethod
    def innerL(i, oS: optStruct):
        Ei = WholeSVM.calcEk(oS, i)
        if (oS.labelMat[i] * Ei < -oS.tol and oS.alphas[i] < oS.C) or (
                oS.labelMat[i] * Ei > oS.tol and oS.alphas[i] > 0):
            j, Ej = WholeSVM.selectJ(i, oS, Ei)
            alphaIold = oS.alphas[i].copy()
            alphaJold = oS.alphas[j].copy()
            if oS.labelMat[i] != oS.labelMat[j]:
                L = max(0, oS.alphas[j] - oS.alphas[i])
                H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
            else:
                L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
                H = min(oS.C, oS.alphas[j] + oS.alphas[i])
            if L == H:
                print("L==H")
                return 0
            eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
            if eta > 0:
                print("eta>0")
                return 0
            oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
            oS.alphas[j] = WholeSVM.clipAlpha(oS.alphas[j], H, L)
            WholeSVM.updateEk(oS, j)
            if abs(oS.alphas[j] - alphaJold) < 0.00001:  # 说明j不再更新了，自然也就没必要再继续往下算了
                print("j not moving enough")
                return 0
            oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
            WholeSVM.updateEk(oS, i)
            b1 = oS.b - Ei - labelMat[i] * (oS.alphas[i] - alphaIold) * \
                 oS.X[i, :] * oS.X[i, :].T - \
                 labelMat[j] * (oS.alphas[j] - alphaJold) * \
                 oS.X[i, :] * oS.X[j, :].T
            b2 = oS.b - Ej - labelMat[i] * (oS.alphas[i] - alphaIold) * \
                 oS.X[i, :] * oS.X[j, :].T - \
                 labelMat[j] * (oS.alphas[j] - alphaJold) * \
                 oS.X[j, :] * oS.X[j, :].T
            if oS.alphas[i] > 0 and oS.alphas[i] < oS.C:
                oS.b = b1
            elif oS.alphas[j] > 0 and oS.alphas[j] < oS.C:
                oS.b = b2
            else:
                oS.b = (b1 + b2) / 2.0
            return 1
        return 0

    @staticmethod
    def smoP(dataMatin, classLabels, C, toler, maxIter, kTup=('lin', 0)):
        oS = optStruct(np.mat(dataMatin), np.mat(classLabels).transpose(), C, toler)
        iter = 0
        entireSet = True
        alphaPairsChanged = 0
        while iter < maxIter and (alphaPairsChanged > 0 or entireSet):
            alphaPairsChanged = 0
            if entireSet:
                for i in range(oS.m):
                    alphaPairsChanged += WholeSVM.innerL(i, oS)
                    print(f"fullSet, iter {iter}: {i}, pairs changed {alphaPairsChanged}")
                iter += 1
            else:
                # 选出支持向量nonBoundIs，如果alphaPairsChanged在else的循环部分执行完后为0,
                # 则代表所有的支持向量都不再更新，代表更新完成，
                # 此时需要将entireSet设置为True，再次进行上面的全量更新，找到新的异常点
                # 然后再看else部分是否又有了新的支持向量需要更新。
                nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
                for i in nonBoundIs:
                    alphaPairsChanged += WholeSVM.innerL(i, oS)
                    print(f"non-bound, iter {iter}: {i}, pairs changed {alphaPairsChanged}")
                iter += 1
            if entireSet:
                entireSet = False
            elif alphaPairsChanged == 0:
                entireSet = True
        return oS.b, oS.alphas


class kernelOptstruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.C = C
        self.labelMat = classLabels
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelSVM.kernelTrans(self.X, self.X[i, :], kTup)


class kernelSVM:
    """核函数svm"""

    # 核函数svm
    @staticmethod
    def kernelTrans(X, A, kTup):
        m, n = np.shape(X)
        K = np.mat(np.zeros((m, 1)))
        if kTup[0] == 'lin':
            K = X * A.T
        elif kTup[0] == 'rbf':
            for j in range(m):
                deltaRow = X[j, :] - A
                K[j] = deltaRow * deltaRow.T
            K = np.exp(K / (-1 * kTup[1] ** 2))
        else:
            raise NameError("核函数只支持lin和rbf")
        return K

    @staticmethod
    def loadDataSet(fileName):
        """加载数据."""
        dataMat = []
        labelMat = []
        fr = open(fileName)
        for line in fr.readlines():
            lineArr = line.strip().split('\t')
            dataMat.append([float(lineArr[0]), float(lineArr[1])])
            labelMat.append(float(lineArr[2]))
        return dataMat, labelMat

    @staticmethod
    def clipAlpha(aj, H, L):
        """对上下界进行截断.

        主要是用于：在smo优化的过程中，由于有限制条件 0<= a <=C，
        因此在更新a的时候，需要据此对上下界进行截断。
        """
        if aj > H:
            aj = H
        if aj < L:
            aj = L
        return aj

    @staticmethod
    def calcEk(oS, k):
        """计算Ek"""
        fXk = float(np.multiply(oS.alphas, oS.labelMat).T * oS.K[:, k]) + oS.b
        Ek = fXk - float(oS.labelMat[k])
        return Ek

    @staticmethod
    def selectJ(i, oS: kernelOptstruct, Ei):
        """选择合适的J，使得Ei-Ej的绝对差值最大."""
        maxK = -1
        maxDeltaE = 0
        Ej = 0
        oS.eCache[i] = [1, Ei]
        validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
        if len(validEcacheList) > 1:
            for k in validEcacheList:
                if k == i:
                    continue
                Ek = kernelSVM.calcEk(oS, k)
                deltaE = abs(Ei - Ek)
                if deltaE > maxDeltaE:
                    maxK = k
                    maxDeltaE = deltaE
                    Ej = Ek
            return maxK, Ej
        else:
            j = kernelSVM.selectJrand(i, oS.m)
            Ej = kernelSVM.calcEk(oS, i)
        return j, Ej

    @staticmethod
    def selectJrand(i, m):
        """i是第一个选择的alpha的下标，m是所有alpha的个数，
        然后随机的从m个数值中选择一个不等于i的变量（命名为j），和i一起进行优化"""
        j = i
        while j == i:
            j = int(np.random.uniform(0, m))
        return j

    @staticmethod
    def updateEk(oS, k):
        """更新Ek."""
        Ek = kernelSVM.calcEk(oS, k)
        oS.eCache[k] = [1, Ek]

    @staticmethod
    def innerL(i, oS: kernelOptstruct):
        Ei = kernelSVM.calcEk(oS, i)
        if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or (
                (oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
            j, Ej = kernelSVM.selectJ(i, oS, Ei)
            alphaIold = oS.alphas[i].copy()
            alphaJold = oS.alphas[j].copy()
            if oS.labelMat[i] != oS.labelMat[j]:
                L = max(0, oS.alphas[j] - oS.alphas[i])
                H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
            else:
                L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
                H = min(oS.C, oS.alphas[j] + oS.alphas[i])
            if L == H:
                print("L==H")
                return 0
            eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
            if eta > 0:
                print("eta>0")
                return 0
            oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
            oS.alphas[j] = kernelSVM.clipAlpha(oS.alphas[j], H, L)
            kernelSVM.updateEk(oS, j)
            if abs(oS.alphas[j] - alphaJold) < 0.00001:  # 说明j不再更新了，自然也就没必要再继续往下算了
                print("j not moving enough")
                return 0
            oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
            kernelSVM.updateEk(oS, i)
            b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * \
                 oS.K[i, i] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
            b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * \
                 oS.K[i, i] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
            if oS.alphas[i] > 0 and oS.alphas[i] < oS.C:
                oS.b = b1
            elif oS.alphas[j] > 0 and oS.alphas[j] < oS.C:
                oS.b = b2
            else:
                oS.b = (b1 + b2) / 2.0
            return 1
        else:
            return 0

    @staticmethod
    def smoP(dataMatin, classLabels, C, toler, maxIter, kTup=('lin', 0)):
        oS = kernelOptstruct(np.mat(dataMatin), np.mat(classLabels).transpose(), C, toler, kTup)
        iter = 0
        entireSet = True
        alphaPairsChanged = 0
        while iter < maxIter and (alphaPairsChanged > 0 or entireSet):
            alphaPairsChanged = 0
            if entireSet:
                for i in range(oS.m):
                    alphaPairsChanged += kernelSVM.innerL(i, oS)
                    print(f"fullSet, iter {iter}: {i}, pairs changed {alphaPairsChanged}")
                iter += 1
            else:
                # 选出支持向量nonBoundIs，如果alphaPairsChanged在else的循环部分执行完后为0,
                # 则代表所有的支持向量都不再更新，代表更新完成，
                # 此时需要将entireSet设置为True，再次进行上面的全量更新，找到新的异常点
                # 然后再看else部分是否又有了新的支持向量需要更新。
                nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
                for i in nonBoundIs:
                    alphaPairsChanged += kernelSVM.innerL(i, oS)
                    print(f"non-bound, iter {iter}: {i}, pairs changed {alphaPairsChanged}")
                iter += 1
            if entireSet:
                entireSet = False
            elif alphaPairsChanged == 0:
                entireSet = True
        return oS.b, oS.alphas


def testRbf(k1=1.3):
    dataArr, labelArr = kernelSVM.loadDataSet('testSetRBF.txt')
    kernel_type = 'rbf'
    b, alphas = kernelSVM.smoP(dataArr, labelArr, 200, 0.0001, 10000, (kernel_type, k1))
    # show_point(dataMat, labelArr, alphas, b)
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print(f"共有{np.shape(sVs)[0]}个支持向量")
    m, n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelSVM.kernelTrans(sVs, datMat[i, :], (kernel_type, k1))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print(f"训练错误率是: {float(errorCount) / m}")
    dataArr, labelArr = kernelSVM.loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat = np.mat(dataArr)
    # labelMat = np.mat(labelArr).transpose()
    m, n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernelSVM.kernelTrans(sVs, datMat[i, :], (kernel_type, k1))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print(f"测试错误率是{float(errorCount) / m}")


if __name__ == '__main__':
    dataMat, labelMat = SVM().loadDataSet('testSet.txt')
    print(dataMat)
    # 简化版的smo算法
    # b, alpha = SVM().smoSimple(dataMat, labelMat, 0.6, 0.001, 40)
    # 正式的smo算法
    # b, alpha = WholeSVM().smoP(dataMat, labelMat, 0.6, 0.001, 40)
    # print(b, alpha)
    # 用于显示
    # show_point(dataMat, labelMat, alpha, b)
    # 核函数svm
    testRbf()
