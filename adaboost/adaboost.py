import numpy as np


def loadSimpleData():
    """构造简单的假数据进行测试使用"""
    dataMat = np.matrix([
        [1., 2.1],
        [2., 1.1],
        [1.3, 1.],
        [1., 1.],
        [2., 1.]
    ])
    classLabel = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabel


def stumpClassify(dataMattix, dimen, threshVal, threshIneq):
    retArray = np.ones((np.shape(dataMattix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMattix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMattix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    """构建单层决策树分类器"""
    dataMatrrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m, 1)))
    minError = float('inf')
    for i in range(n):
        rangeMin = dataMatrrix[:, i].min()
        rangeMax = dataMatrrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):  # 超过边界的也需要考虑，这时候就是将所有数据按照一种来对待
            for inequeal in ('lt', 'gt'):
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrrix, i, threshVal, inequeal)
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                print(f"split: dim:{i}, thresh: {threshVal}, thresh ineqal: {inequeal},"
                      f"the weight error is {float(weightedError)}")
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequeal
    return bestStump, minError, bestClasEst


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    """adaboost训练"""
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1)) / m)
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print("D:", D.T)
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 0.00000001)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print("classEst: ", classEst.T)
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / sum(D)
        aggClassEst += alpha * classEst  # 这个有待商榷
        print('aggClassEst: ', aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print('total error: ', errorRate)
        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst


def adaClassify(datToClass, classifierArr):
    """adaboost测试"""
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)
    return np.sign(aggClassEst)

def loadDataSet(fileName):
    """加载真实数据进行测试"""
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0, 1.0)
    ySum = 0
    numPosClas = sum(np.array(classLabels) == 1)
    yStep = 1 / float(numPosClas)
    xStep = 1 / float(len(classLabels) - numPosClas)
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0,1], [0, 1], 'b--')
    plt.xlabel("False Positive Rate")
    plt.ylabel('True Positive Rate')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print(f'The AUC is {ySum * xStep}')

if __name__ == '__main__':
    """1）构造简单数据进行模拟训练和测试"""
    # D = np.mat(np.ones((5, 1)) / 5)
    # dataMat, classLabel = loadSimpleData()
    # # result = buildStump(dataMat, classLabel, D)
    # # print(result)
    # classifierArray, aggClassEst = adaBoostTrainDS(dataMat, classLabel, 9)
    # # 进行测试
    # class_result = adaClassify([0, 0], classifierArray)
    # print(f'ok 分类结果是{class_result}')
    """2）使用真实数据进行训练和测试"""
    datArr, labelArr = loadDataSet("horseColicTraining2.txt")
    classifierArray, aggClassEst = adaBoostTrainDS(datArr, labelArr, 100)
    testArr, testLabelArr = loadDataSet("horseColicTest2.txt")
    prediction10 = adaClassify(testArr, classifierArray)
    errArr = np.mat(np.ones((len(testArr), 1)))
    errorRate = errArr[prediction10 != np.mat(testLabelArr).T].sum() / len(testArr)
    print(f"错误率是{errorRate}")
    # 画一下ROC曲线
    plotROC(aggClassEst.T, labelArr)

