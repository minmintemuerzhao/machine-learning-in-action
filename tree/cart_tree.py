from numpy import nonzero
import numpy as np


def loadDataSet(fileName):
    """load数据"""
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat


def regLeaf(dataSet):
    return np.mean(dataSet[:, -1])


def regErr(dataSet):
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]


class CreateTreeClass:
    """该类用于回归树构建"""

    @staticmethod
    def binSplitDataSet(dataSet, feature, value):
        """数据集切分"""
        mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
        mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
        return mat0, mat1

    @staticmethod
    def chooseBestSplit(dataSet, leafType, errType, ops):
        tolS = ops[0]
        tolN = ops[1]
        if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
            return None, leafType(dataSet)
        m, n = np.shape(dataSet)
        S = errType(dataSet)
        bestS = np.inf
        bestIndex = 0
        bestValue = 0
        for featIndex in range(n - 1):
            for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):
                mat0, mat1 = CreateTreeClass.binSplitDataSet(dataSet, featIndex, splitVal)
                if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
                    continue
                newS = errType(mat0) + errType(mat1)
                if newS < bestS:
                    bestIndex = featIndex
                    bestValue = splitVal
                    bestS = newS
        if (S - bestS) < tolS:
            return None, leafType(dataSet)
        mat0, mat1 = CreateTreeClass.binSplitDataSet(dataSet, bestIndex, bestValue)
        if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
            return None, leafType(dataSet)
        return bestIndex, bestValue

    @staticmethod
    def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
        feat, val = CreateTreeClass.chooseBestSplit(dataSet, leafType, errType, ops)
        if feat == None:
            return val
        retTree = {}
        retTree["spInd"] = feat
        retTree["spVal"] = val
        lSet, rSet = CreateTreeClass.binSplitDataSet(dataSet, feat, val)
        retTree['left'] = CreateTreeClass.createTree(lSet, leafType, errType, ops)
        retTree['right'] = CreateTreeClass.createTree(rSet, leafType, errType, ops)
        return retTree


class CropTree():
    """该类用于回归树减枝"""

    @staticmethod
    def isTree(obj):
        return type(obj).__name__ == 'dict'

    @staticmethod
    def getMean(tree):
        if CropTree.isTree(tree['right']):
            tree['right'] = CropTree.getMean(tree['right'])
        if CropTree.isTree(tree['left']):
            tree['left'] = CropTree.getMean(tree['left'])
        return (tree['left'] + tree['right']) / 2

    @staticmethod
    def prune(tree, testData):
        if np.shape(testData)[0] == 0:
            return CropTree.getMean(tree)
        if CropTree.isTree(tree['right']) or CropTree.isTree(tree['left']):
            lSet, rSet = CreateTreeClass.binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        if CropTree.isTree(tree['left']):
            tree['left'] = CropTree.prune(tree['left'], lSet)
        if CropTree.isTree(tree['right']):
            tree['right'] = CropTree.prune(tree['right'], rSet)
        if not CropTree.isTree(tree['left']) and not CropTree.isTree(tree['right']):
            lSet, rSet = CreateTreeClass.binSplitDataSet(testData, tree['spInd'], tree['spVal'])
            errorNoMerge = sum(np.power(lSet[:, -1] - tree['left'], 2)) + sum(np.power(rSet[:, -1] - tree['right'], 2))
            treeMean = (tree['left'] + tree['right']) / 2
            errorMerge = sum(np.power(testData[:, -1] - treeMean, 2))
            if errorMerge < errorNoMerge:
                print("merging")
                return treeMean
            else:
                return tree
        else:
            return tree


if __name__ == '__main__':
    myDat = loadDataSet('ex2.txt')
    myMat = np.mat(myDat)
    tree = CreateTreeClass().createTree(myMat, ops=(0, 1))
    mytestDat = loadDataSet('ex2test.txt')
    mytestMat = np.mat(mytestDat)
    CropTree().prune(tree, mytestMat)
