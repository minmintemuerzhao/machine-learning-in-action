from numpy import *
from numpy import array
import numpy as np
import operator


def knn_classifier(inx, dataset, label, k):
    number = dataset.shape[0]
    diff_ = np.tile(inx, (number, 1)) - dataset
    squa = diff_ ** 2
    distance = sum(squa, axis=1) ** 0.5
    index = distance.argsort()
    classCount = dict()
    for i in range(k):
        class_tmp = label[index[i]]
        classCount[class_tmp] = classCount.get(class_tmp, 0) + 1
    class_fenlei = sorted(classCount, key=lambda x: classCount[x], reverse=True)
    return class_fenlei[0]


def creatDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    label = ['A', 'A', 'B', 'B']
    return group, label


def file2matrix(filename):
    fr = open(filename)
    total_data = fr.readlines()
    total_data_number = len(total_data)
    returnMat = np.zeros((total_data_number, 3))
    classLabelVector = []
    for i, data in enumerate(total_data):
        data = data.strip()
        line = data.split('\t')
        # line = data.split(" ")
        returnMat[i, :] = line[:3]
        classLabelVector.append(int(line[-1]))
    return returnMat, classLabelVector


# test
finename = 'datingTestSet2.txt'
data, label = file2matrix(finename)


# 画图找规律
# import matplotlib.pyplot as plt
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(data[:, 1], data[:, 2], 15.0 * array(label), 15.0 * array(label))
# plt.show()

# 对数据进行归一化
def autoNorm(dataset):
    mindata = dataset.min(0)
    maxdata = dataset.max(0)
    range_data = maxdata - mindata
    m = dataset.shape[0]
    norm_data = (dataset - np.tile(mindata, (m, 1))) / np.tile(range_data, (m, 1))
    return norm_data, range_data, mindata

# def autoNorm(dataset):
#     mindata = dataset.min(0)
#     maxdata = dataset.max(0)
#     diff_data = maxdata - mindata
#     number_data = dataset.shape[0]
#     print(dataset[:10])
#     a = (dataset - mindata)
#     print(a[:10])
#     normdata = (dataset - mindata) / diff_data
#     return normdata
