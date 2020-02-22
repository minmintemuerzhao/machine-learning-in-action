from numpy import *
import numpy as np
from KNN.kNN import *

# def kNN_classifier(inX, dataSet, labels, k):
#     dataSetSize = dataSet.shape[0]
#     diffMat = tile(inX, (dataSetSize, 1)) - dataSet
#     sqdiff = diffMat ** 2
#     sqdisstance = sqdiff.sum(axis=1)
#     distance = sqdisstance ** 0.5
#     sortedDistance = distance.argsort()
#     classCount = dict()
#     for i in range(k):
#         voteIlabel = labels[sortedDistance[i]]
#         classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1


'''
第一部分的
'''
dict1 = {'a': 10, 'b': 12, 'c': 14, 'd': 9, 'e': 2}
dict3 = sorted(dict1)
dict2 = sorted(dict1, key=lambda x: dict1[x], reverse=True)

'''
第二部分的
'''


# txt2numpy
def file2matrix(filename):
    data_reader = open(filename)
    data = data_reader.readlines()
    lens = len(data)
    matrix = np.zeros((lens, 3))
    vector_data = []
    for i in range(lens):
        data_tmp = data[i].strip().split('\t')
        matrix[i, :] = data_tmp[0:3]
        vector_data.append(int(data_tmp[-1]))
    return matrix, vector_data


# file2matrix('datingTestSet2.txt')

def datingClassTest():
    data, label = file2matrix('datingTestSet2.txt')
    norm_data, range_data, mindata = autoNorm(data)
    test_rate = 0.1
    test_number = int(norm_data.shape[0] * test_rate)
    k = norm_data.shape[0]
    errorCount = 0
    for i in range(test_number):
        classifier_result = knn_classifier(norm_data[i, :], norm_data[test_number:k, :], label[test_number:k], 1)
        print('the classfier came back with: {}, teh real anwser is: {}'.format(classifier_result, label[i]))
        if classifier_result != label[i]:
            errorCount += 1
    print('the total result error rate is {}'.format(errorCount / float(test_number)))


if __name__ == '__main__':
    datingClassTest()

