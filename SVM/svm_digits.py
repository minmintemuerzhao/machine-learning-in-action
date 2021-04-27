import numpy as np
import os
from SVM.svm import kernelSVM

# 将（32,32）的图像数据转换成（1,1024）的向量来方便计算距离
def img2vector(img_data_path):
    total_data = list()
    total_label = list()
    for dir_path, dir_path_folder_list, dir_path_file_list in os.walk(img_data_path):
        for txt_name in dir_path_file_list:
            cur_data = np.zeros((1, 1024))
            cur_txt_path = dir_path + '/' + txt_name
            label = int(txt_name.split('_')[0])
            if label == 1:
                with open(cur_txt_path) as f:
                    cur = f.readlines()
                    for i, data in enumerate(cur):
                        for j in range(32):
                            cur_data[0, i * 32 + j] = int(data[j])
                total_data.append(cur_data)
                total_label.append(1)
            elif label == 9:
                with open(cur_txt_path) as f:
                    cur = f.readlines()
                    for i, data in enumerate(cur):
                        for j in range(32):
                            cur_data[0, i * 32 + j] = int(data[j])
                total_data.append(cur_data)
                total_label.append(-1)
            else:
                pass
    total_data = np.concatenate(total_data, axis=0)
    return total_data, total_label


def testDigits(kTup=('rbf', )):
    dataArr, labelArr = img2vector('digits/trainingDigits')
    b, alphas = kernelSVM.smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A>0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    m, n = np.shape(datMat)
    print(f"支持向量的个数为{len(svInd)}")
    errorCount = 0
    for i in range(m):
        kernelEval = kernelSVM.kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print(f"训练错误率是: {float(errorCount) / m}")
    dataArr, labelArr = img2vector('digits/testDigits')
    errorCount = 0
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    m, n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernelSVM.kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print(f"测试错误率是: {float(errorCount) / m}")
