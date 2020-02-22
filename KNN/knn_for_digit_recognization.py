import numpy as np
import os


class Data_recognization:
    def __init__(self, data_path_train, data_path_test, k):
        self.data_path_train = data_path_train
        self.data_path_test = data_path_test
        self.k = k

    @staticmethod
    # 将（32,32）的图像数据转换成（1,1024）的向量来方便计算距离
    def img2vector(img_data_path):
        total_data = list()
        total_label = list()
        for dir_path, dir_path_folder_list, dir_path_file_list in os.walk(img_data_path):
            for txt_name in dir_path_file_list:
                cur_data = np.zeros((1, 1024))
                cur_txt_path = dir_path + '/' + txt_name
                label = int(txt_name.split('_')[0])
                with open(cur_txt_path) as f:
                    cur = f.readlines()
                    for i, data in enumerate(cur):
                        for j in range(32):
                            cur_data[0, i * 32 + j] = int(data[j])
                total_data.append(cur_data)
                total_label.append(label)
        total_data = np.concatenate(total_data, axis=0)
        return total_data, total_label

    @staticmethod
    # knn算法细节
    def knn(data_pred_one, train_data, train_label, k):
        dist = np.sum((data_pred_one - train_data) ** 2, axis=1) ** 0.5
        arg_sort = dist.argsort()
        dict_result = dict()
        for i in range(k):
            dict_result[train_label[arg_sort[i]]] = dict_result.get(train_label[arg_sort[i]], 0) + 1
        knn_result = sorted(dict_result, key=lambda x: dict_result[x], reverse=True)
        return knn_result[0]

    # 计算手写数字识别的精度
    def hand_writing_class_by_knn(self):
        train_data, train_label = Data_recognization.img2vector(self.data_path_train)
        test_data, test_label = Data_recognization.img2vector(self.data_path_test)
        errorNumber = 0
        for i in range(test_data.shape[0]):
            result = Data_recognization.knn(test_data[i, :], train_data, train_label, self.k)
            print('the predict is {}, the real label is {}'.format(result, test_label[i]))
            if result != test_label[i]:
                errorNumber += 1
        print('the error rate is {}'.format(errorNumber / len(test_label)))


if __name__ == '__main__':
    a = Data_recognization('digits/trainingDigits', 'digits/testDigits', 3)
    a.hand_writing_class_by_knn()
