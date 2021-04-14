from math import log
import operator


class decision_tree:
    @staticmethod
    # 如果到叶节点，种类还不统一，那么挑选出种类最多的作为当前叶节点的类型
    def majorityCnt(classList):
        classCount = dict()
        for vote in classList:
            if vote not in classCount.keys(): classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(classCount, key=lambda x: classCount[x], reverse=True)
        return sortedClassCount[0]

    @staticmethod
    # 计算熵
    def calcShannonEnt(dataSet):
        total_num = len(dataSet)
        dict_count = dict()
        for i in range(total_num):
            cur_label = dataSet[i][-1]
            dict_count[cur_label] = dict_count.get(cur_label, 0) + 1
        shannonEnt = 0
        for key in dict_count:
            prob_cur = float(dict_count[key]) / total_num
            shannonEnt -= prob_cur * log(prob_cur, 2)
        return shannonEnt

    @staticmethod
    # 根据特征axia取值是否为value进行划分，并据此计算信息增益
    def split_dataset(dataset, axis, value):
        current_feature_list = list()
        for feature_vec in dataset:
            if feature_vec[axis] == value:
                current_feature_list.append(feature_vec)
        return current_feature_list

    @staticmethod
    def chooseBestFeatureToSplit(dataset):
        num_feature = len(dataset[0]) - 1
        entroy_of_data = decision_tree.calcShannonEnt(dataset)
        best_entroy = 0.0
        best_feature = -1
        for i in range(num_feature):
            cur_feature_list = [single_data[i] for single_data in dataset]
            kind_of_cur_feature = set(cur_feature_list)
            condition_entroy_data = 0
            for feature in kind_of_cur_feature:
                split_dataset = decision_tree.split_dataset(dataset, i, feature)
                prob = len(split_dataset) / float(len(dataset))
                condition_entroy_data += prob * decision_tree.calcShannonEnt(split_dataset)
            info_gain = entroy_of_data - condition_entroy_data
            if info_gain > best_entroy:
                best_entroy = info_gain
                best_feature = i
        return best_feature


def creatDataset():
    dataset = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no'],
    ]
    labels = ['no surfacing', 'flippers']
    return dataset, labels


if __name__ == '__main__':
    a = decision_tree()
    data, labels = creatDataset()
    ent = a.split_dataset(data, 0, 1)
    print(ent)
    ent = a.chooseBestFeatureToSplit(data)
    print(ent)
