import numpy as np
import operator


def create_dataset():
    """
    创建数据集
    :return: 特征及其标签
    """
    # 特征
    group = np.array([[1, 101], [5, 89], [108, 5], [115, 8]])
    # 标签
    label = ['爱情片', '爱情片', '动作片', '动作片']
    return group, label


def classify0(idx, dataSet, labels, k):
    """
    简单实现knn算法
    :param idx:
    :param dataSet:
    :param labels:
    :param k:
    :return:
    """
    data_set_size = dataSet.shape[0]
    diffMat = np.tile(idx, (data_set_size, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDisIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDisIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


if __name__ == '__main__':
    # 训练集
    group, labels = create_dataset()
    print(group)
    print(labels)
    # 测试集
    test = [101, 20]
    # knn分类
    test_class = classify0(test, group, labels, 3)
    # 分类结果
    print(test_class)
