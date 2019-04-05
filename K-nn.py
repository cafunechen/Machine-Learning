
from numpy import *
import operator

def createDataSet():
    group = array([[0.1, 0.1], [0, 0], [1, 0.9]])
    lables = ['B', 'B', 'A']
    return group, lables

# 通过计算欧氏距离，返回在k个点中出现次数最多的label(得出预测结果)
def classify0(inX, dataSet, labels, k):
    # k means how many you need to get your P(x)
    dataSetSize = dataSet.shape[0]
    # tile(x,(a,b)) 把数组x在行上重复a次，列上重复b次
    DiffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = DiffMat**2
    sqDistances = sqDiffMat.sum(axis=1) # 求出每行的sum
    distances = sqDistances**0.5  # 开根 到这里求解了欧式距离(并构成了一个ndarray)

    # argsort() 将x中的元素从小到大排列，输出对应的index
    sortedDistances = distances.argsort()

    # 选出距离最小的k个点，并统计
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistances[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    #  operator.itemgetter(a)  项拿到器 拿到第a项 相当于 key = lambda x: x[1]
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse=True) #按照第一个(从0开始数)进行排序
    return sortedClassCount[0][0]   # 返回的出现次数最多的那个标签

# 将file转换为合适的数据结构
def file2Matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines) # 1000行
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()  # 去除首末的空格
        # 把每一行切片  ['40920', '8.326976', '0.953952', 'largeDoses']
        listFromLine = line.split()
        # 将整个行直接粘贴过来，这里自动完成了类型转换的问题, 前三列
        returnMat[index, :] = listFromLine[0:3]
        index += 1
        # 最后一列的标签放到此列表中
        classLabelVector.append(listFromLine[-1])
    return returnMat, classLabelVector

if __name__ == "__main__":
    # group, labels = file2Matrix('data/KNN_Test.txt')
    group, labels = createDataSet()
    x = [1, 1]
    print(classify0(x, group, labels, 2))
'''
sklearn 实现
 '''
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
# n_neighbors:用来指定分类器中K的大小，默认为5
# weights：设置K个点的权重 uniform(默认为平均权重)/distance(越近越高)/或自己编写的以距离为参数的权重计算函数
# algorithm: 用于计算临近点的方法 默认auto   ball_tree/kd_tree/brute


X = np.array([[0.1, 0.1], [0, 0], [1, 0.9]])
y = ['B', 'B', 'A']


neigh = KNeighborsClassifier(n_neighbors=3

# 将训练数据和标签送入分类器进行学习
neigh.fit(X,y)

# 调用predict(),对未知样本分类
print(neigh.predict([[1,0.2],[1,1]])) # 将数据构造为数组形式传入
