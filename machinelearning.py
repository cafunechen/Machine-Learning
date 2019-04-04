# -*- coding:utf-8 -*-
import numpy as np
from sklearn import svm
import sklearn.model_selection
import matplotlib as mpl
import matplotlib.pyplot as plt
def iris_type(s):
    it = {b'Iris-setosa' :0,b'Iris-versicolor' :1,b'Iris-virginica': 2}
    return it[s]

path = u'E:\learning\code\machinelearning\lris.txt'
data = np.loadtxt(path,dtype = float,delimiter = ',',converters = {4:iris_type})
# print(data)
#将数据集分为训练集与测试集
x,y = np.split(data,(4,),axis = 1)
# split(数据，分割位置，轴=0（水平分割） or 1（垂直分割）)
x = x[:, :2]
# x = x[:, :2]是为方便后期画图更直观，故只取了前两列特征值向量训练
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, random_state=1, train_size=0.6,test_size=0.4)
'''
    sklearn.model_selection.train_test_split随机划分训练集与测试集。
    train_test_split(train_data,train_target,test_size=数字, random_state=0)
　　参数解释：
　　train_data：所要划分的样本特征集
　　train_target：所要划分的样本结果
　　test_size：样本占比，如果是整数的话就是样本的数量
　　random_state：是随机数的种子。
　　随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，
    保证得到一组一样的随机数。比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。
    但填0或不填，每次都会不一样。随机数的产生取决于种子，
    随机数和种子之间的关系遵从以下两个规则：种子不同，产生不同的随机数；
    种子相同，即使实例不同也产生相同的随机数。
'''
clf = svm.SVC(C=0.8,kernel = 'rbf',gamma = 20, decision_function_shape='ovr')
clf.fit(x_train,y_train.ravel())
'''
kernel='linear'时，为线性核，C越大分类效果越好，但有可能会过拟合（defaul C=1）。
kernel='rbf'时（default），为高斯核，gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”，分类效果越好，但有可能会过拟合。
decision_function_shape='ovr'时，为one v rest，即一个类别与其他类别进行划分，
decision_function_shape='ovo'时，为one v one，即将类别两两之间进行划分，用二分类的方法模拟多分类的结果。
'''
'''
将多维数组降为一维
numpy.flatten()返回一份拷贝，对拷贝所做的修改不会影响（reflects）原始矩阵
numpy.ravel()返回的是视图，会影响（reflects）原始矩阵。
'''
print(clf.score(x_train, y_train))  # 精度


print(clf.score(x_test, y_test))

# 决策函数
# print('decision_function:\n', clf.decision_function(x_train))
# print( '\npredict:\n', clf.predict(x_test))
#绘制图像
x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围
x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围
x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网格采样点
grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
# stack（增加一维）[]--[[]]

# font
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

# print 'grid_test = \n', grid_test
grid_hat = clf.predict(grid_test)       # 预测分类值
grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同

alpha = 0.5
plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)     # 预测值的显示
plt.plot(x[:, 0], x[:, 1], 'o', alpha=alpha, color='blue', markeredgecolor='k')
plt.scatter(x_test[:, 0], x_test[:, 1], s=120, facecolors='none', zorder=10)  # 圈中测试集样本
plt.xlabel(u'花萼长度', fontsize=13)
plt.ylabel(u'花萼宽度', fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u'鸢尾花SVM二特征分类', fontsize=15)
# plt.grid()
plt.show()
Machine-Learing
