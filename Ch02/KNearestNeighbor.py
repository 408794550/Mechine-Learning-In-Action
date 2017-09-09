# K-邻近算法
# 优点:           精度高、对异常值不敏感、无数据输入假定
# 缺点:           计算复杂度高、空间复杂度高
# 适用数据范围:    数值型和标称型

from numpy import *
import operator
import sys
from Utility import *


def create_dataset():
    group = array([[1.0, 1.1],
                   [1.0, 1.0],
                   [0, 0],
                   [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


group, labels = create_dataset()


# K-近邻算法
def classify0(inX, data_set, labels, k):
    # shape属性输出矩阵的形状，shape[0],则输出行，shape[1]输出列
    data_set_size = data_set.shape[0]
    # tile(A,reps) 功能：X轴上扩展A,reps是重复次数(1不变)如果reps是元组，则第一个元素控制行数，第二个控制重复次数
    # 如(1,2),表A在行数上不变，列重复复制了1次
    # A可以是array,list,tuple,dict,matrix,int,string,float,bool等
    # reps可以是list,dict,array,int,bool
    # 以下3行是欧式距离公式
    diff_mat = tile(inX, (data_set_size,1)) - data_set
    sq_diff_mat = diff_mat ** 2  # **是乘方
    # .sum(axis = 0) or .sum(dxis = 1) 对一个矩阵进行相加,1表示按行相加，0表示按列相加
    sq_distance = sq_diff_mat.sum(axis=1)
    # 欧式距离公式计算结束
    distances = sq_distance ** 0.5 # 取根号，得到距离
    # argsort()函数是将array中的元素从小到大排列，输出其index
    sorted_indexs = distances.argsort()
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_indexs[i]]
        # dict.get(key,default=None)，此方法在dict中查找key，如果没有找到，则返回default
        class_count[vote_label] = class_count.get(vote_label,0) + 1
    #  items()和iteritems()
    #  items()将字典中的所有项，以列表方式返回
    #  iteritems()将字典中的所有项，返回成一个迭代器(Py3.5版本后，变成了items)
    # operator.itemgetter(i)：定义一个函数，获取第i域的值
    # sorted()：这里的sorted将会从迭代器中的列表中以第一个元素来进行排序，reverse代表了倒序
    sorted_class_count = sorted(
        class_count.items(),
        key=operator.itemgetter(1),
        reverse=True
    )
    return sorted_class_count[0][0]


logging.info(classify0([1, 1], group, labels, 3))




