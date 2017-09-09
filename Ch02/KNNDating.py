# K-近邻算法示例:改进约会网站的配对效果
# 重要知识点：归一化数值
from Utility import *
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import KNearestNeighbor


def file2_matrix(file_name):
    # f = open(file_name)
    f = open(file_name)
    array_lines = f.readlines()
    file_lines = len(array_lines)
    logging.info(file_lines)
    # zeros()函数：创建指定类型的矩阵，并初始化为0
    return_mat = zeros((file_lines,3))
    class_label_vec = []
    index = 0
    for line in array_lines:
        # strip(char)函数从头尾移除指定字符，并返回
        line = line.strip()  # 去掉回车
        list_from_line = line.split('\t')  # 分割得到列表
        return_mat[index,:] = list_from_line[0:3]
        # -1可以取到最后一个元素,如果不明确指示这是int，解释器将当做字符串处理
        class_label_vec.append(int(list_from_line[-1]))
        index += 1
    return return_mat,class_label_vec


mat, dating_labels = file2_matrix('datingTestSet2.txt')
logging.info(mat[:15])
# 创建画布
fig = plt.figure()
# add_subplot(111)，参数111的解释：
# 111表示将画布分割成1行1列，在分割出来的区域的第1块作图，所以349,表示3行3列(有12个区域了),在第9个区域作图
ax = fig.add_subplot(111)
# scatter(x,y,s,c,marker,cmap,norm,vmin,vmax,alpha,linewidths,verts,hold,**kwargs)函数解析：
# x,y 对应输入的数据；  s 对应散点的大小；  c 对应散点的颜色; marker 对应散点的形状
# http://blog.csdn.net/u013634684/article/details/49646311 后面的参数详细见链接
ax.scatter(mat[:, 1], mat[:, 2], 15.0*array(dating_labels), 15.0*array(dating_labels))
plt.show()


def auto_norm(data_set):
    # 获取每一 列 的最小值(如果参数是1，则是获取 行 的最小值)
    min_vals = data_set.min(0)
    # 最大值
    max_vals = data_set.max(0)
    ranges = max_vals - min_vals
    # 创建一个数值全为0的形状为shape(data_set)的矩阵
    norm_data_set = zeros(shape(data_set))
    # shape[0]获得矩阵的行数，1获得列数
    m = data_set.shape[0]
    # 此时min_vals是一行，tile函数扩展到m行，和data_set形状一致
    # 归一化公式：第一步，原来的值减去最小值
    norm_data_set = data_set - tile(min_vals,(m,1))
    # 第二步：除以取值范围(最大值与最小值之间的范围),这里ranges同min_vals是一行
    # 下面的代码虽然是两个矩阵相除，但并不是矩阵的除法，矩阵的除法在numpy中另有方法
    norm_data_set = norm_data_set / tile(ranges, (m,1))
    return norm_data_set, ranges, min_vals


def dating_class_test():
    ho_rotio = 0.10
    dating_data_mat, dating_label = file2_matrix('datingTestSet2.txt')
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    m = norm_mat.shape[0]
    # 用来测试的行数
    num_test_vecs = int(m * ho_rotio)
    error_count = 0
    # 对每一行进行测试(这里有100行)
    for i in range(num_test_vecs):
        # 将每行norm_mat[i,:]进行分类，在使用这些数据的时候，我们已经知道了他是属于哪一类，所以可以计算这个算法的分类成功率
        classifier_result = KNearestNeighbor.classify0(norm_mat[i, :],
                                                          norm_mat[num_test_vecs:m, :],
                                                          dating_labels[num_test_vecs:m],
                                                          3)
        logging.info('came back with:%d,the real answer is: %d', classifier_result, dating_labels[i])
        if classifier_result != dating_labels[i]:
            error_count += 1.0
    logging.info('total error rate is：%f', (error_count/float(num_test_vecs)))


dating_class_test()


def classify_person():
    result_list = ['not at all', 'in small doses', 'in large doses']
    # 3.0之后，raw_input()与input()等效，不再有raw_input()
    percent_tats = float(input('percentage of time spent playing video games'))
    ff_miles = float(input('frequent flier miles earned per year?'))
    ice_cream = float(input('liters of ice cream consumed per year?'))
    dating_data_mat, dating_labels = file2_matrix('datingTestSet2.txt')
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    in_arr = array([ff_miles, percent_tats, ice_cream])
    classifier_result = MechineLearningDemo.classify0((in_arr - min_vals)/ranges,
                                                      norm_mat,
                                                      dating_labels,
                                                      3)
    # 下面两个打印的差异，logging如果不加%s打印会出错，而print貌似会自动处理在最后没有%s这个过程
    logging.info('you will probably like this person:%s',result_list[classifier_result - 1])
    # print('you will probably like this person:',result_list[classifier_result - 1])


# classify_person()
