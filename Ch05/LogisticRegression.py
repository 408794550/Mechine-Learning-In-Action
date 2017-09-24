# 梯度上升法（与梯度下降法一样，只是下降法是减法，而梯度上升法是加法），梯度上升法用与求函数的最大值，而梯度下降法用于求函数的最小值
# 梯度上升法的思想：要找到某函数的最大值，最好的方法是沿着该函数的梯度方向探寻。
from numpy import *
from Utility import *
import matplotlib.pyplot as plt


def load_dataset():
    data_mat = []
    label_mat = []
    with open('testSet.txt') as f:
        for line in f.readlines():
            # strip()函数移除字符串头尾指定的字符，默认为空格，split()函数也一样
            line_arr = line.strip().split()
            data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])
            label_mat.append(int(line_arr[2]))
    return data_mat, label_mat


def sigmoid(in_x):
    return 1.0 / (1 + exp(-in_x))


# data_mat_in是100 * 3的数据集矩阵，class——labels是1 * 100的行向量
def grad_ascent(data_mat_in, class_labels):
    #
    data_matrix = mat(data_mat_in)
    # transpose()函数，实现矩阵转置
    label_mat = mat(class_labels).transpose()
    m, n = shape(data_matrix)
    alpha = 0.001  # 移动步长
    max_cycles = 500  # 迭代次数
    weights = ones((n, 1))  # 1列
    for k in range(max_cycles):
        # 这里 h 是个列向量
        h = sigmoid(data_matrix * weights)
        # 相应的，下面进行了列向量中的数值运算，不是一个数
        error = (label_mat - h)
        weights = weights + alpha * data_matrix.transpose() * error
    return weights


data_arr, label_mat = load_dataset()
# logging.info(grad_ascent(data_arr, label_mat))
weights = grad_ascent(data_arr, label_mat)


def plot_best_fit(weight):
    # weight传进来的时候是个矩阵，转换一下
    weights = array(weight)
    data_mat, label_mat = load_dataset()
    data_arr = array(data_mat)
    logging.info('data_arr:{}'.format(data_arr))
    n = shape(data_arr)[0]
    logging.info('n length is :{}'.format(n))
    x_cord1 = []
    y_cord1 = []
    x_cord2 = []
    y_cord2 = []
    for i in range(n):
        if int(label_mat[i] == 1):
            x_cord1.append(data_arr[i, 1])
            y_cord1.append(data_arr[i, 2])
        else:
            x_cord2.append(data_arr[i, 1])
            y_cord2.append(data_arr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 描绘散点图
    ax.scatter(x_cord1, y_cord1, s=30, c='red', marker='s')
    ax.scatter(x_cord2, y_cord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    logging.info(x)
    logging.info(y)
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


# plot_best_fit(weights)


# 训练算法：随机梯度上升
def stoc_grad_ascent0(data_mat, class_labels):
    m, n = shape(data_mat)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(data_mat[i] * weights)
        error = class_labels[i] - h
        weights = weights + alpha * error * data_mat[i]
    return weights


# weights = stoc_grad_ascent(array(data_arr), label_mat)
# plot_best_fit(weights)


# 改进的随机梯度上升算法
def stoc_grad_ascent1(data_mat, class_labels, num_iter=150):
    m, n = shape(data_mat)
    weights = ones(n)
    for j in range(num_iter):
        data_index = range(m)
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            rand_index = int(random.uniform(0, len(data_index)))
            h = sigmoid(sum(data_mat[rand_index] * weights))
            error = class_labels[rand_index] - h
            weights = weights + alpha * error * data_mat[rand_index]
            del (list(data_index)[rand_index])
    return weights


# weights = stoc_grad_ascent1(array(data_arr), label_mat)
# plot_best_fit(weights)


# 从疝气病症预测病马的死亡率
def classify_vector(in_x, weights):
    prob = sigmoid(sum(in_x * weights))
    return 1.0 if prob > 0.5 else 0.0


def colic_test():
    training_set = []
    training_labels = []
    with open('horseColicTraining.txt') as f_train:
        for line in f_train.readlines():
            current_line = line.strip().split('\t')
            line_arr = []
            for i in range(21):
                line_arr.append(float(current_line[i]))
            training_set.append(line_arr)
            training_labels.append(float(current_line[21]))
    train_weights = stoc_grad_ascent1(array(training_set), training_labels, num_iter=500)
    # train_weights = stoc_grad_ascent1(array(training_set), training_labels)
    error_count = 0
    test_count_vec = 0.0
    with open('horseColicTest.txt') as f_test:
        for line in f_test.readlines():
            test_count_vec += 1.0
            current_line = line.strip().split('\t')
            line_arr = []
            for i in range(21):
                line_arr.append(float(current_line[i]))
            if int(classify_vector(array(line_arr), train_weights)) != int(current_line[21]):
                error_count += 1
        error_rate = (float(error_count / test_count_vec))
        logging.info('error rate of this test is :{}'.format(error_rate))
    return error_rate


def multi_test():
    tests_count = 10
    error_count = 0.0
    for k in range(tests_count):
        error_count += colic_test()
    logging.info(
        'after {} iterations the average error rate is:{}'.format(tests_count, error_count / float(tests_count)))


# 调整colic_test中的迭代次数，和 stoch_grad_ascent1()中的步长，平均错误率可以降低
multi_test()
