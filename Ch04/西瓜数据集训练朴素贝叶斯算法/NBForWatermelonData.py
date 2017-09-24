# 使用西瓜数据机训练朴素贝叶斯算法
'''
朴素贝叶斯算法
优点:           在数据较少的情况下依然有效，可以处理多类别问题
缺点:           对于输入数据的准备方式较为敏感
适用数据类型:     标称型数据
贝叶斯决策论的核心思想：选择具有最高概率的决策
'''
from numpy import *
from Utility import *
from math import sqrt


def load_dataset(file_name):
    training_set = []
    labels_set = []
    with open(file_name) as f_train:
        for line in f_train:
            current_line = line.strip().split(',')
            line_arr = []
            for i in range(1, 7):
                line_arr.append(current_line[i])
            for i in range(7, 10):
                line_arr.append(float(current_line[i]))
            training_set.append(line_arr)
            labels_set.append(float(current_line[9]))
    return training_set, labels_set


# 这两个变量用来存储训练集中各个特征的概率，避免2次计算
p_pos_feature_dic = {}
p_neg_feature_dic = {}
training_set, labels_set = load_dataset('waterMelonTraining.txt')


# 前6个特征的概率计算
def p_special_feature(dataset, classfy_labels, special_feature, i):
    # 检查字典中是否已经含有这个特征的概率，有就不用计算了
    if special_feature in p_pos_feature_dic.keys() | p_pos_feature_dic.keys():
        # logging.info('special feature is already save')
        return p_pos_feature_dic[special_feature], p_neg_feature_dic[special_feature]

    for j in range(2):
        # 拉普拉斯修正
        feature_count = 1
        for watermelon in dataset:
            if watermelon[-1] == j:
                if special_feature == watermelon[i]:
                    feature_count += 1
        length = len(set([x[i] for x in dataset]))
        p = feature_count/(sum(classfy_labels)+length) if j == 1 else feature_count/(len(dataset)-sum(classfy_labels)+length)
        dic = p_pos_feature_dic if j == 1 else p_neg_feature_dic
        dic[special_feature] = p
    return [p_pos_feature_dic[special_feature], p_neg_feature_dic[special_feature]]


# 第7个特征的均值与标准差
getx = lambda dataset, y, index: [x[index] for x in dataset if x[-1] == y]
pos_mean_value7 = mean(getx(training_set, 1.0, 6))
pos_std_value7 = std(getx(training_set, 1.0, 6))
neg_mean_value7 = mean(getx(training_set, 0.0, 6))
neg_std_value7 = std(getx(training_set, 0.0, 6))
# 第8个特征的均值与标准差
pos_mean_value8 = mean(getx(training_set, 1.0, 7))
pos_std_value8 = std(getx(training_set, 1.0, 7))
neg_mean_value8 = mean(getx(training_set, 0.0, 7))
neg_std_value8 = std(getx(training_set, 0.0, 7))


# 7 8特征概率的计算
def p_continuity_feature(dataset, classfy_labels, value7, value8, pos=True):
    p7_pos = 0.0; p8_pos = 0.0; p7_neg = 0.0; p8_neg = 0.0
    if pos:
        p7_pos = (1 / (sqrt(2 * pi) * pos_std_value7)) * exp(-((value7 - pos_mean_value7) ** 2 / (2 * (pos_std_value7 ** 2))))
        p8_pos = (1 / (sqrt(2 * pi) * pos_std_value8)) * exp(-((value8 - pos_mean_value8) ** 2 / (2 * (pos_std_value8 ** 2))))
    else:
        p7_neg = (1 / (sqrt(2 * pi) * neg_std_value7)) * exp(-((value7 - neg_mean_value7) ** 2 / (2 * (neg_std_value7 ** 2))))
        p8_neg = (1 / (sqrt(2 * pi) * neg_std_value8)) * exp(-((value8 - neg_mean_value8) ** 2 / (2 * (neg_std_value8 ** 2))))
    return (p7_pos, p8_pos) if pos else (p7_neg, p8_neg)


def classify_watermelon(watermelon):
    p_pos = 1.0; p_neg = 1.0
    for i in range(6):
        p_pos *= p_special_feature(training_set, labels_set, watermelon[i], i)[0]
        p_neg *= p_special_feature(training_set, labels_set, watermelon[i], i)[1]
    p_pos *= (lambda x: x[0]*x[1])(p_continuity_feature(training_set, labels_set, watermelon[6], watermelon[7]))
    p_neg *= (lambda x: x[0]*x[1])(p_continuity_feature(training_set, labels_set, watermelon[6], watermelon[7], pos=False))
    return 1.0 if p_pos > p_neg else 0.0


def train_classify():
    error = 0
    for watermelon in training_set:
        if classify_watermelon(watermelon) != watermelon[-1]:
            error += 1
            logging.info('{}'.format(watermelon))

    logging.info(error)
    logging.info('the error rate is {}'.format(error/len(training_set)))


# 训练数据集
train_classify()








'''
# 自助法，从训练集中分出测试集
def self_help():
    self_help_list = []
    for i in range(20):
        self_help_list.append(training_set[int(random.uniform(0, 16))])
    return self_help_list


test_set = self_help()
# 自助发导出的西瓜数据集，打印出来后存储的。如果使用上面的方法导出的，每次都在变
test_set2 = [
    ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, 0.264, 1.0],
    ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, 0.318, 1.0],
    ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0.245, 0.057, 0.0],
    ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, 0.091, 0.0],
    ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, 0.264, 1.0],
    ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.343, 0.099, 0.0],
    ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267, 0.0],
    ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.46, 1.0],
    ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 0.437, 0.211, 1.0],
    ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.46, 1.0],
    ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, 0.161, 0.0],
    ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.403, 0.237, 1.0],
    ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, 0.091, 0.0],
    ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, 0.149, 1.0],
    ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, 0.198, 0.0],
    ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267, 0.0],
    ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 0.437, 0.211, 1.0],
    ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267, 0.0],
    ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, 0.198, 0.0],
    ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.46, 1.0]]


# 测试数据集
def test_classify():
    error = 0
    for watermelon in test_set2:
        if classify_watermelon(watermelon) != watermelon[-1]:
            error += 1
            logging.info('test{}'.format(watermelon))
    logging.info(error)
    logging.info('the error rate is {}'.format(error/len(training_set)))


test_classify()
'''