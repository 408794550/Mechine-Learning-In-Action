# 朴素贝叶斯算法
# 优点:           在数据较少的情况下依然有效，可以处理多类别问题
# 缺点:           对于输入数据的准备方式较为敏感
# 适用数据类型:     标称型数据
# 贝叶斯决策论的核心思想：选择具有最高概率的决策

# encoding: utf-8
from Utility import logging
from numpy import *


def load_dataset():
    posting_list = [['my',   'dog',       'has',       'flea',      'problems', 'help', 'please'],
                   ['maybe', 'not',       'take',      'him',       'to',       'dog',  'park', 'stupid'],
                   ['my',    'dalmation', 'is',        'so',        'cute',     'I',    'love', 'him'],
                   ['stop',  'posting',   'stupid',    'worthless', 'garbage'],
                   ['mr',    'licks',     'ate',       'my',        'steak',    'how',  'to',   'stop', 'him'],
                   ['quit',  'buying',    'worthless', 'dog',       'food',     'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]
    return posting_list, class_vec


# 创建在所有文档中出现的不重复的列表
def create_vocabulary_list(dataset):
    vocabulary_set = set([])
    for document in dataset:
        vocabulary_set = vocabulary_set | set(document)  # "|"求两个集合的并集，也是按位或操作，这里用的都是同一个
    return list(vocabulary_set)


# 输入input_set，输出向量化的input_set,通过比对vocabulary，如果input_set中的相应位置在vocabulay中出现，则相应位置置1
def set_word_to_vec(vocabulary_list, input_set):
    # 创建一个向量，长度与vocabulary_list等长，值为0，表示单词未出现
    return_vec = [0]*len(vocabulary_list)
    for word in input_set:
        if word in vocabulary_list:
            # 如果单词出现在input_set中，则相应位置置1
            return_vec[vocabulary_list.index(word)] = 1
        else:
            log.info('the word:%s is not in my vocabulary!', word)
    return return_vec


posts_list, classes_list = load_dataset()
my_vocabulary_list = create_vocabulary_list(posts_list)
logging.info(my_vocabulary_list)


def train_naive_bayes(train_matrix, train_category):
    train_docs_count = len(train_matrix)
    words_count = len(train_matrix[0])
    p_abusive = sum(train_category)/float(train_docs_count)
    # 刚开始的
    p0_count_vec = ones(words_count)
    p1_count_vec = ones(words_count)
    p0_denom = 2.0
    p1_denom = 2.0
    for i in range(train_docs_count):
        if train_category[i] == 1:  # 1代表侮辱性文字
            # 两向量的加法
            p1_count_vec += train_matrix[i]
            p1_denom += sum(train_matrix[i])
            logging.info(p1_denom)
        else:
            p0_count_vec += train_matrix[i]
            p0_denom += sum(train_matrix[i])
    # 取对数的原因：有时候计算因子太小，程序会下溢（很多小数相乘，最后四舍五入得0），多以取自然对数
    p1_vec = log(p1_count_vec/p1_denom)
    p0_vec = log(p0_count_vec/p0_denom)
    return p0_vec, p1_vec, p_abusive


train_mat = []
for post_in_doc in posts_list:
    # 在这里把测试文本数值化,既表示某个词是否在文档中出现了
    train_mat.append(set_word_to_vec(my_vocabulary_list, post_in_doc))
p0_vec, p1_vec, p_abusive = train_naive_bayes(train_mat, classes_list)
logging.info(p_abusive)


def classify_naive_bayes(vec2_classify, p0_vec, p1_vec, p_class1):
    # 向量中的元素先各自乘 * p1-vec,然后再相加，最后比较p1与p0
    p1 = sum(vec2_classify * p1_vec) + log(p_class1)
    p0 = sum(vec2_classify * p0_vec) + log(1.0 - p_class1)
    return 1 if p1 > p0 else 0


def testing_nb():
    test_entry = ['love', 'my', 'dalmation']
    this_doc = array(set_word_to_vec(my_vocabulary_list, test_entry))
    logging.info('{},classified as: {}'.format(test_entry, classify_naive_bayes(this_doc, p0_vec, p1_vec, p_abusive)))
    test_entry = ['stupid', 'garbage']
    this_doc = array(set_word_to_vec(my_vocabulary_list, test_entry))
    logging.info('{},classified as: {}'.format(test_entry, classify_naive_bayes(this_doc, p0_vec, p1_vec, p_abusive)))


testing_nb()


# 词袋模型
def bag_words_to_vec(vocabulary_list, input_set):
    return_vec = [0] * len(vocabulary_list)
    for word in input_set:
        if word in vocabulary_list:
            # 如果单词出现在input_set中，则相应位置置1
            return_vec[vocabulary_list.index(word)] += 1  # 上面与这个方法类似的，只是置1，没有记录词出现的次数
    return return_vec


# 使用朴素贝叶斯过滤垃圾邮件
def text_parse(big_str):
    import re
    # 字符串前加 r ,字符串内部的 \ 就不会被特殊处理
    # \w匹配数字字符或者下划线
    # re.split()  #第一个参数表示要匹配的模式（正则表达式）,第二个是要匹配的对象
    tokens_list = re.split(r'\w', big_str)
    return [tok.lower() for tok in tokens_list if len(tok) > 2]


def spam_test():
    doc_list = []
    classes_list = []
    full_text = []
    for i in range(1,26):
        # open()打开失败怎么办？解决：使用with open
        with open('email/spam/%d.txt' % i) as f:
            word_list = text_parse(f.read())
        doc_list.append(word_list)
        full_text.append(word_list)
        classes_list.append(1)
        with open('email/ham/%d.txt' % i) as f:
            word_list = text_parse(f.read())
        doc_list.append(word_list)
        full_text.append(word_list)
        classes_list.append(1)
    vocabulary_list = create_vocabulary_list(doc_list)
    training_set = range(50)
    test_set = []
    # 随机选10个作为测试样本
    for i in range(10):
        rand_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del (training_set[rand_index])
    train_mat = []
    train_classes = []
    # 训练

    for doc_index in training_set:
        train_mat.append(bag_words_to_vec(vocabulary_list, doc_list[doc_index]))
        train_classes.append(class_list[doc_index])
    p0_vec, p1_vec, p_spam = train_naive_bayes(array(train_mat), array(train_classes))
    error_count = 0
    # 测试
    for doc_index in test_set:
        word_vector = bag_words_to_vec(vocabulary_list, doc_listist[doc_index])
        if classify_naive_bayes(array(word_vector), p0_vec, p1_vec, p_spam) != classList[doc_index]:
            error_count += 1
            logging.info("classification error{}".format(docList[docIndex]))

    logging.info('the error rate is:{} '.format(float(errorCount) / len(testSet)))


spam_test()


# 使用朴素贝叶斯分类器从个人广告中获取区域倾向



