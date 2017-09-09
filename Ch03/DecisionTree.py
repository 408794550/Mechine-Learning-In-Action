# 决策树
# 优点：           计算复杂度不高，输出结果容易理解，对中间值的缺失不敏感，可以处理不相关特征数据
# 缺点：           可能产生过度匹配问题
# 适合数据类型：     数值型和标称型
# 决策树的两种主要算法：ID3算法和C4.5算法。后者是前者的改进。详细了解在这里 http://blog.csdn.net/acdreamers/article/details/44661149
from Utility import *
from math import log
import operator
import matplotlib.pyplot as plt


# 计算给定数据集的香农熵
def calculate_shannon_entropy(dataset):
    num_entries = len(dataset)
    label_counts = {}
    for feat_vec in dataset:
        # logging.info(feat_vec)
        current_label = feat_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


def create_dataset():
    dataset = [[1, 1, 'yes'],
               [1, 0, 'no'],
               [1, 1, 'yes'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataset, labels


my_dat, labels = create_dataset()


# logging.info(my_dat)
# logging.info(calculate_shannon_entropy(my_dat))


def split_dataset(dataset, axis, value):
    # 为了不修改到dataset而建立一个新列表
    ret_dataset = []
    for feature_vec in dataset:
        # axis表示依据某个特征来划分，所以在数据集的每个数据中找到找到这个特征既：feature_vec[axis]
        # 因为划分数据集依据是特征值是否为value所以这里做了这个判断
        if feature_vec[axis] == value:
            reduced_feature_vec = feature_vec[:axis]
            reduced_feature_vec.extend(feature_vec[axis + 1:])
            ret_dataset.append((reduced_feature_vec))
    return ret_dataset


# logging.info(split_dataset(my_dat, 0,1))
# logging.info(split_dataset(my_dat, 1,0))


def choose_best_feature_to_split(dataset):
    # 获取每条数据的特征数量
    # 这里 -1 的原因是：因为给的数据的最后一条不是特征，是目标结论,实际上我们上面的my_dat只有2个特征
    num_features = len(dataset[0]) - 1
    logging.info(num_features)
    # 计算整个数据集的原始香农熵
    base_entropy = calculate_shannon_entropy(dataset)
    best_info_gain = 0.0  # information_gain信息增益
    best_feature = -1
    for i in range(num_features):
        # 获得从当前特征到n的列表
        feature_list = [example[i] for example in dataset]
        unique_vals = set(feature_list)
        new_entropy = 0.0
        for value in unique_vals:
            # 对于每一个特征，划分一次数据集，计算新熵
            sub_dataset = split_dataset(dataset, i, value)
            prob = len(sub_dataset) / float(len(dataset))
            # 熵求和
            new_entropy += prob * calculate_shannon_entropy(sub_dataset)
        # 计算信息增益
        logging.info('new_entropy:%s', new_entropy)
        info_gain = base_entropy - new_entropy
        logging.info(info_gain)
        # 确定最佳分类特征
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


# logging.info(choose_best_feature_to_split(my_dat))
def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
        sorted_class_count = sorted(class_count.items(),
                                    key=operator.itemgetter(1),
                                    reverse=True)
    return sorted_class_count[0][0]


# 如果不太理解，可以单步调一调就理解了
def create_tree(dataset, labels):
    # 每个数据的最后一个为其分类，这里是取出了分类标签
    class_list = [example[-1] for example in dataset]
    # list.count(obj) 统计obj在list中的个数
    # 这里的判断如果成立，则说明此次递归中所有的元素都是同一种标签，就可以返回了，递归结束条件
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(dataset[0]) == 1:
        return majority_cnt(class_list)  # 挑选出现次数最多的类别作为返回值
    best_feature = choose_best_feature_to_split(dataset)
    best_feature_label = labels[best_feature]
    # 使用字典类型来存储树的信息
    my_tree = {best_feature_label: {}}
    del (labels[best_feature])
    feature_valus = [example[best_feature] for example in dataset]
    unique_vals = set(feature_valus)
    for value in unique_vals:
        # 这里对分类标签做了一次复制，因为Python在传参时传的是引用，上面的del会改变到labels列表，所以新创建一个，就影响不到原始标签labels
        sub_labels = labels[:]
        my_tree[best_feature_label][value] = create_tree(split_dataset(dataset,
                                                                       best_feature,
                                                                       value), sub_labels)
        print(best_feature_label)
        print(value)
        print(my_tree[best_feature_label][value])
    return my_tree


# logging.info(create_tree(my_dat,labels))
decision_node = dict(boxstyle="sawtooth", fc="0.8")  # 显示的注解框的形状(锯齿)
leaf_node = dict(boxstyle="round4", fc="0.8")  # 椭圆框 具体有哪些形状的框，可以看这里 https://matplotlib.org/users/annotations.html
arrow_args = dict(arrowstyle="<-")


def plot_node(node_txt, center_pt, parent_pt, node_type):
    # plt.annotate()注解:
    create_plot.ax1.annotate(node_txt,  # 要在图上显示的字串
                             xy=parent_pt,  # 箭头的坐标
                             xycoords="axes fraction",  # 坐标系选择，'axes fraction'坐标系起始位置：(0,0)为左下角
                             xytext=center_pt,  # 显示的字串的起始坐标
                             textcoords="axes fraction",  # 坐标系的选择
                             va="center",
                             ha="center",
                             bbox=node_type,  # 显示的框是什么形状
                             arrowprops=arrow_args  # 箭头相关
                             )


def create_plot():
    ''' 有关plt.figure的讲解 查看文档，或者官方文档：https://matplotlib.org/api/figure_api.html#matplotlib.figure.Figure
    Figure(figsize=None,  # 宽高
           dpi=None,
           facecolor=None,  # 背景色
           edgecolor=None,  # 边沿的颜色
           linewidth=0.0,
           frameon=None,
           subplotpars=None,
           tight_layout=None)
    '''
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    # 为函数增加了一个属性值：ax1  (注意只是函数名，没有括号，有括号就成调用了)
    # Python万物皆对象，这里给create_plot加了属性ax1,则在其他地方也可以访问到这个变量了
    create_plot.ax1 = plt.subplot(111, frameon=False)
    # total_width存储树的宽度
    # total_depth存储树的深度(高度)
    plot_node('a decision node', (0.5, 0.1), (0.1, 0.5), decision_node)
    plot_node('a leaf node', (0.8, 0.1), (0.3, 0.8), leaf_node)
    plt.show()


# create_plot()


def get_leafs_count(my_tree):
    leafs_count = 0
    # 在这一章中，因为my_tree是用字典来表示的，所以下边取了字典字第一个key
    first_str = list(my_tree.keys())[0]
    second_dict = my_tree[first_str]  # 取到这个key值下的所有节点
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':  # 这样可以判定这个节点是否还是一个字典
            leafs_count += get_leafs_count(second_dict[key])  # 还是的话，递归遍历
        else:
            leafs_count += 1  # 不是字典就是节点了
    return leafs_count


def get_tree_depth(my_tree):
    max_depth = 0
    # 在这一章中，因为my_tree是用字典来表示的，所以下边取了字典字第一个key
    first_str = list(my_tree.keys())[0]
    second_dict = my_tree[first_str]  # 取到这个key值下的所有节点
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':  # 这样可以判定这个节点是否还是一个字典
            this_depth = 1 + get_tree_depth(second_dict[key])  # 递归继续找子树高度
        else:
            this_depth = 1  # 如果key值下面的节点不对应一个字典，那么不管他就确定这个节点对应的高度只能增加1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth


def retrieve_tree(i):
    # 这里树用了json格式
    trees_list = \
        [
            {
                'no surfacing': {
                    0: 'no',
                    1: {
                        'flippers': {
                            0: 'no',
                            1: 'yes'
                        }
                    }
                }
            },
            {
                'no surfacing': {
                    0: 'no',
                    1: {
                        'flippers': {
                            0: {
                                'head': {
                                    0: 'no',
                                    1: 'yes'
                                }
                            },
                            1: 'no'
                        }
                    }
                }
            }
        ]
    return trees_list[i]


my_tree = retrieve_tree(0)
logging.info(get_leafs_count(my_tree))
logging.info(get_tree_depth(my_tree))


# 在两坐标点(cntr_pt和parent_pt中间)的中间绘制txt_str内容（这里是0，1）
def plot_mid_text(cntr_pt, parent_pt, txt_str):
    x_mid = (parent_pt[0] - cntr_pt[0]) / 2.0 + cntr_pt[0]
    y_mid = (parent_pt[1] - cntr_pt[1]) / 2.0 + cntr_pt[1]
    # logging.info('%s,%s', x_mid, y_mid)
    create_plot2.ax1.text(x_mid, y_mid, txt_str)


def plot_tree(my_tree, parent_pt, node_txt):
    leafs_count = get_leafs_count(my_tree)
    # tree_depth = get_tree_depth(my_tree) # 没用上
    first_str = list(my_tree.keys())[0]
    # (1.0 + float(leafs_count)) / 2.0 这个值确保了每次进入这个函数时，第一个key的节点绘制在当前tree(递归的子树，或者刚开始的树)的所有节点的中间位置
    # 再除以total_width是获取相对整个绘图区的位置
    cntr_pt = (plot_tree.x_off + (1.0 + float(leafs_count)) / 2.0 / plot_tree.total_width, plot_tree.y_off)
    plot_mid_text(cntr_pt, parent_pt, node_txt)

    # 绘制节点
    plot_node2(first_str,
               cntr_pt,
               parent_pt,
               decision_node)
    # 获取第一个key的节点
    second_dict = my_tree[first_str]
    # 此时要访问的是这个key的子节点，所以高度-1，对应y_off偏移要减少
    plot_tree.y_off = plot_tree.y_off - 1.0 / plot_tree.total_depth
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':  # 子节点是字典，递归
            plot_tree(second_dict[key], cntr_pt, str(key))
        else:  # 子节点是叶节点
            # 每次找到一个叶节点，则增加1个单位的x_off偏移，1单位为 1/plot_tree.total_depth
            plot_tree.x_off = plot_tree.x_off + 1.0 / plot_tree.total_width
            plot_node2(second_dict[key],
                      (plot_tree.x_off, plot_tree.y_off),  # 箭头位置指向了这个节点
                      cntr_pt,  # cntr_pt此时是父节点的位置
                      leaf_node)  # 绘制的节点类型
            # 在父节点和子节点的中间位置绘制str
            plot_mid_text((plot_tree.x_off, plot_tree.y_off), cntr_pt, str(key))
    # 当子节点绘制完毕时，y_off的偏移位置要返回到父节点的位置
    plot_tree.y_off = plot_tree.y_off + 1.0 / plot_tree.total_depth


def plot_node2(node_txt, center_pt, parent_pt, node_type):
    # plt.annotate()注解:
    create_plot2.ax1.annotate(node_txt,  # 要在图上显示的字串
                              xy=parent_pt,  # 箭头的坐标
                              xycoords="axes fraction",  # 坐标系选择，'axes fraction'坐标系起始位置：(0,0)为左下角
                              xytext=center_pt,  # 显示的字串的起始坐标
                              textcoords="axes fraction",  # 坐标系的选择
                              va="center",
                              ha="center",
                              bbox=node_type,  # 显示的框是什么形状
                              arrowprops=arrow_args  # 箭头相关
                              )


def create_plot2(in_tree):
    ''' 有关plt.figure的讲解 查看文档，或者官方文档：https://matplotlib.org/api/figure_api.html#matplotlib.figure.Figure
    Figure(figsize=None,  # 宽高
           dpi=None,
           facecolor=None,  # 背景色
           edgecolor=None,  # 边沿的颜色
           linewidth=0.0,
           frameon=None,
           subplotpars=None,
           tight_layout=None)
    '''
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    # 为函数增加了一个属性值：ax1  (注意只是函数名，没有括号，有括号就成调用了)
    create_plot2.ax1 = plt.subplot(111, frameon=False, **axprops)
    # total_width存储树的宽度
    plot_tree.total_width = float(get_leafs_count(in_tree))
    # total_depth存储树的深度(高度)
    plot_tree.total_depth = float(get_tree_depth(in_tree))
    # 还没有绘制前x_off的起始位置，在坐标系左侧。因为此时还没有节点要绘制，没绘制一个节点，增加1/plot_tree_width
    plot_tree.x_off = -0.5 / plot_tree.total_width
    # y_off因为树是从上往下绘制的，所以刚开始在最高的位置1
    plot_tree.y_off = 1.0
    # 第三个参数是空字符串，因为第一次绘制时，不需要绘制0或者1
    plot_tree(in_tree, (0.5, 1.0), '')
    # plot_node('a decision node', (0.5, 0.1), (0.1, 0.5), decision_node)
    # plot_node('a leaf node', (0.8, 0.1), (0.3, 0.8), leaf_node)
    plt.show()


logging.info(my_tree)
create_plot2(my_tree)


def classify(input, feature_labels, test_vec):
    pass

