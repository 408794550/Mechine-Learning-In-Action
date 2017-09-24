from Utility import logging
from numpy import *
import random


def load_dataset(file_name):
    data_mat = []
    label_mat = []
    with open(file_name) as fr:
        for line in fr.readlines():
            line_arr = line.strip().split('\t')
            data_mat.append([float(line_arr[0]), float(line_arr[1])])
            label_mat.append(float(line_arr[2]))
    return data_mat, label_mat


def selectJ_rand(i, m):
    j = i
    while(j == i):
        j = int(random.uniform(0, m))
    return j


def clip_alpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


data_arr, label_arr = load_dataset('testSet.txt')
logging.info('{}'.format(label_arr))


def smo_simple(data_mat_in,  # 数据集
               class_labels,  # 类别标签
               C,  # 常数C
               toler,  # 容错率
               max_iter):  # 取消前最大的循环次数
    data_matrix = mat(data_mat_in)  # 列表转成了矩阵
    label_mat = mat(class_labels).transpose()  # 转置得到了一个列向量,于是数据矩阵的每一行都和类别标签向量意义对应
    b = 0
    m, n = shape(data_matrix)  # 100行2列
    alphas = mat(zeros((m, 1)))  # 构建alphas列矩阵(100行1列，都为0)
    iter_count = 0
    while iter_count < max_iter:
        alpha_pairs_changed = 0  # 变量用于记录alpha是否已经进行优化，每次循环结束时得知
        for i in range(m):
            # multiply实现对应元素相乘，同使用 * 号
            # .T 效果同使用.transpose(),只是在，只有一行的情况下，返回的是本身
            f_Xi = float(multiply(alphas, label_mat).T * (data_matrix * data_matrix[i, :].T)) + b
            Ei = f_Xi - float(label_mat[i])  # 计算误差Ei
            if ((label_mat[i] * Ei < -toler) and (alphas[i] < C)) or ((label_mat[i] * Ei) > toler) and (alphas[i] > 0):
                j = selectJ_rand(i, m)  # 随机选择第二个alpha值
                f_Xj = float(multiply(alphas, label_mat).T * (data_matrix * data_matrix[j, :].T)) + b
                Ej = f_Xj - float(label_mat[j])  # 同样计算第二个alpha的误差
                alphaI_old = alphas[i].copy()
                alphaJ_old = alphas[j].copy()
                # L H用来将alpha[j]调整到0与C之间
                if label_mat[i] != label_mat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    logging.info('L == H')
                    continue
                # eta 为最优修改量
                eta = 2.0 * data_matrix[i, :] * data_matrix[j, :].T - data_matrix[i, :] * data_matrix[i, :].T - data_matrix[j, :] *data_matrix[j, :].T
                if eta >= 0:
                    logging.info('eta >= 0')
                    continue
                alphas[j] -= label_mat[j] * (Ei - Ej) / eta
                alphas[j] = clip_alpha(alphas[j], H, L)
                if abs(alphas[j] - alphaJ_old) < 0.00001 :
                    logging.info('j not moving enough')
                    continue
                alphas[i] += label_mat[j] * label_mat[i] * (alphaJ_old - alphas[j])
                b1 = b - Ei - label_mat[i] * (alphas[i] - alphaI_old) * data_matrix[i, :] * data_matrix[i, :].T - label_mat[j] * (alphas[j] - alphaJ_old) * data_matrix[i, :] * data_matrix[j, :].T
                b2 = b - Ej - label_mat[i] * (alphas[i] - alphaI_old) * data_matrix[i, :] * data_matrix[j, :].T - label_mat[j] * (alphas[j] - alphaJ_old) * data_matrix[j, :] * data_matrix[j, :].T
                if 0 < alphas[j] and C > alphas[i]:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alpha_pairs_changed += 1
                logging.info('iter:{} i:{},pairs changed {}'.format(iter_count, i, alpha_pairs_changed))
            if alpha_pairs_changed == 0:
                iter_count += 1
            else:
                iter_count = 0
            logging.info('iteration number: {}'.format(iter_count))
    return b, alphas


b, alphas = smo_simple(data_arr, label_arr, 0.6, 0.001, 40)
logging.info('{}'.format(b))
logging.info('{}'.format(alphas[alphas > 0]))


'''  这里没理解SMO算法的真谛，选择了先跳过完整的SMO算法。
class opt_struct:
    def __init__(self, datamat_in, class_labels, C, toler):
        self.X = datamat_in
        self.label_mat = class_labels
        self.C = C
        self.tol = toler
        self.m = shape(datamat_in)[0]
        self.b = 0
        self.alphas = mat(zeros(self.m, 1))
        self.e_cache = mat(zeros(self.m, 2))


    def calculate_Ek(opt_stru, k):
        f_Xk = float(multiply(opt_stru.alphas, opt_stru.label_mat).T * (opt_stru.X * opt_stru.X[k, :].T)) + opt_stru.b
        Ek = f_Xk - float(opt_stru.label_mat[k])
        return Ek


    def select_j(i, opt_stru, Ei):
        max_k = -1
        max_delta_E = 0
        Ej = 0
        opt_stru.e_cache[i] = [1, Ei]
        validE_ecache_list = nonzero(opt_stru.e_cache[:, 0].A)[0]
        if len(validE_ecache_list) > 1:
            for k in validE_ecache_list:
                if k == i:
                    continue
                Ek = calculate_Ek(opt_stru, k)
                delta_E = abs(Ei - Ek)
                if delta_E > max_delta_E:
                    max_k = k
                    max_delta_E = delta_E
                    Ej = Ek
            return  max_k, Ej
        else:
            j = selectJ_rand(i, opt_stru.m)
            Ej = calculate_Ek(opt_stru, j)
        return j, Ej


    def update_Ek(opt_stru, k):
        Ek = calculate_Ek(opt_stru, k)
        opt_stru.e_cache[k] = [1, Ek]
    '''


