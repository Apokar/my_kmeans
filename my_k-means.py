# -*- coding: utf-8 -*-
# @Time    : 2020/8/12 10:54
# @Author  : Apokar
# @Email   : Apokar@163.com
# @File    : my_k-means.py
# @Comment :

# 导包，初始化图形参数，导入样例数据集
from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# 定义距离计算函数
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


def do_k_means():
    plt.rcParams['figure.figsize'] = (16, 9)
    plt.style.use('ggplot')

    data = pd.read_csv('xclara.csv')
    # print(data.head())

    # 将数据集转换为二维数组，并绘制二维坐标图
    f1 = data["V1"].values
    f2 = data["V2"].values

    X = np.array(list(zip(f1, f2)))
    # print(X[:2])
    plt.scatter(f1, f2, c="black", s=6)

    # 根据二维坐标图，设定分区数为3
    k = 3
    # 随机生成中心点的x坐标
    C_x = np.random.randint(0, np.max(X) - 20, size=k)
    # 随机生成中心点的y坐标
    C_y = np.random.randint(0, np.max(X) - 20, size=k)
    C = np.array(list(zip(C_x, C_y)), dtype=np.float32)

    # 将初始化中心点和样例数据画到同一个坐标系上
    plt.scatter(f1, f2, c='black', s=6)

    plt.scatter(C_x, C_y, marker="*", c='red', s=200)
    plt.show()

    # # K-means的核心迭代

    # 用于保存中心点更新前的坐标
    C_old = np.zeros(C.shape)
    # print(C)
    # print(C_old)

    # 用于保存数据所属的中心点
    clusters = np.zeros(len(X))

    # 迭代标识位，通过计算新旧中心点的距离
    iteration_flag = dist(C, C_old, 1)
    #
    tmp=1
    # 若中心点不再变化或循环不超过20次，则退出循环
    while iteration_flag.any() != 0 and tmp<20:
        # 循环计算出每个点对应的最近中心点
        for i in range(len(X)):
            # 计算出每个点与中心点的距离
            distances = dist(X[i], C, 1)
            print(f"distances:{distances}")
            # 记录0 到 k-1个点中距离近的点
            cluster = np.argmin(distances)
            print(f"cluster:{cluster}")
            # 记录每个样例点与哪个中心点距离最近
            clusters[i] = cluster
            print(f"clusters[i]:{clusters[i]}")
        # 采用深拷贝将当前的中心点保存下来
        # print("the distinct of clusters: ", set(clusters))
        C_old = deepcopy(C)
        # 从属于中心点放到一个数组中，然后按照列的方向取平均值

        for i in range(k):
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            # print(points)
            # print(np.mean(points, axis=0))
            C[i] = np.mean(points, axis=0)
            # print(C[i])
        # print(C)

        # 计算新旧节点的距离
        print ('循环第%d次' % tmp)
        tmp = tmp + 1
        iteration_flag = dist(C, C_old, 1)
        print("新中心点与旧点的距离：", iteration_flag)
    #
    # 最终结果图示
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    fig, ax = plt.subplots()
    # 不同的子集使用不同的颜色
    for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
    ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='black')

    plt.show()


if __name__ == '__main__':
    # x = np.array([
    #     [0, 3],
    #     [0, 1],
    #     [0, 2]])
    #
    # y = np.array([
    #     [0, 0],
    #     [0, 0],
    #     [0, 0]
    # ])
    #
    # iteration_flag = dist(x,y,ax=1)
    #
    # print(iteration_flag.any())

    do_k_means()