#!/usr/bin/env python
# coding: utf-8

# # 官方作者Aditya Grover代码解读
# 
# 同济子豪兄
# 
# 2022-7-17
# 
# ## 参考资料
# 
# Node2Vec官方作者Aditya Grover代码：https://github.com/aditya-grover/node2vec
# 
# 深度之眼 赵老师

# ## 导入工具包
import warnings

from sklearn.cluster import KMeans
from scipy import spatial

warnings.filterwarnings('ignore')
import argparse
import numpy as np
import networkx as nx
from gensim.models import Word2Vec
import random

import matplotlib.pyplot as plt


# ## 读入命令行参数

def parse_args():
    """
    Parses the node2vec arguments.
    """
    # 使用parser加载信息
    parser = argparse.ArgumentParser(description="Run node2vec.")
    # 输入文件：邻接表
    parser.add_argument('--input', nargs='?', default='karate.edgelist',
                        help='Input graph path')
    # 输出文件：节点嵌入表
    parser.add_argument('--output', nargs='?', default='karate.emb',
                        help='Embeddings path')
    # embedding嵌入向量维度
    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')
    # 随机游走序列长度
    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')
    # 每个节点生成随机游走序列次数
    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')
    # word2vec窗口大小，word2vec参数
    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')
    # SGD优化时epoch数量，word2vec参数
    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')
    # 并行化核数，word2vec参数
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')
    # 参数p
    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')
    # 参数q
    parser.add_argument('--q', type=float, default=2,
                        help='Inout hyperparameter. Default is 2.')
    # 连接是否带权重
    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    # 有向图还是无向图
    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args(args=[])


# # 可视化
# pos = nx.spring_layout(G, seed=4)
# nx.draw(G, pos, with_labels=True)
# plt.show()


# ## Alias Sampling
# 
# 参考博客
# 
# https://keithschwarz.com/darts-dice-coins
# https://www.bilibili.com/video/av798804262
# https://www.cnblogs.com/Lee-yl/p/12749070.html


def alias_setup(probs):
    """
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    :params probs: 一个节点，到其它节点的归一probs
    :return J, q
    """
    K = len(probs)
    # q corrsespond to Prob
    q = np.zeros(K)
    # J Alias
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []

    # 将各个概率分成两组，一组的概率值大于1，另一组的概率值小于1
    for kk, prob in enumerate(probs):
        q[kk] = K * prob  # 每类事件的概率 乘 事件个数

        # 判定”劫富”和“济贫“的对象
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    # 使用贪心算法，将概率值小于1的不断填满
    # pseudo code step 3
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        # 更新概率值，劫富济贫，削峰填谷
        q[large] = q[large] - (1 - q[small])
        if q[large] < 1.0:
            # 这里不可能是负数。因为q[large]在更新前是大于等于1的
            smaller.append(large)  # 把被打倒的土豪归为贫农
        else:
            # 仍然大于等于1，就给他放回去larger里，后面还可以用来填别人的坑
            larger.append(large)

    return J, q


def alias_draw(J, q):
    """
    Draw sample from a non-uniform discrete distribution using alias sampling.
    O(1)的采样
    这是核心方法！！！
    """
    K = len(J)  # 事件个数

    kk = int(np.floor(np.random.rand() * K))  # 生成1到K的随机整数
    if np.random.rand() < q[kk]:
        return kk  # 取自己本来就对应的事件
    else:
        return J[kk]  # 取alias事件


def get_alias_edge(src, dst, args, G):
    """
    src就是t，dst就是v。
    返回：
    """
    p = args.p
    q = args.q

    unnormalized_probs = []

    # 论文3.2.2节核心算法，计算各条边的转移权重
    for dst_nbr in sorted(G.neighbors(dst)):
        if dst_nbr == src:
            # dtx = 0
            unnormalized_probs.append(G[dst][dst_nbr]['weight'] / p)
        elif G.has_edge(dst_nbr, src):
            # dtx = 1
            unnormalized_probs.append(G[dst][dst_nbr]['weight'])
        else:
            # dtx = 2
            unnormalized_probs.append(G[dst][dst_nbr]['weight'] / q)

    # 归一化各条边的转移权重
    norm_const = sum(unnormalized_probs)
    normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

    # 执行 Alias Sampling
    return alias_setup(normalized_probs)


# ## 生成一条随机游走序列
def node2vec_walk(walk_length, start_node, alias_nodes, alias_edges, G):
    """
    从指定的起始节点，生成一个随机游走序列
    """
    # 上一步计算出的alias table，完成O(1)的采样

    walk = [start_node]

    #  直到生成长度为walk_length的节点序列位为止
    while len(walk) < walk_length:
        cur = walk[-1]
        # 对邻居节点排序，目的是和alias table计算时的顺序对应起来
        cur_nbrs = sorted(G.neighbors(cur))
        if len(cur_nbrs) > 0:
            # 节点序列只有一个节点的情况
            if len(walk) == 1:
                J, q = alias_nodes[cur]
            # 节点序列大于一个节点的情况
            else:
                # 看前一个节点,prev是论文中的节点t
                prev = walk[-2]
                J, q = alias_edges[(prev, cur)]

            rand_idx = alias_draw(J, q)
            next_node = cur_nbrs[rand_idx]
            walk.append(next_node)
        else:
            break

    return walk


# ## 采样得到所有随机游走序列
def simulate_walks(num_walks, walk_length, alias_nodes, alias_edges, G):
    """
    图中每个节点作为起始节点，生成 num_walk 个随机游走序列
    """
    walks = []
    nodes = list(G.nodes())
    print('Walk iteration:')
    for walk_iter in range(num_walks):
        print(str(walk_iter + 1), '/', str(num_walks))
        # 打乱节点顺序
        random.shuffle(nodes)
        for node in nodes:
            one_walk_seq = node2vec_walk(walk_length=walk_length, start_node=node, alias_nodes=alias_nodes,
                                         alias_edges=alias_edges, G=G)
            walks.append(one_walk_seq)

    return walks


def make_graph(args):
    # 连接带权重
    if args.weighted:
        G = nx.read_edgelist(args.input, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    # 连接不带权重
    else:
        G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = np.abs(np.random.randn())

    # 无向图
    if not args.directed:
        G = G.to_undirected()
        print('not directed')

    return G


def make_alias(G):
    # key是节点号，value是一个tuple，2个元素，一个是J，一个是q
    alias_nodes = {}

    # 节点概率alias sampling和归一化
    for node in G.nodes():
        # node的所有邻居的weight/probs
        unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
        norm_const = sum(unnormalized_probs)

        # 归一化
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
        alias_nodes[node] = alias_setup(normalized_probs)

        # 信息展示, debug
        if node == 25:
            print('25号节点')
            print(unnormalized_probs)
            print(norm_const)
            print(normalized_probs)
            print(alias_nodes[node])

    return alias_nodes


def make_edges(args, G):
    """
    前一个节点是t，当前节点是v，计算周围邻居节点的采样概率。
    对每条边都如是计算。
    """
    is_directed = args.directed

    alias_edges = {}

    # 边概率alias sampling和归一化
    if is_directed:
        for edge in G.edges():
            alias_edges[edge] = get_alias_edge(edge[0], edge[1], args, G)
    else:
        for edge in G.edges():
            # 两个方向
            alias_edges[edge] = get_alias_edge(edge[0], edge[1], args, G)
            alias_edges[(edge[1], edge[0])] = get_alias_edge(edge[1], edge[0], args, G)

    # 边概率alias sampling和归一化

    return alias_edges


def make_training_data(args, G):
    """
    这是node2vec的核心代码
    """

    alias_nodes = make_alias(G)

    alias_edges = make_edges(args, G)

    # 生成训练用的随机游走序列
    walks = simulate_walks(args.num_walks, args.walk_length, alias_nodes, alias_edges, G)

    # 将node的类型int转化为string
    walk_str = []
    for walk in walks:
        tmp = []
        for node in walk:
            tmp.append(str(node))
        walk_str.append(tmp)

    return walk_str


def val_model(model, G):
    # ## 结果分析和可视化

    # ### 查看 Node Embedding
    model.wv.get_vector('17')

    # ### 查找相似节点
    pos = nx.spring_layout(G, seed=4)
    nx.draw(G, pos, with_labels=True)
    plt.show()

    # 节点对 相似度
    print(model.wv.similarity('25', '26'))
    print(model.wv.similarity('17', '25'))

    # 找到最相似的节点
    print(model.wv.most_similar('25'))

    # 自定义相似性距离度量指标

    def cos_similarity(v1, v2):
        # 余弦相似度
        return 1 - spatial.distance.cosine(v1, v2)

    v1 = model.wv.get_vector('25')
    v2 = model.wv.get_vector('26')

    cos_similarity(v1, v2)

    # ### Node Embedding聚类

    # #### 运行聚类

    # # DBSCAN聚类
    # from sklearn.cluster import DBSCAN
    # cluster_labels = DBSCAN(eps=0.5, min_samples=6).fit(X).labels_
    # print(cluster_labels)

    # KMeans聚类

    X = model.wv.vectors
    cluster_labels = KMeans(n_clusters=3, random_state=9).fit(X).labels_

    # #### 将networkx中的节点和词向量中的节点对应
    colors = []
    nodes = list(G.nodes)
    for node in nodes:  # 按 networkx 的顺序遍历每个节点
        idx = model.wv.key_to_index[str(node)]  # 获取这个节点在 embedding 中的索引号
        colors.append(cluster_labels[idx])  # 获取这个节点的聚类结果

    # #### 可视化聚类效果
    pos = nx.spring_layout(G, seed=4)
    nx.draw(G, pos, node_color=colors, with_labels=True)
    plt.show()


def main():
    args = parse_args()

    G = make_graph(args)

    walk_str = make_training_data(args, G)

    print('----training------')
    # ## 利用word2vec训练Node2Vec模型
    # 调用 gensim 包运行 word2vec
    model = Word2Vec(walk_str, vector_size=args.dimensions, window=args.window_size, min_count=0, sg=1,
                     workers=args.workers)

    # 导出embedding文件
    model.wv.save_word2vec_format(args.output)

    val_model(model, G)


if __name__ == '__main__':
    main()
