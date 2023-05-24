import dgl
import numpy as np
import torch as th


class Sampler:
    def __init__(
        self, graph, walk_length, num_walks, window_size, num_negative
    ):
        self.graph = graph
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.num_negative = num_negative
        self.node_weights = self.compute_node_sample_weight()

    def sample(self, batch, sku_info):
        """
        Given a batch of target nodes, sample postive pairs and negative pairs from the graph.
        :param batch：list of size B. 每个元素是一个scalar tensor。 应该node id。
        :param sku_info: dict. sku_info[sku_id] = [sku_id, brand_id, shop_id, cate_id]. key里与value里的第一个元素是相同的，是node id。也就是skku_info里有所有node的信息。
        """
        # numpy。shape：[B*self.num_walks,]
        # tensor居然能够转换成numpy！！！
        batch = np.repeat(batch, self.num_walks)

        # a list. 每个元素是（src, dst, label）
        pos_pairs = self.generate_pos_pairs(batch)

        # a list. 每个元素是（src, dst, label）
        neg_pairs = self.generate_neg_pairs(pos_pairs)

        # get sku info with id
        # 将node扩展到side_info。
        srcs, dsts, labels = [], [], []
        for pair in pos_pairs + neg_pairs:
            src, dst, label = pair
            src_info = sku_info[src]
            dst_info = sku_info[dst]

            srcs.append(src_info)
            dsts.append(dst_info)
            labels.append(label)

        return th.tensor(srcs), th.tensor(dsts), th.tensor(labels)

    def filter_padding(self, traces):
        for i in range(len(traces)):
            traces[i] = [x for x in traces[i] if x != -1]

    def generate_pos_pairs(self, nodes):
        """
        For seq [1, 2, 3, 4] and node NO.2,
        the window_size=1 will generate:
            (1, 2) and (2, 3)

        :param nodes: numpy. shape:[B*self.num_walks,]
        """
        # random walk

        # traces: Tensor. shape:[B*self.num_walks, self.walk_length+1]。从node里面的节点作为起始节点，为每个起始节点寻找长为self.walk_length+1的路径。
        # types: Tensor. [self.walk_length+1, ]
        # 不够self.walk_length+1，就用-1 padding。
        traces, types = dgl.sampling.random_walk(
            g=self.graph, nodes=nodes, length=self.walk_length, prob="weight"
        )
        traces = traces.tolist()
        self.filter_padding(traces)

        # skip-gram
        # 分别向左、向右看self.window_size个。
        pairs = []
        for trace in traces:
            for i in range(len(trace)):
                center = trace[i]
                left = max(0, i - self.window_size)
                right = min(len(trace), i + self.window_size + 1)

                # 正样本
                pairs.extend([[center, x, 1] for x in trace[left:i]])
                pairs.extend([[center, x, 1] for x in trace[i + 1 : right]])

        return pairs

    def compute_node_sample_weight(self):
        """
        Using node degree as sample weight
        """
        return self.graph.in_degrees().float()

    def generate_neg_pairs(self, pos_pairs):
        """
        Sample based on node freq in traces, frequently shown
        nodes will have larger chance to be sampled as
        negative node.
        """
        # sample `self.num_negative` neg dst node
        # for each pos node pair's src node.
        negs = th.multinomial(
            self.node_weights,
            len(pos_pairs) * self.num_negative,
            replacement=True,
        ).tolist()

        tar = np.repeat([pair[0] for pair in pos_pairs], self.num_negative)
        assert len(tar) == len(negs)
        neg_pairs = [[x, y, 0] for x, y in zip(tar, negs)]

        return neg_pairs
