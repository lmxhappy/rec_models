import torch as th


class EGES(th.nn.Module):
    def __init__(self, dim, num_nodes, num_brands, num_shops, num_cates):
        super(EGES, self).__init__()
        self.dim = dim
        # embeddings for nodes
        base_embeds = th.nn.Embedding(num_nodes, dim)
        brand_embeds = th.nn.Embedding(num_brands, dim)
        shop_embeds = th.nn.Embedding(num_shops, dim)
        cate_embeds = th.nn.Embedding(num_cates, dim)

        # 是一个list。每个元素是th.nn.Embedding（B，dim），分别对应node、brand、shop、cate
        self.embeds = [base_embeds, brand_embeds, shop_embeds, cate_embeds]

        # weights for each node's side information
        self.side_info_weights = th.nn.Embedding(num_nodes, 4)

    def forward(self, srcs, dsts):
        """
        分别算开头节点、目标节点的embed。

        :param srcs: sku_id(就是item id), brand_id, shop_id, cate_id. shape:[B, num_side_info]. num_side_info包括node本身。
        :type srcs:
        :param dsts: 同srcs。
        :type dsts:
        :return: 两个。每个的shape都是[B, dim]
        :rtype:
        """
        srcs = self.query_node_embed(srcs)
        dsts = self.query_node_embed(dsts)

        return srcs, dsts

    def query_node_embed(self, nodes):
        """
        @nodes: tensor of shape (batch_size, num_side_info)
        return: (batch_size, dim)
        """
        batch_size = nodes.shape[0]
        # query side info weights, (batch_size, 4)
        # exp处理
        side_info_weights = th.exp(self.side_info_weights(nodes[:, 0]))

        # merge all embeddings

        # side_info_weighted_embeds_sum: 现在是空的，for后，4个元素。每个元素的shape:[B, dim]
        side_info_weighted_embeds_sum = []

        # shape: 现在是空的，for后，4个元素。每个元素的shape:[B, 4]
        side_info_weights_sum = []
        for i in range(4):
            # weights for i-th side info, (batch_size, ) -> (batch_size, 1)
            i_th_side_info_weights = side_info_weights[:, i].view(
                (batch_size, 1)
            )

            # [B, 16]
            i_th_side_info_embed = self.embeds[i](nodes[:, i])

            # batch of i-th side info embedding * its weight, (batch_size, dim)
            side_info_weighted_embeds_sum.append(i_th_side_info_weights * i_th_side_info_embed)

            side_info_weights_sum.append(i_th_side_info_weights)

        # stack: (batch_size, 4, dim), sum: (batch_size, dim)
        side_info_weighted_embeds_sum = th.sum(
            th.stack(side_info_weighted_embeds_sum, dim=1), dim=1
        )

        # stack: (batch_size, 4), sum: (batch_size, )
        side_info_weights_sum = th.sum(
            th.stack(side_info_weights_sum, dim=1), dim=1
        )

        # (batch_size, dim)
        H = side_info_weighted_embeds_sum / side_info_weights_sum

        return H

    def loss(self, srcs, dsts, labels):
        dots = th.sigmoid(th.sum(srcs * dsts, axis=1))
        dots = th.clamp(dots, min=1e-7, max=1 - 1e-7)

        return th.mean(
            -(labels * th.log(dots) + (1 - labels) * th.log(1 - dots))
        )
