import torch
import torch.nn as nn

from dgl.sampling import node2vec_random_walk
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader


class Node2vec(nn.Module):
    """Node2vec model from paper dgb_node2vec: Scalable Feature Learning for Networks <https://arxiv.org/abs/1607.00653>
    Attributes
    ----------
    g: DGLGraph
        The graph.
    embedding_dim: int
        Dimension of node embedding.
    walk_length: int
        Length of each trace.
    p: float
        Likelihood of immediately revisiting a node in the walk.  Same notation as in the paper.
    q: float
        Control parameter to interpolate between breadth-first strategy and depth-first strategy.
        Same notation as in the paper.
    num_walks: int
        Number of random walks for each node. Default: 10.
    window_size: int
        Maximum distance between the center node and predicted node. Default: 5.
    num_negatives: int
        The number of negative samples for each positive sample.  Default: 5.
    use_sparse: bool
        If set to True, use PyTorch's sparse embedding and optimizer. Default: ``True``.
    weight_name : str, optional
        The name of the edge feature tensor on the graph storing the (unnormalized)
        probabilities associated with each edge for choosing the next node.

        The feature tensor must be non-negative and the sum of the probabilities
        must be positive for the outbound edges of all nodes (although they don't have
        to sum up to one).  The result will be undefined otherwise.

        If omitted, DGL assumes that the neighbors are picked uniformly.
    """

    def __init__(
            self,
            g,
            embedding_dim,
            walk_length,
            p,
            q,
            num_walks=10,
            window_size=5,
            num_negatives=5,
            use_sparse=True,
            weight_name=None,
    ):
        super(Node2vec, self).__init__()

        assert walk_length >= window_size

        self.g = g
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.num_walks = num_walks
        self.window_size = window_size
        self.num_negatives = num_negatives
        self.N = self.g.num_nodes()
        if weight_name is not None:
            self.prob = weight_name
        else:
            self.prob = None

        self.embedding = nn.Embedding(self.N, embedding_dim, sparse=use_sparse)

    def reset_parameters(self):
        self.embedding.reset_parameters()

    def sample(self, batch):
        """
        Generate positive and negative samples.
        Positive samples are generated from random walk
        Negative samples are generated from random sampling
        batch: list of scalar tensor. size: batch_size.
        :return
            [B*num_walks* 47, window_size]
            [B*num_negatives* 47, window_size]
        """
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)  # [B,]

        # [B*num_walks, ]
        batch = batch.repeat(self.num_walks)
        # positive [B*num_walks, walk_length+1]
        pos_traces = node2vec_random_walk(
            self.g, batch, self.p, self.q, self.walk_length, self.prob
        )

        # rolling window, # [B*num_walks, 47, window_size]
        pos_traces = pos_traces.unfold(1, self.window_size, 1)

        # [B*num_walks* 47, window_size]
        pos_traces = pos_traces.contiguous().view(-1, self.window_size)

        neg_traces = self.neg_sampling(batch)

        return pos_traces, neg_traces

    def pos_sampling(self, batch):
        pass

    def neg_sampling(self, pos_batch):
        """
        negative sampling.
        :param pos_batch: [B*num_walks, ]
        """
        # [B*num_negatives, ]
        neg_batch = pos_batch.repeat(self.num_negatives)

        # [B*num_negatives, walk_length]
        neg_traces = torch.randint(
            self.N, (neg_batch.size(0), self.walk_length)
        )

        # [B*num_negatives, walk_length+1]
        # neg_batch.view(-1, 1), neg_traces，他们两者的行数相同，但是列数不相同
        neg_traces = torch.cat([neg_batch.view(-1, 1), neg_traces], dim=-1)
        # rolling window, [B*num_negatives, 47, window_size]
        neg_traces = neg_traces.unfold(1, self.window_size, 1)
        # [B*num_negatives* 47, window_size]
        neg_traces = neg_traces.contiguous().view(-1, self.window_size)

        return neg_traces

    def forward(self, nodes=None):
        """
        Returns the embeddings of the input nodes
        Parameters
        ----------
        nodes: Tensor, optional
            Input nodes, if set `None`, will return all the node embedding.

        Returns
        -------
        Tensor
            Node embedding

        """
        emb = self.embedding.weight
        if nodes is None:
            return emb
        else:
            return emb[nodes]

    def loss(self, pos_trace, neg_trace):
        """
        Computes the loss given positive and negative random walks.
        Parameters
        ----------
        pos_trace: Tensor
            positive random walk trace
            shape [B*num_walks* 47, window_size]
        neg_trace: Tensor
            negative random walk trace
            shape [B*num_negatives* 47, window_size]
        """
        pos_loss = self.pos_loss(pos_trace)
        neg_loss = self.neg_loss(neg_trace)

        return pos_loss + neg_loss

    def pos_loss(self, pos_trace):
        """
        pos_trace: Tensor
            positive random walk trace
            shape [B*num_walks* 47, window_size]
        """
        e = 1e-15

        # Positive，拆分成两部分，第一列、其它列
        #  pos_start [B*num_walks* 47]
        #  pos_rest [B*num_walks* 47, window_size-1]
        pos_start, pos_rest = (
            pos_trace[:, 0],
            pos_trace[:, 1:].contiguous(),
        )  # start node and following trace

        # 分别经过embed layer
        #  w_start [B*num_walks* 47, 1, embed_size]
        #  w_rest [B*num_walks* 47, window_size-1, embed_size]
        w_start = self.embedding(pos_start).unsqueeze(dim=1)
        w_rest = self.embedding(pos_rest)

        # 算内积   [B*num_walks* 47* (window_size-1),]
        # todo 他们是负相关的吗？
        pos_out = (w_start * w_rest).sum(dim=-1).view(-1)

        # compute loss, todo logloss?
        # scalar
        pos_loss = -torch.log(torch.sigmoid(pos_out) + e).mean()

        return pos_loss

    def neg_loss(self, neg_trace):
        """
        以下实现可以看出来，neg sampling算loss的时候，share了输入层的weights。
        还可以看出来neg_loss和pos_loss的唯一区别，就是输入的shape不相同,以及loss公式不同。

        neg_trace: Tensor
            negative random walk trace
            shape [B*num_negatives* 47, window_size]
        """
        e = 1e-15

        # Negative
        # neg_start：shape[B * num_negatives * 47, ]
        # neg_rest：shape[B * num_negatives * 47, window_size-1]
        neg_start, neg_rest = neg_trace[:, 0], neg_trace[:, 1:].contiguous()

        # 分别经过embed layer
        #  w_start [B*num_negatives* 47, 1, embed_size]
        #  w_rest [B*num_negatives* 47, window_size-1, embed_size]
        w_start = self.embedding(neg_start).unsqueeze(dim=1)
        w_rest = self.embedding(neg_rest)

        neg_out = (w_start * w_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(neg_out) + e).mean()

        # scalar
        return neg_loss

    def loader(self, batch_size):
        """
        挨个node遍历

        Parameters
        ----------
        batch_size: int
            batch size

        Returns
            [B*num_walks* 47, window_size]
            [B*num_negatives* 47, window_size]
        -------
        DataLoader
            Node2vec training data loader

        """
        return DataLoader(
            torch.arange(self.N),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.sample,
        )

    @torch.no_grad()
    def evaluate(self, x_train, y_train, x_val, y_val):
        """
        Evaluate the quality of embedding vector via a downstream classification task with logistic regression.
        """
        x_train = self.forward(x_train)
        x_val = self.forward(x_val)

        x_train, y_train = x_train.cpu().numpy(), y_train.cpu().numpy()
        x_val, y_val = x_val.cpu().numpy(), y_val.cpu().numpy()
        lr = LogisticRegression(
            solver="lbfgs", multi_class="auto", max_iter=150
        ).fit(x_train, y_train)

        return lr.score(x_val, y_val)


class Node2vecModel(object):
    """
    Wrapper of the ``Node2Vec`` class with a ``train`` method.
    Attributes
    ----------
    g: DGLGraph
        The graph.
    embedding_dim: int
        Dimension of node embedding.
    walk_length: int
        Length of each trace.
    p: float
        Likelihood of immediately revisiting a node in the walk.
    q: float
        Control parameter to interpolate between breadth-first strategy and depth-first strategy.
    num_walks: int
        Number of random walks for each node. Default: 10.
    window_size: int
        Maximum distance between the center node and predicted node. Default: 5.
    num_negatives: int
        The number of negative samples for each positive sample.  Default: 5.
    use_sparse: bool
        If set to True, uses PyTorch's sparse embedding and optimizer. Default: ``True``.
    weight_name : str, optional
        The name of the edge feature tensor on the graph storing the (unnormalized)
        probabilities associated with each edge for choosing the next node.

        The feature tensor must be non-negative and the sum of the probabilities
        must be positive for the outbound edges of all nodes (although they don't have
        to sum up to one).  The result will be undefined otherwise.

        If omitted, DGL assumes that the neighbors are picked uniformly. Default: ``None``.
    eval_set: list of tuples (Tensor, Tensor)
        [(nodes_train,y_train),(nodes_val,y_val)]
        If omitted, model will not be evaluated. Default: ``None``.
    eval_steps: int
        Interval steps of evaluation.
        if set <= 0, model will not be evaluated. Default: ``None``.
    device: str
        device, default 'cpu'.
    """

    def __init__(
            self,
            g,
            embedding_dim,
            walk_length,
            p=1.0,
            q=1.0,
            num_walks=1,
            window_size=5,
            num_negatives=5,
            use_sparse=True,
            weight_name=None,
            eval_set=None,
            eval_steps=-1,
            device="cpu",
    ):
        self.model = Node2vec(
            g,
            embedding_dim,
            walk_length,
            p,
            q,
            num_walks,
            window_size,
            num_negatives,
            use_sparse,
            weight_name,
        )
        self.g = g
        self.use_sparse = use_sparse
        self.eval_steps = eval_steps
        self.eval_set = eval_set

        if device == "cpu":
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _train_step(self, model, loader, optimizer, device):
        model.train()
        total_loss = 0
        for pos_traces, neg_traces in loader:
            # [B * num_walks * 47, window_size]
            # [B * num_negatives * 47, window_size]
            pos_traces, neg_traces = pos_traces.to(device), neg_traces.to(
                device
            )
            optimizer.zero_grad()
            loss = model.loss(pos_traces, neg_traces)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def _evaluate_step(self):
        nodes_train, y_train = self.eval_set[0]
        nodes_val, y_val = self.eval_set[1]

        acc = self.model.evaluate(nodes_train, y_train, nodes_val, y_val)
        return acc

    def train(self, epochs, batch_size, learning_rate=0.01):
        """

        Parameters
        ----------
        epochs: int
            num of train epoch
        batch_size: int
            batch size
        learning_rate: float
            learning rate. Default 0.01.

        """

        self.model = self.model.to(self.device)
        loader = self.model.loader(batch_size)
        if self.use_sparse:
            optimizer = torch.optim.SparseAdam(
                list(self.model.parameters()), lr=learning_rate
            )
        else:
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=learning_rate
            )
        for i in range(epochs):
            loss = self._train_step(self.model, loader, optimizer, self.device)
            if self.eval_steps > 0:
                if epochs % self.eval_steps == 0:
                    acc = self._evaluate_step()
                    print(
                        "Epoch: {}, Train Loss: {:.4f}, Val Acc: {:.4f}".format(
                            i, loss, acc
                        )
                    )

    def embedding(self, nodes=None):
        """
        Returns the embeddings of the input nodes
        Parameters
        ----------
        nodes: Tensor, optional
            Input nodes, if set `None`, will return all the node embedding.

        Returns
        -------
        Tensor
            Node embedding.
        """

        return self.model(nodes)
