import numpy as np

def multiple_dot_product(a, b):
    n, dim = a.shape[:2]
    t1 = np.repeat(a, dim)
    t2 = np.repeat(b, dim, axis=0).flatten()
    t = (t1 * t2).reshape(n, dim, dim)
    return t

def single_vector_dot(a):
    return np.dot(a[:, np.newaxis], a[np.newaxis, :])

class ClusterNode:
    def __init__(self,
                 X,
                 w,
                 cluster_inds,
                 node_id):
        self.X = X
        self.node_id = node_id
        self.cluster_inds = cluster_inds
        self.w = w

        # statistics
        wx = w.reshape(-1, 1) * X
        self.R = multiple_dot_product(wx, X).sum(axis=0)
        self.m = wx.sum(axis=0)
        self.N = w.sum()

        self.n_samples, self.dim = X.shape[:2]

        # quantization level; corresponds to the mean value of this cluster
        self.q = self.m / self.N

        # covariance matrix
        self.R_cov = self.R - single_vector_dot(self.m) / self.N

        # determine unit vector e; this is the direction in which cluster variance is the greatest
        V, E = np.linalg.eig(self.R_cov)
        index = np.argmax(np.abs(V))
        self.e_value = V[index]
        self.e = E[index]

    def get_cluster_mask(self, img):
        h, w = img.shape[:2]
        mask = np.zeros(h * w, np.bool)
        mask[self.cluster_inds] = 1
        return mask

    def __str__(self):
        return "[id={}, X_size={}]".format(self.node_id, self.n_samples)


class ClusterTree:
    def __init__(self,
                 X,
                 w,
                 convergence_threshold=0.01,
                 delete_arrays_on_node_split=True
                 ):
        # X: samples; array of shape (N, d), where N is the number of samples and d is the dimensionality of each sample
        # W: sample weights; array of shape (N)
        # convergence_threshold: tree will be split until N * convergence_threshold < WTSE is achieved.

        self.convergence_threshold = convergence_threshold
        self.delete_arrays_on_node_split = delete_arrays_on_node_split

        self.X = X
        self.root_id = 1

        root_node = ClusterNode(X, w, np.arange(X.shape[0]), self.root_id)
        self.nodes = {self.root_id: root_node}

        while self.compute_wtse() > X.shape[0] * self.convergence_threshold:
            self.split()

        print("Finished clustering; final tree has {} nodes".format(len(self.nodes)))

    def get_child_ids(self, node_id):
        return node_id * 2, node_id * 2 + 1

    def is_leaf(self, node_id):
        id1, id2 = self.get_child_ids(node_id)
        return not (id1 in self.nodes and id2 in self.nodes)

    def get_leaf_nodes(self):
        leafs = []
        for node_id in self.nodes:
            if self.is_leaf(node_id):
                leafs.append(self.nodes[node_id])
        return leafs

    def split(self):
        sorted_nodes = sorted(self.get_leaf_nodes(), key=lambda node: node.e_value)

        index = 1
        while index <= len(sorted_nodes):
            if self.split_node(sorted_nodes[-index].node_id):
                break
            index += 1

    def split_node(self, node_id):
        node = self.nodes[node_id]

        # print("Splitting node {}".format(node))

        # compute threshold for splitting
        v_tresh = (node.e * node.q).sum()
        C = (node.e * node.X).sum(axis=1)

        # split samples
        split_mask = C <= v_tresh
        inds1 = split_mask == 1
        inds2 = np.logical_not(inds1)

        # split
        X_1 = node.X[inds1]
        X_2 = node.X[inds2]
        w1 = node.w[inds1]
        w2 = node.w[inds2]
        cluster_inds1 = node.cluster_inds[inds1]
        cluster_inds2 = node.cluster_inds[inds2]

        if len(X_1) < 1 or len(X_2) < 1:
            # unable to split into two nodes if the split results in a node with zero elements
            return False

        if self.delete_arrays_on_node_split:
            # delete array from parent node to save memory
            del node.X, node.w, node.cluster_inds

        # children
        id1, id2 = self.get_child_ids(node_id)
        self.nodes[id1] = (ClusterNode(X_1, w1, cluster_inds1, id1))
        self.nodes[id2] = (ClusterNode(X_2, w2, cluster_inds2, id2))

        return True

    def get_cluster_stats(self):
        means = []
        covars = []
        for node in self.get_leaf_nodes():
            means.append(node.q)
            covars.append(node.R_cov)
        return means, covars

    def compute_wtse(self):
        wtse = 0
        for node in self.get_leaf_nodes():
            wtse += (node.w[:, np.newaxis] * (node.X - node.q) ** 2).sum()
        print("computed wtse", wtse)
        return wtse

    def __str__(self):
        string = "Cluster tree:\n"
        for node_id in self.nodes:
            string += "Node " + str(self.nodes[node_id]) + "\n"
        return string
