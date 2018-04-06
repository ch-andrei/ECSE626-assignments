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
                 node_id,
                 precision,
                 sigma_C):
        self.X = X
        self.node_id = node_id
        self.cluster_inds = cluster_inds
        self.w = w

        # statistics
        wx = w.reshape(-1, 1) * X
        self.R = multiple_dot_product(wx, X).sum(axis=0)
        self.m = wx.sum(axis=0)
        self.N = w.sum() + precision  # add some small constant to prevent div by zero

        self.n_samples, self.dim = X.shape[:2]

        # quantization level; corresponds to the mean value of this cluster
        self.q = self.m / self.N

        # covariance matrix
        self.R_cov = self.R - single_vector_dot(self.m) / self.N + precision * np.eye(self.dim)

        self.q = np.nan_to_num(self.q)
        self.R_cov = np.nan_to_num(self.R_cov)

        # determine unit vector e; this is the direction in which cluster variance is the greatest
        V, E = np.linalg.eig(self.R_cov)
        index = np.argmax(np.abs(V))
        self.e_value = V[index]
        self.e = E[index]

        # print("before", self.R_cov)
        # add camera variance
        u, s, v = np.linalg.svd(self.R_cov)
        self.R_cov = u @ np.diag(s + sigma_C * sigma_C) @ v

        # print("after", self.R_cov)

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
                 max_splits=5,
                 delete_arrays_on_node_split=True,
                 precision=1e-6,
                 sigma_C=0.01,
                 min_var=5e-2
                 ):
        # X: samples; array of shape (N, d), where N is the number of samples and d is the dimensionality of each sample
        # W: sample weights; array of shape (N)
        # convergence_threshold: tree will be split until N * convergence_threshold < WTSE is achieved.

        self.min_var = min_var
        self.delete_arrays_on_node_split = delete_arrays_on_node_split
        self.precision = precision
        self.sigma_C = sigma_C

        self.X = X
        self.root_id = 1

        root_node = ClusterNode(X, w, np.arange(X.shape[0]), self.root_id, self.precision, self.sigma_C)
        self.nodes = {self.root_id: root_node}

        splits = 0
        while splits < max_splits:
            node = max(self.get_leaf_nodes(), key=lambda node: node.e_value)
            if node.e_value > self.min_var:
                split = self.split_node(node)
                if split:
                    splits += 1
                else:
                    # if self.split() returns false, no more splits are possible
                    break
            else:
                break


        # print("Finished clustering; final tree has {} nodes".format(len(self.nodes)))

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

    def split_node(self, node):
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
        id1, id2 = self.get_child_ids(node.node_id)
        self.nodes[id1] = ClusterNode(X_1, w1, cluster_inds1, id1, self.precision, self.sigma_C)
        self.nodes[id2] = ClusterNode(X_2, w2, cluster_inds2, id2, self.precision, self.sigma_C)

        return True

    def get_cluster_stats(self):
        means = []
        covars = []
        for node in self.get_leaf_nodes():
            means.append(node.q)
            covars.append(node.R_cov)
        return np.array(means), np.array(covars)

    # def compute_wtse(self):
    #     wtse = 0
    #     for node in self.get_leaf_nodes():
    #         wtse += (node.w[:, np.newaxis] * (node.X - node.q) ** 2).sum()
    #     # print("computed wtse", wtse)
    #     return wtse

    def __str__(self):
        string = "Cluster tree:\n"
        for node_id in self.nodes:
            string += "Node " + str(self.nodes[node_id]) + "\n"
        return string
