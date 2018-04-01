import numpy as np

delete_arrays_on_node_split = True

def multiple_dot_product(a, b):
    n, dim = a.shape[:2]
    t1 = np.repeat(a, dim)
    t2 = np.repeat(b, dim, axis=0).flatten()
    return (t1 * t2).reshape(n, dim, dim)

def single_vector_dot(a):
    return np.dot(a[:, np.newaxis], a[np.newaxis, :])

class ClusterNode:
    def __init__(self,
                 X,
                 cluster_inds,
                 node_id):
        self.X = X
        self.node_id = node_id
        self.cluster_inds = cluster_inds

        # statistics
        self.R = multiple_dot_product(X, X).sum(axis=0)
        self.m = X.sum(axis=0)
        self.N, self.dim = X.shape[:2]

        # quantization level; corresponds to the average value of all samples in this cluster
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
        return "[id={}, X_size={}]".format(self.node_id, self.N)


class ClusterTree:
    def __init__(self,
                 X
                 ):
        # X: array of shape (N, d), where N is the number of samples and d is the dimensionality of each sample
        self.X = X
        self.root_id = 1
        self.nodes = {self.root_id: ClusterNode(X, np.arange(X.shape[0]), self.root_id)}

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
        leaf_nodes = self.get_leaf_nodes()
        node_to_split = max(leaf_nodes, key=lambda node: node.e_value)
        self.split_node(node_to_split.node_id)

    def split_node(self, node_id):
        node = self.nodes[node_id]

        print("Splitting node {}".format(node))

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
        cluster_inds1 = node.cluster_inds[inds1]
        cluster_inds2 = node.cluster_inds[inds2]

        if delete_arrays_on_node_split:
            # delete array from parent node to save memory
            del node.X, node.cluster_inds

        # children
        id1, id2 = self.get_child_ids(node_id)
        self.nodes[id1] = (ClusterNode(X_1, cluster_inds1, id1))
        self.nodes[id2] = (ClusterNode(X_2, cluster_inds2, id2))

    def __str__(self):
        string = "Cluster tree:\n"
        for node_id in self.nodes:
            string += "Node " + str(self.nodes[node_id]) + "\n"
        return string
