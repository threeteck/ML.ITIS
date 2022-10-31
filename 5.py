import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import lil_matrix


class DSU:
    def __init__(self):
        self.parent = dict()
        self.rank = dict()

    def make_set(self, v):
        self.parent[v] = v
        self.rank[v] = 1

    def find_set(self, v):
        if v == self.parent[v]:
            return v
        self.parent[v] = self.find_set(self.parent[v])
        return self.parent[v]

    def union_sets(self, a, b):
        a = self.find_set(a)
        b = self.find_set(b)
        if a != b:
            if self.rank[a] < self.rank[b]:
                (a, b) = (b, a)
            self.parent[b] = a
            if self.rank[a] == self.rank[b]:
                self.rank[a] += 1


def sort_coo(m, reverse=False):
    tuples = zip(m.row, m.col, m.data)
    return sorted(tuples, key=lambda x: x[2], reverse=reverse)


def get_msf(matrix):
    n = matrix.shape[0]
    dsu = DSU()
    edges = sort_coo(matrix)
    result_matrix = lil_matrix((n, n))
    for i in range(n):
        dsu.make_set(i)

    for a, b, weight in edges:
        if dsu.find_set(a) != dsu.find_set(b):
            result_matrix[a, b] = result_matrix[b, a] = weight
            dsu.union_sets(a, b)

    return result_matrix.tocoo()


def generate_graph(n_nodes=10, edge_density=0.5):
    matrix = lil_matrix((n_nodes, n_nodes), dtype=np.int32)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if np.random.random() <= edge_density:
                matrix[i, j] = matrix[j, i] = np.random.randint(0, 100)

    return matrix.tocoo()


def get_nx_graph(matrix):
    graph = nx.from_scipy_sparse_array(matrix)
    return graph, nx.spring_layout(graph, seed=7), nx.get_edge_attributes(graph, "weight")


def draw_graph(graph, pos, edge_labels, index, edge_color=None):
    plt.figure(index)
    ax = plt.gca()
    plt.axis("off")
    plt.tight_layout()
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=400)
    nx.draw_networkx_edges(graph, pos, ax=ax, width=4, edge_color=edge_color)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels, ax=ax)

    return ax


def split_to_clusters(matrix, k=5):
    edges = sort_coo(matrix, reverse=True)
    result_matrix = matrix.copy().tolil()
    for a, b, w in edges[:(k - 1) * 2]:
        result_matrix[a, b] = 0

    result_matrix = result_matrix.tocoo()
    n = matrix.shape[0]
    dsu = DSU()
    for i in range(n):
        dsu.make_set(i)
    for i, j, w in zip(result_matrix.row, result_matrix.col, result_matrix.data):
        if w != 0:
            dsu.union_sets(i, j)

    clusters = []
    current_cluster = 0
    cluster_map = {}
    for i in range(n):
        cluster = dsu.find_set(i)
        if cluster not in cluster_map:
            cluster_map[cluster] = current_cluster
            current_cluster += 1
        clusters.append(cluster_map[cluster])
    return result_matrix, clusters


def main():
    matrix = generate_graph(10, 0.5)
    msf = get_msf(matrix)
    graph, pos, edge_labels = get_nx_graph(matrix)
    ax = draw_graph(graph, pos, edge_labels, 0)
    nx.draw_networkx_edges(graph, pos, ax=ax, width=6, edge_color='red', edgelist=list(zip(msf.row, msf.col)))
    nx.draw_networkx_edge_labels(graph, pos, edge_labels, ax=ax)

    graph, pos, edge_labels = get_nx_graph(msf)
    draw_graph(graph, pos, edge_labels, 1, edge_color='red')

    split_graph, clusters = split_to_clusters(msf)
    label_map = dict()
    for i in range(len(clusters)):
        label_map[i] = clusters[i]
    graph, pos, edge_labels = get_nx_graph(split_graph)
    draw_graph(graph, pos, edge_labels, 2)
    nx.draw_networkx_labels(graph, pos, font_size=20, font_family="sans-serif", labels=label_map)
    print(clusters)

    plt.show()


if __name__ == '__main__':
    main()