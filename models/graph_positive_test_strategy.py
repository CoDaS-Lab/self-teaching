import queue
import numpy as np
from models import dag
from models import utils


class GraphPositiveTestStrategy:
    def __init__(self, graphs):
        self.graphs = graphs
        self.n_graphs = len(graphs)

        self.actions = np.array([0, 1, 2])
        self.n_actions = len(self.actions)

    def descendant_links(self, graph, action):
        """Calculate the number of descendant links of a graph after
        performing a specified action"""
        descendants = queue.Queue()
        n_descendants = 0

        items = list(graph.get_children(action, graph.graph))
        for item in items:
            descendants.put(item)

        while not descendants.empty():
            n_descendants += 1
            descendant = descendants.get()
            items = list(graph.get_children(descendant, graph.graph))
            for item in items:
                descendants.put(item)

        return n_descendants

    def total_links(self, graph):
        """Calculate the total number of links in a graph"""
        n_links = np.sum(graph.graph == 1)
        return n_links

    def positive_test_strategy(self):
        descendant_links = np.zeros((self.n_graphs, self.n_actions))
        total_links = np.zeros(self.n_graphs)

        for i, graph in enumerate(self.graphs):
            total_links[i] = self.total_links(graph)
            for j, action in enumerate(self.actions):
                descendant_links[i, j] = self.descendant_links(graph, action)

        pts = np.max((descendant_links.T / total_links).T, axis=0)
        pts = pts / np.sum(pts)
        return pts
