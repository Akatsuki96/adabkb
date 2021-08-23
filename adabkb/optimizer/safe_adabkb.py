from scipy.linalg import solve_triangular, svd, qr, qr_update, LinAlgError
import numpy as np
from adabkb.options import OptimizerOptions

from sklearn.gaussian_process.kernels import Kernel

from adabkb.utils import GreedyExpansion, diagonal_dot, stable_invert_root, PartitionTreeNode

from pytictoc import TicToc
import itertools as it
import time

from adabkb.optimizer import AdaBKB

class SafeAdaBKB(AdaBKB):

    def __init__(self, kernel, options):
        super().__init__(kernel, options)
        assert self.j_min is not None

    @property
    def j_min(self):
        return self.options.jmin


    def initialize(self, search_space, N : int = 2, h_max: int = 10):
        super().initialize(search_space, N, h_max)
        self._evaluable_centroid_()

    def _evaluable_centroid_(self):

        self.S_idx = []
        
        for i in range(len(self.leaf_set)):
            node_idx = self._get_node_idx(self.leaf_set[i])
            lcb = self.means[node_idx] - self.beta * np.sqrt(self.variances[node_idx]) + self._compute_V(self.leaf_set[i].level)
             #means[node_idx] + self.beta * np.sqrt(self.variances[node_idx]) + self._compute_V(self.leaf_set[i].level)
            if True or lcb >= self.j_min or\
                (self.leaf_set[i].level < self.h_max or self.h_max is None):
                self.S_idx.append(i)
        self.S = [self.leaf_set[i] for i in self.S_idx]

    def update_model(self, idx, yt):
        self._update_model([idx], [yt], False)
        self._evaluable_centroid_()

    def _select_node(self):
        selected_idx = np.argmax(self.I[self.S_idx])
        Vh = self._compute_V(self.S[selected_idx].level)
        return self.S_idx[selected_idx], Vh

    def _centroid_accurate(self, node_idx, Vh):
        return np.sqrt(self.variances[node_idx]) * self.beta <= Vh 

    def _unsafe_centroid(self, node_idx, Vh):
        lcb = (self.means[node_idx] - np.sqrt(self.variances[node_idx]) * self.beta)
        return lcb < self.j_min

    def step(self):

        while True:
            if len(self.leaf_set)==0:
                return self.current_best, None
            leaf_idx, Vh = self._select_node()
            node_idx =self.node2idx[tuple(self.leaf_set[leaf_idx].x)]
            self._update_best(node_idx, leaf_idx, Vh)
            if (self._centroid_accurate(node_idx, Vh) or self._unsafe_centroid(node_idx, Vh)) and self.leaf_set[leaf_idx].level <= self.h_max: # and self.leaf_set[leaf_idx].level < self.h_max:
                self._expand(leaf_idx, node_idx, len(self.leaf_set) == 1)
                self._evaluable_centroid_()
            elif  self._unsafe_centroid(node_idx, Vh): #self._centroid_accurate(node_idx, Vh) and
                self.leaf_set = np.delete(self.leaf_set, leaf_idx, 0)
                self.I = np.delete(self.I, leaf_idx, 0)
                self._evaluable_centroid_()          
            else:
                return self.leaf_set[leaf_idx].x, node_idx
