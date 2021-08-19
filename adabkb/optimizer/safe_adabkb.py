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

    def _remove_unsafe_partitions_(self):
        old_leaf_set_size = len(self.leaf_set)
        unsafe_indices = self.I < self.j_min
        self.I = np.delete(self.I, unsafe_indices, 0)
        self.leaf_set = np.delete(self.leaf_set, unsafe_indices, 0)
        assert len(self.leaf_set) <= old_leaf_set_size

    def initialize(self, target_fun, search_space, N : int = 2, budget : int = 1000, h_max: int = 10):
        super().initialize(target_fun, search_space, N, budget, h_max)
        #self._remove_unsafe_partitions_()
        self._evaluable_centroid_()

    def _update_model(self, indices, ys, init_phase : bool = False):
        if self.verbose:
            tictoc = TicToc()
            tictoc.tic()
        for i in range(len(indices)):
            self.Y[indices[i]] += ys[i]
            self.pulled_arms_count[indices[i]] += 1
        if not init_phase:
            self._resample_dict()
        self._update_embedding()
        if init_phase:
            self._update_mean_variances(idx_to_update=[0])
        else:
            to_upd = np.concatenate([
                [self._get_node_idx(node) for node in self.leaf_set],
                list(set([self._get_node_idx(node.father) for node in self.leaf_set]))
            ])
            self._update_mean_variances(idx_to_update=to_upd)
        self._update_beta()
        if not init_phase:
            self._compute_index()
        if self.verbose:
            tictoc.toc('[--] update completed in')
        
    def update_model(self, idx, yt):
        self._update_model([idx], [yt], False)

    def _resample_dict(self):
        resample_dict = self.random_state.rand(self.X.shape[0]) < (self.variances * self.pulled_arms_count * self.qbar)
        assert resample_dict.sum() > 0
        self.dict_arms_count = np.zeros(self.X.shape[0])
        self.dict_arms_count[resample_dict] = 1
        self.active_set = self.X[self.dict_arms_count != 0, :]
        self.m = self.active_set.shape[0]

    def _compute_index(self, lfset_indices = None):
        if lfset_indices is None:
            lfset_indices = list(range(len(self.leaf_set)))

        for i in lfset_indices:
            node = self.leaf_set[i]
            self.I[i] = self.means[self.node2idx[tuple(node.x)]] + self.beta * np.sqrt(self.variances[self.node2idx[tuple(node.x)]]) + self._compute_V(node.level)

    def _evaluable_centroid_(self):
        self.S_idx = [i for i in range(len(self.leaf_set))]
        #if self.means[self.node2idx[tuple(self.leaf_set[i].x)]] - self.beta * np.sqrt(self.variances[self.node2idx[tuple(self.leaf_set[i].x)]]) + self._compute_V(self.leaf_set[i].level) >= self.j_min 
        #or 
        #(self.leaf_set[i].level < self.h_max or self.h_max is None))
        #]
        #and self.means[self.node2idx[tuple(self.leaf_set[i].x)]] + self.beta * np.sqrt(self.variances[self.node2idx[tuple(self.leaf_set[i].x)]] >= self.best_lcb[1]]
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

    def _update_best(self, node_idx, leaf_idx, Vh):
        if self._centroid_accurate(node_idx, Vh) and self.pulled_arms_count[node_idx] > 0:
            lcb = self.means[node_idx] - self.beta * np.sqrt(self.variances[node_idx])
            avg_rew = self.Y[node_idx] / self.pulled_arms_count[node_idx]
            if avg_rew > self.current_best[1] or self.leaf_set[leaf_idx] == self.current_best[0]:
                self.best_lcb = (self.leaf_set[leaf_idx].x, lcb)
                self.current_best = (self.leaf_set[leaf_idx], avg_rew)

    def step(self):
        while True:
            leaf_idx, Vh = self._select_node()
            node_idx =self.node2idx[tuple(self.leaf_set[leaf_idx].x)]
            self._update_best(node_idx, leaf_idx, Vh)
            if (self._centroid_accurate(node_idx, Vh) or self._unsafe_centroid(node_idx, Vh)) and self.leaf_set[leaf_idx].level <= self.h_max: # and self.leaf_set[leaf_idx].level < self.h_max:
                self._expand(leaf_idx, node_idx)
                self._evaluable_centroid_()
            elif  self._unsafe_centroid(node_idx, Vh) and self.I[leaf_idx] >= self.j_min: #self._centroid_accurate(node_idx, Vh) and
                print("[--] unsafe centroid on safe partition")
                print("[--] h: {}\tVh: {}".format(self.leaf_set[leaf_idx].level, Vh))
                #self.leaf_set = np.delete(self.leaf_set, leaf_idx, 0)
                #self.I = np.delete(self.I, leaf_idx, 0)
                print("[--] leaf set size: {}".format(len(self.leaf_set)))
                self._expand(leaf_idx, node_idx)
                print("[--] bef rem => leaf set size: {}".format(len(self.leaf_set)))
                self._remove_unsafe_partitions_()
                print("[--] after rem => leaf set size: {}".format(len(self.leaf_set)))
                #    dummy_y = self.means[node_idx] - self.beta*np.sqrt(self.variances[node_idx]) - Vh
            #    if self.verbose:
            #        print("[--] x: {}\tfake y: {}".format(self.leaf_set[leaf_idx].x, dummy_y))
            #    self.update_model(node_idx, dummy_y)
            #    self._evaluable_centroid_()
            #elif self._unsafe_centroid(node_idx, Vh):
            #    self.leaf_set = np.delete(self.leaf_set, leaf_idx, 0)
            #    self.I = np.delete(self.I, leaf_idx, 0)  
                self._evaluable_centroid_()          
            else:
                print("[--] lcb: {}\t j_min: {}".format(self.means[node_idx] - self.beta * np.sqrt(self.variances[node_idx]), self.j_min))
                return self.leaf_set[leaf_idx].x, node_idx
