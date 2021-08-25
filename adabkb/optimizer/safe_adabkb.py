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


    def initialize(self, search_space, y0, N : int = 2, h_max: int = 10):
        super().initialize(search_space, N, h_max)
        self.update_model(0, y0, True)
        self._evaluable_centroid_()

    def _add_x(self, x):
        if tuple(x) not in self.node2idx:
            self._add_new_node(x)
            self.X_norms = diagonal_dot(self.X, self.dot)
            self._update_embedding()
            self._update_mean_variances([self.node2idx[tuple(x)]])
        return self.node2idx[tuple(x)]

    def _evaluable_centroid_(self):

        self.S_idx = []
        
        for i in range(len(self.leaf_set)):
            node_idx = self._get_node_idx(self.leaf_set[i])
            lcb = self.means[node_idx] - self.beta * np.sqrt(self.variances[node_idx]) + self._compute_V(self.leaf_set[i].level)
             #means[node_idx] + self.beta * np.sqrt(self.variances[node_idx]) + self._compute_V(self.leaf_set[i].level)
            #print("[-] i: {}\tlcb: {}".format(i,lcb))
            if lcb >= self.j_min: #or\
                #(self.leaf_set[i].level < self.h_max or self.h_max is None):
                self.S_idx.append(i)
        self.S = [self.leaf_set[i] for i in self.S_idx]

    def update_model(self, idx, yt, first=False):
        self._update_model([idx], [yt], first)
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

    def _predict(self, x):
        Xt_emb = (self.dot(x, self.active_set)).dot(self.U_thin * self.S_thin_inv_sqrt.T)
        norm_emb = np.square(np.linalg.norm(Xt_emb, axis = 1))
        pred_mean = Xt_emb.dot(self.w)
        tmp = solve_triangular(self.R,
                                    (Xt_emb.dot(self.Q)).T,
                                    overwrite_b=True,
                                    check_finite=False).T
        tmp *= Xt_emb
        #print("[--] norm emb: {}".format(norm_emb))
        #print("[--] norm: {}".format(diagonal_dot(x.reshape(-1,self.d), self.dot)))
        pred_var = (diagonal_dot(x.reshape(-1,self.d), self.dot) - norm_emb) / self.lam + np.sum(tmp, axis=1)
        return pred_mean, pred_var

    def _measure_lcb(self, x):
        mu, sig = self._predict(x)
       # print("[--] mu: {}\n[--] var: {}".format(mu, sig))
        return mu - self.beta * np.sqrt(sig)

    def search_safe_point(self, leaf_idx):
        xk = self.leaf_set[leaf_idx].x
        grad_scale = 1e-10
        tolerance = 1e-7
        lcb = self._measure_lcb(xk)
        #print("[--] xk: {}\tlcb: {}".format(xk, lcb))
        c = 1
        alpha = 1/c
        while lcb < self.j_min:
            Pk = np.eye(self.d)
            dir_grad = 0
            for i in range(self.d):
                dir_grad += ((self._measure_lcb(xk + grad_scale * Pk[:,i]) - self._measure_lcb(xk))/grad_scale) * Pk[:,i] 
            xk = xk + alpha * dir_grad 
            lcb = self._measure_lcb(xk)
            print("[--] xk: {}\t|grad|: {}\tlcb: {}\tj_min: {}".format(xk, np.linalg.norm(dir_grad), lcb, self.j_min))
            c+=1
            alpha = 1/c
      #      print("[--] xk: {}\tlcb: {}".format(xk, lcb))
        self.leaf_set[leaf_idx].x = xk
        print("[--] Found safe xk: {}".format(xk))
        return xk, self._add_x(xk)
        #raise Exception("Test")

    def step(self):

        while True:
            if len(self.I[self.S_idx])==0:
                return self.current_best, None
            leaf_idx, Vh = self._select_node()
            node_idx =self.node2idx[tuple(self.leaf_set[leaf_idx].x)]
            self._update_best(node_idx, leaf_idx, Vh)
            print("[--] I: {}\tVh: {}".format(self.I[leaf_idx], Vh))
            if self._centroid_accurate(node_idx, Vh) and self.leaf_set[leaf_idx].level <= self.h_max: 
                print("[--] expand")
                self._expand(leaf_idx, node_idx, len(self.leaf_set) == 1)
                self._evaluable_centroid_()
            elif self._unsafe_centroid(node_idx, Vh): #self._centroid_accurate(node_idx, Vh) and
                # partition identified by leaf_set[leaf_idx] is safe (i.e. exists x safe in leaf_set[leaf_idx]) 
                return self.search_safe_point(leaf_idx)
            else:
                return self.leaf_set[leaf_idx].x, node_idx
