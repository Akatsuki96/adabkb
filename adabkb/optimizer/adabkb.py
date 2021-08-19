from scipy.linalg import solve_triangular, svd, qr, qr_update, LinAlgError
import numpy as np
from adabkb.options import OptimizerOptions

from sklearn.gaussian_process.kernels import Kernel

from adabkb.utils import GreedyExpansion, diagonal_dot, stable_invert_root, PartitionTreeNode

from pytictoc import TicToc
import itertools as it
import time

from adabkb.optimizer import AbsOptimizer

class AdaBKB(AbsOptimizer):

    @property
    def pulled_arms_matrix(self):
        return self.X_embedded[self.pulled_arms_count != 0, :]

    @property
    def embedding_size(self):
        return self.m

    
    def _evaluate_model(self, Xstar):
        K_sm = self.dot(Xstar, self.active_set)
        Xstar_embedded = K_sm.dot(self.U_thin * self.S_thin_inv_sqrt.T)
        evaluated_means = Xstar_embedded.dot(self.w)
        return evaluated_means

    def _expand_embedding(self):
        self.K_km = self.dot(self.X, self.active_set)
        self.X_embedded = self.K_km.dot(self.U_thin * self.S_thin_inv_sqrt.T)
        self.X_norms_embedded = np.square(np.linalg.norm(self.X_embedded, axis = 1))

    def _update_embedding(self):
        self.K_mm = self.dot(self.active_set)
        self.K_km = self.dot(self.X, self.active_set)
        try:
            U, S, _ = svd(self.K_mm)
        except LinAlgError:
            U, S, _ = svd(self.K_mm, lapack_driver='gesvd')
        self.U_thin, self.S_thin_inv_sqrt = stable_invert_root(U, S)
        
        self.X_embedded = self.K_km.dot(self.U_thin * self.S_thin_inv_sqrt.T)
        self.X_norms_embedded = np.square(np.linalg.norm(self.X_embedded, axis = 1))
        self.m = len(self.S_thin_inv_sqrt)

    def _update_variances(self, idx_to_update=None):
        if idx_to_update is None:
            temp = solve_triangular(self.R,
                                    (self.X_embedded.dot(self.Q)).T,
                                    overwrite_b=True,
                                    check_finite=False).T
            temp *= self.X_embedded
            self.variances = (self.X_norms - self.X_norms_embedded) / self.lam + np.sum(temp, axis=1)
        else:
            temp = solve_triangular(self.R,
                                    (self.X_embedded[idx_to_update, :].dot(self.Q)).T,
                                    overwrite_b=True,
                                    check_finite=False).T
            temp *= self.X_embedded[idx_to_update, :]
            self.variances[idx_to_update] = (
                    (self.X_norms[idx_to_update] - self.X_norms_embedded[idx_to_update]) / self.lam
                    + np.sum(temp, axis=1)
            )
        assert np.all(self.variances >= 0.)
        assert np.all(np.isfinite(self.variances))


    def _update_mean_variances(self, idx_to_update=None):
        pulled_arms_matrix = self.pulled_arms_matrix
        reweight_counts_vec = np.sqrt(self.pulled_arms_count[self.pulled_arms_count != 0].reshape(-1, 1))

        self.A = ((pulled_arms_matrix * reweight_counts_vec).T.dot(pulled_arms_matrix * reweight_counts_vec)
                  + self.lam * np.eye(self.m))

        self.Q, self.R = qr(self.A)

        self.w = solve_triangular(self.R, self.Q.T.dot(pulled_arms_matrix.T.dot(self.Y[self.pulled_arms_count != 0])))
        self.means[idx_to_update] = self.X_embedded[idx_to_update, :].dot(self.w)
        assert np.all(np.isfinite(self.means))

        self._update_variances(idx_to_update)

    def _update_beta(self):
        self.logdet = (self.variances * self.pulled_arms_count).sum() * np.log(self.pulled_arms_count.sum())
        self.beta = np.sqrt(self.lam) * self.fnorm + np.sqrt(self.noise_variance * (self.logdet + np.log(1 / self.delta)))
        assert np.isfinite(self.beta)


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
        self._update_model([idx], [yt], len(self.leaf_set) == 1)

    def _resample_dict(self):
        resample_dict = self.random_state.rand(self.X.shape[0]) < (self.variances * self.pulled_arms_count * self.qbar)
        assert resample_dict.sum() > 0
        self.dict_arms_count = np.zeros(self.X.shape[0])
        self.dict_arms_count[resample_dict] = 1
        self.active_set = self.X[self.dict_arms_count != 0, :]
        self.m = self.active_set.shape[0]
    

    def _extend_search_space(self, leaf_set, first_expansion = False):
        extended = False
        new_x = []
        for node in leaf_set:
            if tuple(node.x) not in self.node2idx:
                new_x.append(node.x)
                self.node2idx[tuple(node.x)] = self.num_nodes
                self.num_nodes += 1
                self.means = np.append(self.means, 0)
                self.variances = np.append(self.variances, 0)
                self.pulled_arms_count = np.append(self.pulled_arms_count, 0)
                self.Y = np.append(self.Y, 0).reshape(-1)
                self.X = np.append(self.X, node.x).reshape(-1,self.d)
                extended = True
        if extended:
            self.X_norms = diagonal_dot(self.X, self.dot)
            if first_expansion:
                self._update_embedding()
            else:
                self._expand_embedding() #only_extend=True)
        return new_x



    def initialize(self, search_space, N : int = 2, h_max : int = 100):
        root = PartitionTreeNode(search_space, N, None, expansion_procedure=self.expand_fun)
        self._init_maps(root.x, search_space.shape[0])
        self.h_max = h_max
        self.leaf_set = [root]
        self.I = np.zeros(len(self.leaf_set))
        self.current_best = None
        #ne = 0 # number of function evaluation performed
        #Vh_root = self._compute_V(0)
        #t = TicToc()
        #t.tic()
        #while ne < budget:
        #    yt = target_fun(root.x)
        #    if self.verbose:
        #        t.toc('[--] f(x_root) evaluated [n. %d] in' % ne)
        #    ne += 1
        #    self._update_model([0], [yt], init_phase=True)
        #    if self.verbose:
        #        print("[--] mu[root] : {}\t beta*sigma[root]: {}\t V_0: {}\n".format(self.means[0], self.beta*np.sqrt(self.variances[0]), Vh_root))
        #    root_std = np.sqrt(self.variances[0])
        #    if self.beta * root_std <= Vh_root:
        #        avgrew = self.Y[0] / self.pulled_arms_count[0]
        #        self.best_lcb = (root.x, self.means[0] - self.beta*np.sqrt(self.variances[0]))
        #        self.current_best = (root, avgrew)
        #        self.leaf_set = root.expand_node()
        #        self.I = np.zeros(len(self.leaf_set))
        #        self._extend_search_space(self.leaf_set)
        #        to_upd = np.concatenate([
        #            [self._get_node_idx(node) for node in self.leaf_set],
        #            [0]
        #        ])
        #        self._update_mean_variances(to_upd)
        #        self._compute_index()
        #        return self.leaf_set
        #raise Exception("Initialization not completed! You should increase budget!")

    def _compute_index(self, lfset_indices = None):
        if lfset_indices is None:
            lfset_indices = list(range(len(self.leaf_set)))

        for i in lfset_indices:
            node = self.leaf_set[i]
            self.I[i] = np.min([
                            self.means[self.node2idx[tuple(node.x)]] + self.beta * np.sqrt(self.variances[self.node2idx[tuple(node.x)]]),
                            self.means[self.node2idx[tuple(node.father.x)]] + self.beta * np.sqrt(self.variances[self.node2idx[tuple(node.father.x)]]) + self._compute_V(node.father.level) 
                        ]) + self._compute_V(node.level)

    def _select_node(self):
        selected_idx = np.argmax(self.I)
        Vh = self._compute_V(self.leaf_set[selected_idx].level)
        return selected_idx, Vh


    def _can_be_expanded(self, node_idx, h, Vh):
        return np.sqrt(self.variances[node_idx]) * self.beta <= Vh and h < self.h_max

    def _expand(self, leaf_idx, node_idx, first_expansion = False):
        new_nodes = self.leaf_set[leaf_idx].expand_node()
        self.leaf_set = np.delete(self.leaf_set, leaf_idx, 0)
        self.I = np.delete(self.I, leaf_idx, 0)
        self.leaf_set = np.concatenate([self.leaf_set, new_nodes])
        self.I = np.concatenate([self.I, np.zeros(len(new_nodes))])
        assert len(self.I) == len(self.leaf_set) 
        new_x = np.array(self._extend_search_space(new_nodes, first_expansion)).reshape(-1, self.X.shape[1])
        if first_expansion:
            to_upd = np.concatenate([
                    [self._get_node_idx(node) for node in self.leaf_set],
                    [0]
                ])
            self._update_mean_variances(to_upd)
        else:
            new_means = self._evaluate_model(new_x)
            new_node_idx = [self.node2idx[tuple(x)] for x in new_x]
            for i in range(len(new_means)):
                self.means[self.node2idx[tuple(new_x[i])]] = new_means[i]
            self._update_variances(new_node_idx)
        self._compute_index(list(range(len(self.leaf_set) - len(new_nodes), len(self.leaf_set))))

    def _update_best(self, node_idx, leaf_idx, Vh):
        if np.sqrt(self.variances[node_idx]) * self.beta <= Vh and self.pulled_arms_count[node_idx] > 0:
            lcb = self.means[node_idx] - self.beta * np.sqrt(self.variances[node_idx])
            avg_rew = self.Y[node_idx] / self.pulled_arms_count[node_idx]
            if self.current_best is None or avg_rew > self.current_best[1] or self.leaf_set[leaf_idx] == self.current_best[0]:
                self.best_lcb = (self.leaf_set[leaf_idx].x, lcb)
                self.current_best = (self.leaf_set[leaf_idx], avg_rew)
        
    def step(self):
        while True:
            leaf_idx, Vh = self._select_node()
            node_idx =self.node2idx[tuple(self.leaf_set[leaf_idx].x)]
            self._update_best(node_idx, leaf_idx, Vh)
            if self._can_be_expanded(node_idx, self.leaf_set[leaf_idx].level, Vh ):
                self._expand(leaf_idx, node_idx, len(self.leaf_set) == 1)
            else:
                return self.leaf_set[leaf_idx].x, node_idx
