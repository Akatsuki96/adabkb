from scipy.linalg import solve_triangular, svd, qr, qr_update, LinAlgError
import numpy as np
from options import OptimizerOptions

from sklearn.gaussian_process.kernels import Kernel

import numpy as np
from scipy.linalg import solve_triangular, svd, qr, qr_update, LinAlgError
from utils import *
from options import OptimizerOptions

import itertools as it
import time

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel
from pytictoc import TicToc


class AdaBKB:
    def __init__(self, dot_fun: Kernel, options: OptimizerOptions = None):
        self.dot = dot_fun
        self.options = options
        self.node2idx = {}
        self.father_ucbs = {}
        #self.X = np.zeros((1, options.search_space))
        #self.Y = np.zeros(1)
        self.num_nodes = 0
        self.logdet = 1
        self.beta = 1
        #self.d = d

    @property
    def expand_fun(self):
        return self.options.expand_fun

    @property
    def qbar(self):
        return self.options.qbar

    @property
    def lam(self):
        return self.options.lam

    @property
    def random_state(self):
        return self.options.random_state

    @property
    def fnorm(self):
        return self.options.fnorm

    @property
    def noise_variance(self):
        return self.options.noise_var

    @property
    def delta(self):
        return self.options.delta

    @property
    def pulled_arms_matrix(self):
        return self.X_embedded[self.pulled_arms_count != 0, :]

    @property
    def unique_arms_pulled(self):
        return np.count_nonzero(self.pulled_arms_count)

    @property
    def dict_size(self):
        return self.dict_arms_count.sum()

    @property
    def dict_size_unique(self):
        return np.count_nonzero(self.dict_arms_count)

    @property
    def embedding_size(self):
        return self.m

    @property
    def early_stopping(self):
        return self.options.early_stopping

    @property
    def v_1(self):
        return self.options.v_1
    
    @property
    def rho(self):
        return self.options.rho

    @property
    def gfun(self):
        return self.options.gfun

    @property
    def verbose(self):
        return self.options.verbose


    def _compute_V(self, h):
        """Given a level \\(h \\geq 0\\), it compute \\(V_h\\) s.t. \\(\\forall i\\)
        \\[\\sup\\limits_{x, x^{\\prime} \\in X_{h,i}} |f(x) - f(x^\\prime)| \\leq V_h\\]

        Returns
        -------
        Vh : float
            upper bound on the function variation in cell at level h
        """
        return self.gfun(self.v_1 * (self.rho**h) ) * self.fnorm
        
    def _init_maps(self, xroot, d: int = 1):
        assert d > 0
        self.X = np.zeros((1, d))
        self.Y = np.zeros(1)
        self.X[0] = xroot
        self.node2idx[tuple(xroot)] = self.num_nodes
        self.num_nodes += 1
        self.d = d
        self.X_norms = diagonal_dot(self.X, self.dot)
        self.pulled_arms_count = np.zeros(self.X.shape[0])
        self.dict_arms_count = np.zeros(1)
        self.active_set = self.X
        self.means = np.zeros(1)
        self.variances = np.zeros(1)
    
    def _evaluate_model(self, Xstar):
        #K_sm = self.dot(self.X, self.active_set)
        K_sm = self.dot(Xstar, self.active_set)
        Xstar_embedded = K_sm.dot(self.U_thin * self.S_thin_inv_sqrt.T)
        evaluated_means = Xstar_embedded.dot(self.w)
        return evaluated_means

    def _update_embedding(self, only_extend = False):
        if only_extend:
            self.K_km = self.dot(self.X, self.active_set)
            self.X_embedded = self.K_km.dot(self.U_thin * self.S_thin_inv_sqrt.T)
            self.X_norms_embedded = np.square(np.linalg.norm(self.X_embedded, axis = 1))
        else:
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

        self.means = self.X_embedded.dot(self.w)
        assert np.all(np.isfinite(self.means))

        self._update_variances(idx_to_update)

    def _update_beta(self):
        self.logdet = (self.variances * self.pulled_arms_count).sum() * np.log(self.pulled_arms_count.sum())
        self.beta = np.sqrt(self.lam) * self.fnorm + np.sqrt(self.noise_variance * (self.logdet + np.log(1 / self.delta)))
        assert np.isfinite(self.beta)


    def _update_model(self, idx, yt, init_phase : bool = False):
        tictoc = TicToc()
        tictoc.tic()
        self.Y[idx] += yt
        self.pulled_arms_count[idx] += 1
        if not init_phase:
            self._resample_dict()
        
        self._update_embedding()
        self._update_mean_variances()
        self._update_beta()
        if not init_phase:
            self._compute_index()
        if self.verbose:
            tictoc.toc('[--] update completed in')
        
    def update_model(self, idx, yt):
        self._update_model(idx, yt, False)

    def _resample_dict(self):
        resample_dict = self.random_state.rand(self.X.shape[0]) < (self.variances * self.pulled_arms_count * self.qbar)
        assert resample_dict.sum() > 0
        self.dict_arms_count = np.zeros(self.X.shape[0])
        self.dict_arms_count[resample_dict] = 1
        self.active_set = self.X[self.dict_arms_count != 0, :]
        self.m = self.active_set.shape[0]
    

    def _extend_search_space(self, leaf_set):
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
            self._update_embedding(only_extend=True)
        return new_x



    def initialize(self, target_fun, search_space, N : int = 2, budget : int = 1000, h_max : int = 100):
        assert budget > 0
        root = PartitionTreeNode(search_space, N, None, expansion_procedure=self.expand_fun)
        self._init_maps(root.x, search_space.shape[0])
        self.h_max = h_max
        ne = 0 # number of function evaluation performed
        Vh_root = self._compute_V(0)
        t = TicToc()
        t.tic()
        while ne < budget:
            yt = target_fun(root.x)
            if self.verbose:
                t.toc('[--] f(x_root) evaluated [n. %d] in' % ne)
            ne += 1
            self._update_model(0, yt, init_phase=True)
            if self.verbose:
                print("[--] mu[root] : {}\t beta*sigma[root]: {}\t V_0: {}\n".format(self.means[0], self.beta*np.sqrt(self.variances[0]), Vh_root))
            root_std = np.sqrt(self.variances[0]) 
            if self.beta * root_std <= Vh_root:
                self.leaf_set = root.expand_node()
                self.I = np.zeros(len(self.leaf_set))
                self._extend_search_space(self.leaf_set)
                self._update_embedding(only_extend=True)
                self._update_mean_variances()
                self._update_beta()
                self._compute_index()
                return self.leaf_set
        raise Exception("Initialization not completed! You should increase budget!")

    def _compute_index(self, lfset_indices = None):
        if lfset_indices is None:
            lfset_indices = list(range(len(self.leaf_set)))

        for i in lfset_indices:
            node = self.leaf_set[i]
            self.I[i] = np.min([
                            self.means[self.node2idx[tuple(node.x)]] + np.sqrt(self.variances[self.node2idx[tuple(node.x)]]),
                            self.means[self.node2idx[tuple(node.father.x)]] + np.sqrt(self.variances[self.node2idx[tuple(node.father.x)]]) + self._compute_V(node.father.level) 
                        ]) + self._compute_V(node.level)

    def _select_node(self):
        selected_idx = np.argmax(self.I)
        #selected_idx_node= self.node2idx[tuple(self.leaf_set[selected_idx].x)]

        Vh = self._compute_V(self.leaf_set[selected_idx].level)
        return selected_idx, Vh

    def step(self):
        while True:
            leaf_idx, Vh = self._select_node()
            node_idx =self.node2idx[tuple(self.leaf_set[leaf_idx].x)]
            if np.sqrt(self.variances[node_idx]) * self.beta <= Vh and self.leaf_set[leaf_idx].level < self.h_max:
                new_nodes = self.leaf_set[leaf_idx].expand_node()
                self.leaf_set = np.delete(self.leaf_set, leaf_idx, 0)
                self.I = np.delete(self.I, leaf_idx, 0)
                self.leaf_set = np.concatenate([self.leaf_set, new_nodes])
                self.I = np.concatenate([self.I, np.zeros(len(new_nodes))])
                assert len(self.I) == len(self.leaf_set) 
                new_x = np.array(self._extend_search_space(new_nodes)).reshape(-1, self.X.shape[1])
                new_means = self._evaluate_model(new_x)
                new_node_idx = [self.node2idx[tuple(x)] for x in new_x]
                for i in range(len(new_means)):
                    self.means[self.node2idx[tuple(new_x[i])]] = new_means[i]
                self._update_variances(new_node_idx)
                self._compute_index(list(range(len(self.leaf_set) - len(new_nodes), len(self.leaf_set))))
            else:
                return self.leaf_set[leaf_idx].x, node_idx
