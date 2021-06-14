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
        self.dict_arms_count[idx] += 1
        if not init_phase:
            self._resample_dict()
        self._update_embedding()
        self._update_mean_variances()
        self._update_beta()
        if self.verbose:
            tictoc.toc('[--] update completed in')
        
    def _resample_dict(self):
        resample_dict = self.random_state.rand(self.X.shape[0]) < (self.variances * self.pulled_arms_count * self.qbar)
        assert resample_dict.sum() > 0
        self.dict_arms_count = np.zeros(self.X.shape[0])
        self.dict_arms_count[resample_dict] = 1
        self.active_set = self.X[self.dict_arms_count != 0, :]
        self.m = self.active_set.shape[0]
    

    def _extend_search_space(self, leaf_set):
        extended = False
        for node in leaf_set:
            if tuple(node.x) not in self.node2idx:
                self.node2idx[tuple(node.x)] = self.num_nodes
                self.num_nodes += 1
                self.means = np.append(self.means, 0)
                self.variances = np.append(self.variances, 0)
                self.pulled_arms_count = np.append(self.pulled_arms_count, 0)
                self.Y = np.append(self.Y, 0).reshape(-1)
                self.X = np.append(self.X, node.x).reshape(-1,self.d)
                self.X_norms = diagonal_dot(self.X, self.dot)
                extended = True
        if extended:
            self._update_embedding(only_extend=True)




    def initialize(self, target_fun, search_space, N : int = 2, budget : int = 1000):
        assert budget > 0
        root = PartitionTreeNode(search_space, N, None, expansion_procedure=self.expand_fun)
        self._init_maps(root.x, search_space.shape[0])
        ne = 0 # number of function evaluation performed
        Vh_root = self._compute_V(1)
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
            root_variance = np.sqrt(self.variances[0]) 
            if self.beta * root_variance <= Vh_root:
                #expand and return children list
                self.father_ucbs[0] = self.means[0] + self.beta*root_variance + Vh_root
                leaf_set = root.expand_node()
                self._extend_search_space(leaf_set)
                self._update_embedding(only_extend=True)
                self._update_mean_variances()
                self._update_beta()
                return leaf_set
        raise Exception("Initialization not completed! Try to increase budget!")



    #def _update_embedding(self, only_extend = False):
    #    if only_extend:
    #        self.K_km = self.dot(self.X, self.active_set)
    #        self.X_embedded = self.K_km.dot(self.U_thin * self.S_thin_inv_sqrt.T)
    #        self.X_norms_embedded = np.square(np.linalg.norm(self.X_embedded, axis = 1))
    #    else:
    #        self.K_mm = self.dot(self.active_set)
    #        self.K_km = self.dot(self.X, self.active_set)
    #        #print("K_km: ",K_km)
    #        try:
    #            U, S, _ = svd(self.K_mm)
    #        except LinAlgError:
    #            U, S, _ = svd(self.K_mm, lapack_driver='gesvd')
#
    #        self.U_thin, self.S_thin_inv_sqrt = stable_invert_root(U, S)
    #        #self.U_thin, self.S_thin_inv_sqrt = U_thin, S_thin_inv_sqrt
    #        
    #        self.X_embedded = self.K_km.dot(self.U_thin * self.S_thin_inv_sqrt.T)
    #        self.X_norms_embedded = np.square(np.linalg.norm(self.X_embedded, axis = 1))
    #    # print("[--] X norm embedded: ",self.X_norms_embedded)
    #        self.m = len(self.S_thin_inv_sqrt)
#
    #def _update_variances(self, idx_to_update=None):
    #    if idx_to_update is None:
    #        temp = solve_triangular(self.R,
    #                                (self.X_embedded.dot(self.Q)).T,
    #                                overwrite_b=True,
    #                                check_finite=False).T
    #        temp *= self.X_embedded
    #        self.variances = (self.X_norms - self.X_norms_embedded) / self.lam + np.sum(temp, axis=1)
    #    else:
    #        temp = solve_triangular(self.R,
    #                                (self.X_embedded[idx_to_update, :].dot(self.Q)).T,
    #                                overwrite_b=True,
    #                                check_finite=False).T
    #        temp *= self.X_embedded[idx_to_update, :]
    #        self.variances[idx_to_update] = (
    #                (self.X_norms[idx_to_update] - self.X_norms_embedded[idx_to_update]) / self.lam
    #                + np.sum(temp, axis=1)
    #        )
    #    
    #    assert np.all(self.variances >= 0.)
    #    assert np.all(np.isfinite(self.variances))
#
#
    #def _update_mean_variances(self, idx_to_update=None):
    #    pulled_arms_matrix = self.pulled_arms_matrix
    #    reweight_counts_vec = np.sqrt(self.pulled_arms_count[self.pulled_arms_count != 0].reshape(-1, 1))
#
    #    self.A = ((pulled_arms_matrix * reweight_counts_vec).T.dot(pulled_arms_matrix * reweight_counts_vec)
    #              + self.lam * np.eye(self.m))
#
    #    self.Q, self.R = qr(self.A)
#
    #    self.w = solve_triangular(self.R, self.Q.T.dot(pulled_arms_matrix.T.dot(self.Y[self.pulled_arms_count != 0])))
#
    #    self.means = self.X_embedded.dot(self.w)
    #    assert np.all(np.isfinite(self.means))
#
    #    self._update_variances(idx_to_update)
    #    self.conf_intervals = self.beta * np.sqrt(self.variances)
#
    #def _update_beta(self):
    #    self.logdet = (self.variances * self.pulled_arms_count).sum() * np.log(self.pulled_arms_count.sum())
    #    self.beta = np.sqrt(self.lam) * self.fnorm + np.sqrt(self.noise_variance * (self.logdet + np.log(1 / self.delta)))
    #    assert np.isfinite(self.beta)
#
#
    #def initialize(self, target_fun):
#
#
    #def initialize(self, target_fun, root, budget):
    #    ne = 0
    #    self.node2idx[tuple(root.x)] = self.num_nodes
    #    self.num_nodes += 1
    #    self.X[0] = root.x 
    #    self.X_norms = diagonal_dot(self.X, self.dot)
    #    self.pulled_arms_count = np.zeros(self.X.shape[0])
    #   # self.active_set = self.X.reshape((-1,self.d))
    #    self.dict_arms_count = np.zeros(1)
    #    self.active_set = np.zeros((1, self.d))
    #    self.means = np.zeros(1)
    #    self.variances = np.zeros(1)
    #   # self.conf_intervals = np.zeros(1)
    #    while ne < budget:
    #        yt = target_fun(root.x)
    #        ne += 1 
    #     #   print(yt)
    #        self.Y[0] += yt
    #        self.pulled_arms_count[0] += 1
    #        self.dict_arms_count[0] += 1
    #        self.active_set = self.X[self.dict_arms_count != 0, :]
    #        Vh = self._compute_V(root.level+1)
    #        self._update_embedding()
    #        self._update_mean_variances()
    #        self._update_beta()
    #        # print("[--] beta*sigma: {}\tVh: {}".format(self.conf_intervals[0], Vh))
    #        if self.conf_intervals[0] <= Vh:
    #            self.father_ucbs[root.index] = self.means[0] + self.conf_intervals[0] + Vh
    #            leaf_set = root.expand_node()
    #            self._extend_search_space(leaf_set)
    #            self._update_embedding(only_extend=True)
    #            self._update_mean_variances()
    #            self._update_beta()
    #            avg_root_reward = self.Y[0]/ self.pulled_arms_count[0]   
    #            #self._init_leaf_mean_variances(tonp([node.x for node in leaf_set]).reshape(-1,self.d))
    #            return leaf_set, ne, avg_root_reward
    #        self.leaf_set_size.append(1)
    #    avg_root_reward = self.Y[0]/ self.pulled_arms_count[0]
    #    return root.expand_node(), ne, avg_root_reward
#
    #def _extend_search_space(self, leaf_set):
    #    extended = False
    #    for node in leaf_set:
    #        if tuple(node.x) not in self.node2idx:
    #            self.node2idx[tuple(node.x)] = self.num_nodes
    #            self.num_nodes += 1
    #            self.means = np.append(self.means, 0)
    #            self.variances = np.append(self.variances, 0)
    #            self.pulled_arms_count = np.append(self.pulled_arms_count, 0)
    #            self.Y = np.append(self.Y, 0).reshape(-1)
    #            self.X = np.append(self.X, node.x).reshape(-1,self.d)
    #            self.X_norms = diagonal_dot(self.X, self.dot)
    #            extended = True
    #    if extended:
    #        self._update_embedding(only_extend=True)
#
#
    #def _select_node(self, leaf_set):
    #    assert self.conf_intervals.shape[0] == self.variances.shape[0]
    #    I = []
    #    node_ucbs = []
    #    for i in range(0, len(leaf_set)):
    #        node = leaf_set[i]
    #        node_idx = self.node2idx[tuple(node.x)]
    #        father_idx = self.node2idx[tuple(node.father.x)]
    #        node_ucb = self.means[node_idx] + self.conf_intervals[node_idx]
    #        father_ucb = self.means[father_idx] + self.conf_intervals[father_idx] + self._compute_V(node.level)
    #        node_ucbs.append(node_ucb)
    #        Vh = self._compute_V(node.level + 1)
    #        I.append(np.min([node_ucb, father_ucb]) + Vh)
    #    selected_idx = np.argmax(I)
    #    selected_idx_node= self.node2idx[tuple(leaf_set[selected_idx].x)]
#
    #    Vh = self._compute_V(leaf_set[selected_idx].level + 1)
    #    return selected_idx, self.conf_intervals[selected_idx_node], Vh
#
#
    #def _resample_dict(self):
    #    resample_dict = self.random_state.rand(self.X.shape[0]) < (self.variances * self.pulled_arms_count * self.qbar)
    #    assert resample_dict.sum() > 0
    #    self.dict_arms_count = np.zeros(self.X.shape[0])
    #    self.dict_arms_count[resample_dict] = 1
    #    self.active_set = self.X[self.dict_arms_count != 0, :]
    #    self.m = self.active_set.shape[0]
    #    #print("Active set shape: {}".format(self.active_set.shape))
#
    #
#
    #def run(self, target_fun, search_space, N, budget, real_best, out_dir ="./", h_max = None):
    #    assert N > 1 and budget > 0
    #    t = 0
    #    if h_max is None:
    #        h_max = int(np.log(budget))
#
    #    n_eval = 0
    #    last_selected = None
    #    root = PartitionTreeNode(search_space, N, None, 0, 0, self.expand_fun)
    #    self.init_time = time.time()
    #    self.leaf_set_size = [1]
    #    ne = 0
    #    leaf_set, _, yroot = self.initialize(target_fun, root, budget)
    #    self.init_time = time.time() - self.init_time
    #    current_best = (root.x, yroot)
    #    #self.reg_analyzer.add_regret(current_best[1], current_best[1], real_best)
    #   # self.iteration_time = []
    #   # self.active_set_size = [self.embedding_size]
    #   # self.leaf_set_size.append(len(leaf_set))
    #   # self.cumulative_expansion_time = []
    #   # self.cumulative_evaluation_time = []
    #   # self.time_over_budget = []
    #   # self.cumulative_regret = []
#
    #    leaf_indices = list(np.concatenate([[self.node2idx[tuple(node.x)] for node in leaf_set], [self.node2idx[tuple(root.x)]]]))
    #    #time_over = time.time()
    #    #tt = time.time()
    #    while ne < budget:
    #        it_time = time.time()
    #        idx, conf_interval, Vh = self._select_node(leaf_set)
    #        last_selected = leaf_set[idx]
    #        node = leaf_set[idx]
    #        #print("node selected: {}".format(list(node.partition)))
    #     #   print("[AdaBKB] ne: {}/{}\tSelected index: {}\tx: {}\tConfidence interval: {}\tVh: {}\th: {}/{}".format(ne, budget, idx, node.x, conf_interval, Vh, node.level,h_max))
    #     #   print("[AdaBKB] lf size: {}\t as size: {}".format(len(leaf_set), self.embedding_size))
    #        if conf_interval <= Vh and leaf_set[idx].level < h_max:
    #            children = leaf_set[idx].expand_node()
    #            leaf_set[idx] = children
    #            leaf_set = flatten_list(leaf_set)
    #            self._extend_search_space(children)
    #            leaf_indices = [self.node2idx[tuple(node.x)] for node in leaf_set]
    #            leaf_indices = list(np.concatenate([leaf_indices, list(set([self.node2idx[tuple(node.father.x)] for node in leaf_set]))]))
    #            self._update_mean_variances(tonp(leaf_indices).reshape(-1))
    #            # self._update_mean_variances(tonp([self.node2idx[tuple(node.x)] for node in leaf_set]).reshape(-1))
    #            #self._update_beta()
    #      #      self.cumulative_expansion_time.append(time.time() - it_time)
    #      #      self.cumulative_evaluation_time.append(0)
    #            
    #        else:
    #            yt = target_fun(leaf_set[idx].x)
    #       #     print("\t[AdaBKB] yt: {}".format(yt))
    #            self.pulled_arms_count[self.node2idx[tuple(node.x)]] +=1
    #            self.Y[self.node2idx[tuple(node.x)]] += yt
    #            avg_reward = self.Y[self.node2idx[tuple(node.x)]]/ self.pulled_arms_count[self.node2idx[tuple(node.x)]] 
    #            avg_best_reward = self.Y[self.node2idx[tuple(current_best[0])]]/ self.pulled_arms_count[self.node2idx[tuple(current_best[0])]]
    #            if  avg_reward > avg_best_reward:# and np.all(node.x != current_best[0]):
    #                current_best = (node.x, avg_reward)
    #            avg_best_reward = self.Y[self.node2idx[tuple(current_best[0])]]/ self.pulled_arms_count[self.node2idx[tuple(current_best[0])]]
    #            #self.dict_arms_count[self.node2idx[tuple(node.x)]] +=1
    #        #    self.reg_analyzer.add_regret(yt, avg_best_reward, real_best)
    #         #   self.cumulative_regret.append(np.abs(real_best - avg_reward))
    #            ne += 1
    #            self._resample_dict()
    #            self._update_embedding()
    #           # to_update_leaf = [self.node2idx[tuple(node.x)] for node in leaf_set]
    #           # to_update_father = [self.node2idx[tuple(node.father.x)] for node in leaf_set]
    #           # to_update = list(set(np.concatenate([to_update_leaf, to_update_father])))
    #            self._update_mean_variances(leaf_indices)#[self.node2idx[tuple(node.x)], self.node2idx[tuple(node.father.x)]])
    #            self._update_beta()
    #          #  self.cumulative_expansion_time.append(0)
    #          #  self.cumulative_evaluation_time.append(time.time() - it_time)
    #          #  self.time_over_budget.append(time.time() - time_over)
    #          #  time_over = time.time()
    #          #  avgreg = self.reg_analyzer.get_average_regret()[n_eval]
    #           # print("[AdaBKB] Average regret: {}".format(avgreg))
    #          #  with open(out_dir+"adakbk_trace.log", "a") as f:
    #          #      f.write("{},{}\n".format(leaf_set[idx].x,yt))
    #          #  with open(out_dir + "adabkb_out.log", "a") as f:
    #          #      f.write("{},".format(avg_reward))
    #            #if self.early_stopping != None and self.early_stopping(t, avgreg):
    #            #    break                    
    #        
    #            n_eval += 1
    #       # self.iteration_time.append(time.time() - it_time)
    #       # self.active_set_size.append(self.embedding_size)
    #       # self.leaf_set_size.append(len(leaf_set))
    #        t += 1
    #       # print("[AdaBKB] Time spent: {}s".format(time.time() - tt))
    #    #self.cumulative_regret = np.cumsum(self.cumulative_regret)
    #    self.t = t
    #    return leaf_set, last_selected, current_best[0]
#
#