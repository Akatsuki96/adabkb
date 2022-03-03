import numpy as np
<<<<<<< HEAD

from adabkb.optimizer import AdaptiveOptimizer
from adabkb.surrogate import BKB

from adabkb.partition_tree import PartitionTreeNode

=======
from pytictoc import TicToc

from adabkb.optimizer import AbsOptimizer

from scipy.linalg import solve_triangular, svd, qr, LinAlgError, qr_update
from adabkb.utils import diagonal_dot, stable_invert_root, PartitionTreeNode

from adabkb.options import OptimizerOptions

class AdaBKB(AbsOptimizer):
>>>>>>> a82973072a828de84640becd1014a956bae86e78

class AdaBKB(AdaptiveOptimizer):
        
    def __init__(self, search_space, options):
        super().__init__(options)
        self.model = BKB(**options.model_options)
        root = PartitionTreeNode(
            partition = search_space, 
            N = self.options.N, 
            father = None, 
            expansion_procedure = self.options.expand_fun
        )
        self.num_eval = 0
        self.leaf_set = np.array([root]) #search_space
        self.node_idx = np.zeros(1, dtype=int)
        self.best_lcb = (None, -np.inf, -1) # (best_x, best_lcb)

        self.I = np.zeros(1)
        self.Vh = np.array([self.compute_Vh(i) for i in range(self.h_max+1)], dtype=float)
        self.register_nodes([root])
        self.model.initialize(root.x)
   
    @property
    def g(self):
        return self.model.kernel.confidence_function
   
    @property
    def ucb(self):
        return self.model.ucbs
    
    @property
<<<<<<< HEAD
    def lcb(self):
        return self.model.lcbs
   
    @property
    def means(self):
        return self.model.means
    
    @property
    def stds(self):
        return np.sqrt(self.model.variances)
    
    @property
    def beta(self):
        return self.model.beta
   
    @property
    def h_max(self):
        return self.options.h_max
   
    def compute_Vh(self, level):
        return self.g(self.options.v_1 * pow(self.options.rho, level))

    def __can_be_expanded(self, node_idx, level):
  #      print(self.beta * self.stds[node_idx], self.Vh[level])
        return self.beta * self.stds[node_idx] <= self.Vh[level] and level < self.h_max

    def __select_candidate(self):
        return np.argmax(self.I)

    def __expand_leaf(self, node, leaf_idx, node_idx):
        children = node.expand_node()
        zeros = np.zeros(len(children))
        self.leaf_set = np.concatenate((np.delete(self.leaf_set, leaf_idx), children))
        self.I = np.concatenate((np.delete(self.I, leaf_idx), zeros))
        new_nodes, nodes_idx = self.register_nodes(children)
        self.model.extend_arm_set(new_nodes)
        self.node_idx = np.concatenate((np.delete(self.node_idx, leaf_idx), nodes_idx))
        new_leaf_idx = list(range(self.leaf_set.shape[0] - len(children) , self.leaf_set.shape[0]))
        self.__update_I(new_leaf_idx)
    #    self.__prune_leafset(np.asarray(new_leaf_idx))
   #     print("-----------------------------------------------------")
=======
    def sigma(self):
        return self.options.sigma

    def __init__(self, options: OptimizerOptions = None):
        super().__init__(options)
        self.tau = 0 
        self.Q, self.R = None, None

    def _evaluate_model(self, Xstar):
        K_sm = self.dot(Xstar, self.active_set)
        Xstar_embedded = K_sm.dot(self.U_thin * self.S_thin_inv_sqrt.T)
        return Xstar_embedded.dot(self.w)

    def _expand_embedding(self):
        self.K_km = self.dot(self.X, self.active_set)
        self.X_embedded = self.K_km.dot(self.U_thin * self.S_thin_inv_sqrt.T)
        self.X_norms_embedded = np.square(np.linalg.norm(self.X_embedded, axis = 1))
>>>>>>> a82973072a828de84640becd1014a956bae86e78

    
    def __prune_leafset(self, leaf_idx):
        
<<<<<<< HEAD
        
        node_idx = self.node_idx[leaf_idx]
        levels = np.asarray([node.level for node in self.leaf_set[leaf_idx]])

        subopt_partitions = (self.I[leaf_idx] < self.best_lcb[1]) & (self.node_idx[leaf_idx] != self.best_lcb[2])
        idx_to_erase = leaf_idx[subopt_partitions]
#        idx_to_erase = idx_to_erase[self.node_idx[subopt_partitions] != self.best_lcb[2]]
=======
        self.X_embedded = self.K_km.dot(self.U_thin * self.S_thin_inv_sqrt.T)
        self.X_norms_embedded = np.square(np.linalg.norm(self.X_embedded, axis = 1))
        self.m = len(self.S_thin_inv_sqrt)

    def _update_variances(self, idx_to_update=None):
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
        
        for i in range(len(indices)):
            self.Y[indices[i]] += ys[i]
            self.pulled_arms_count[indices[i]] += 1
        if self.tau > 0:
            self._resample_dict()
        self._update_embedding()
        if len(self.leaf_set) == 1 and self.leaf_set[0].level == 0:
            self._update_mean_variances(idx_to_update=[0])
        else:
            to_upd = np.concatenate(
                [
                    [self._get_node_idx(node) for node in self.leaf_set],
                    np.unique([self._get_node_idx(node.father) for node in self.leaf_set]),
                ]
            )
            self._update_mean_variances( idx_to_update=to_upd)
        self._update_beta()
        if self.tau > 0:
            self._compute_index()
>>>>>>> a82973072a828de84640becd1014a956bae86e78
        
        
        print("[PRUNING] IDX: {}".format(leaf_idx))   
        print("[--] node_idx: {}".format(self.node_idx[leaf_idx]))
        print("[--] subopt: {}".format(subopt_partitions))
        print("[--] best lcb: {}".format(self.best_lcb))
     #   print("[--] SUBOPT: {}".format(subopt_partitions))
        print("[--] idx to erase: {}".format(idx_to_erase)) 
        
        self.leaf_set = np.delete(self.leaf_set, idx_to_erase)
        self.I = np.delete(self.I, idx_to_erase)
        self.node_idx = np.delete(self.node_idx, idx_to_erase)
         
    def step(self):
        while True:
            leaf_idx = self.__select_candidate()
            node = self.leaf_set[leaf_idx]
            node_idx = self.get_node_idx(node)
            print("[!!] SELECTED:  node: {}\tleaf: {}".format(node_idx, leaf_idx))
            if self.__can_be_expanded(node_idx, node.level) and (node.level > 0 or self.num_eval > 0):
                print("-------------------- EXPAND ----------------")
                self.__expand_leaf(node, leaf_idx, node_idx)
            else:
                # Evaluation step
                print("-------------------- EVAL ----------------")
                return node, leaf_idx

    def __update_I(self, leaf_indices):
        father_idx = [] 
        levels = []
        for node in self.leaf_set[leaf_indices]:
            father_idx.append(self.get_node_idx(node.father))
            levels.append(node.level)


        father_idx, levels = np.asarray(father_idx), np.asarray(levels)
      
        self.I[leaf_indices] = np.min([self.model.ucbs[self.node_idx[leaf_indices]], self.model.ucbs[father_idx] + self.Vh[levels - 1]], axis=0) + self.Vh[levels]


<<<<<<< HEAD
    def __update_best_lcb(self, xs, node_indices, leaf_indices):
        for i in range(node_indices.shape[0]):
            if self.best_lcb[0] is None or node_indices[i] == self.best_lcb[2] or self.best_lcb[1] < self.lcb[node_indices[i]]:
                self.best_lcb = (xs[i], self.lcb[node_indices[i]], node_indices[i])   
    
    def update(self, leaf_idx, xs, ys):
        xs = np.asarray(xs)
        idx = np.asarray([self.node_idx[i] for i in leaf_idx])
        leaf_indices = np.asarray(leaf_idx)
        self.num_eval+=1
        if idx == 0 and self.leaf_set[0].level==0:
            self.model.full_update(idx, np.asarray([ys]))
            self.I[0] = self.model.ucbs[0] + self.Vh[0] 
        else:
            self.model.update_emb(idx, np.asarray([ys]))
            self.model.update_mean_variances(self.node_idx)
            self.__update_I(list(range(len(self.leaf_set))))
        self.__update_best_lcb(xs, idx, leaf_indices)
        if self.leaf_set[0].level > 0:
            self.__prune_leafset(np.array(list(range(self.leaf_set.shape[0]))))
=======
    def _prune_leafset(self):
        ucbs = np.array([self.means[self._get_node_idx(leaf)] + self.beta * np.sqrt(self.variances[self._get_node_idx(leaf)]) + self._compute_V(leaf.level) 
        for leaf in self.leaf_set 
        ])
        #print("UCBS: ", ucbs)
        #if self.best_lcb is not None:
        #    print("best LCB: ", self.best_lcb[1]) 
        lset_size = len(self.leaf_set)
        to_rem = []
        lset = []
        for i in range(len(self.leaf_set)):
            if self.best_lcb is None or ucbs[i] >= self.best_lcb[1] or np.all(self.best_lcb[0] == self.leaf_set[i].x):
                lset.append(self.leaf_set[i])
            else:
                to_rem.append(i)
       # print("[-] lset: {}".format(lset_size))
        self.leaf_set = lset #self.leaf_set[self.best_lcb is None or ucbs >= self.best_lcb[1]]
       # print("[-] lset: {}".format(len(self.leaf_set)))
        self.pruned = (lset_size - len(self.leaf_set))
        self.I = np.delete(self.I, to_rem, 0)
        self.cpruned += len(to_rem) #(lset_size - len(self.leaf_set))
        assert len(self.I) == len(self.leaf_set)
        if not self.leaf_set or (
            len(self.leaf_set) == 1 and self.leaf_set[0].level == self.h_max
        ):
            self.estop = True


    def _can_be_expanded(self, node_idx, h, Vh):
        return np.sqrt(self.variances[node_idx]) * self.beta <= Vh and h < self.h_max

    def _expand(self, leaf_idx, first_expansion = False):
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
           # self.means[new_node_idx] = new_means
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
            node_idx = self._get_node_idx(self.leaf_set[leaf_idx])
            self._update_best(node_idx, leaf_idx, Vh)
            if not self._can_be_expanded(
                node_idx, self.leaf_set[leaf_idx].level, Vh
            ):
                self.tau+=1
                return self.leaf_set[leaf_idx].x, node_idx

            self._expand(leaf_idx, len(self.leaf_set) == 1 and self.leaf_set[0].level == 0)
            self._prune_leafset()
            self.tau += 1
>>>>>>> a82973072a828de84640becd1014a956bae86e78
