import numpy as np


from adabkb.optimizer import AdaBKB
from adabkb.utils import SplitOnRepresenter, diagonal_dot, stable_invert_root, PartitionTreeNode
from scipy.linalg import solve_triangular, svd, qr, qr_update, LinAlgError


from cma import CMAEvolutionStrategy

class CMABKB(AdaBKB):


    def initialize(self, search_space, N, h_max : int = 100,\
         pop_size : int = 2,\
         sigma0 : float = 1.0,\
         max_gen : int = 5,\
         cmaes_seed: int = 42):
        super().initialize(search_space, N, h_max)
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.sigma0 = sigma0
        self.cmaes_seed= cmaes_seed
        self.d = search_space.shape[0]

    def explore_partition(self, node, leaf_idx, node_idx):
        opts = {
            'seed' : self.cmaes_seed,
            'popsize' : self.pop_size,
            'bounds' : [node.partition[:,0], node.partition[:,1]],
            'verb_disp' : 0
        }
        #def target(x):
        #    mu, sigma = self.eval_ucb(x)
        #    return -(mu + self.beta * sigma)
        #
        #
        cmaes = CMAEvolutionStrategy(node.partition.mean(axis=1), self.sigma0, opts)
        for t in range(self.max_gen):
            xt = np.array(cmaes.ask()).reshape(-1,node.partition.shape[0])
            mu, sigma = self.eval_ucb(xt)
            yt = -(mu + self.beta * sigma)
            cmaes.tell(xt, yt)
        
        result = cmaes.result
        xnew = result.xbest
        self.node2idx[tuple(xnew)] = self.node2idx[tuple(node.x)]
        node.x = result.xbest
        mu, sigma = self.eval_ucb(node.x)
        self.means[node_idx] = mu
        self.variances[node_idx] = sigma**2
        self.I[leaf_idx] = np.min([
            mu + self.beta * sigma,
            self.means[self._get_node_idx(node.father)] + self.beta * np.sqrt(self.variances[self._get_node_idx(node.father)]) + self._compute_V(node.father.level)
        ]) + self._compute_V(node.level) if node.level > 0 else ucb_x + self._compute_V(0)
        self.leaf_set[leaf_idx] = node
        return node


    def eval_ucb(self, x):
        K_sm = self.dot(x, self.active_set)
        Xstar_embedded = K_sm.dot(self.U_thin * self.S_thin_inv_sqrt.T)
        mu = Xstar_embedded.dot(self.w)
        temp = solve_triangular(self.R,
                        (Xstar_embedded.dot(self.Q)).T,
                        overwrite_b=True,
                        check_finite=False).T
        temp *= Xstar_embedded
        norm = diagonal_dot(x.reshape(-1, self.d), self.dot)
        Xstar_norms_embedded = np.square(np.linalg.norm(Xstar_embedded, axis = 1))
        var = (
            (norm - Xstar_norms_embedded) / self.lam
            + np.sum(temp, axis=1)
        )
        return mu, np.sqrt(var)# + self.beta * np.sqrt(var)

    def step(self):
        while True:
            leaf_idx, Vh = self._select_node()
            node = self.leaf_set[leaf_idx]
            node_idx = self._get_node_idx(node) 
            self._update_best(node_idx, leaf_idx, Vh)
            if self._can_be_expanded(node_idx, node.level, Vh ):
                self._expand(leaf_idx, node_idx, len(self.leaf_set) == 1)
            else:
                node = self.explore_partition(node,leaf_idx, node_idx)
                return self.leaf_set[leaf_idx].x, node_idx
