import numpy as np


from adabkb.optimizer import AdaBKB
from adabkb.utils import SplitOnRepresenter, diagonal_dot, stable_invert_root, PartitionTreeNode
from scipy.linalg import solve_triangular, svd, qr, qr_update, LinAlgError


class GradBKB(AdaBKB):


    def initialize(self, search_space, N, h_max : int = 100,\
         l : int = 1,\
         opt_budget : int = 100,\
         opt_tol : float = 1e-7,\
         fd_app : float = 1e-10):
        super().initialize(search_space, N, h_max)
        self.opt_budget = opt_budget
        self.opt_tol = opt_tol
        self.l = l
        self.fd_app = fd_app
        self.d = search_space.shape[0]
        self.eye = np.eye(self.d, self.d)


    def eval_ucb(self, x):
        K_sm = self.dot(x, self.active_set)
        Xstar_embedded = K_sm.dot(self.U_thin * self.S_thin_inv_sqrt.T)
        mu = Xstar_embedded.dot(self.w)
        temp = solve_triangular(self.R,
                        (Xstar_embedded.dot(self.Q)).T,
                        overwrite_b=True,
                        check_finite=False).T
        temp *= Xstar_embedded
        norm = diagonal_dot(x.reshape(-1, self.Pk.shape[0]), self.dot)
        Xstar_norms_embedded = np.square(np.linalg.norm(Xstar_embedded, axis = 1))
        var = (
            (norm - Xstar_norms_embedded) / self.lam
            + np.sum(temp, axis=1)
        )
        return mu, np.sqrt(var)# + self.beta * np.sqrt(var)


    def explore_partition(self, node, leaf_idx, node_idx):
        x = x0 = node.partition.mean(axis=1)
        k = 1
        while k < self.opt_budget:
            dirs = self.random_state.choice(range(self.d), size=self.l, replace=False) 
            self.Pk = self.eye[dirs].T
            
            #print(self.Pk)
            grad_app = 0
            mu_x, sigma_x = self.eval_ucb(x)
            ucb_x = mu_x + self.beta * sigma_x

            #print("[--] f(x) = {}".format(ucb_x))

            for j in range(self.Pk.shape[1]):
                mu_dir, sigma_dir = self.eval_ucb(x + self.fd_app * self.Pk[:, j])
                ucb_dir = mu_dir + self.beta * sigma_dir                
                grad_app += ((ucb_dir - ucb_x)/self.fd_app) * self.Pk[:,j]
            #print("[--] ucb(x): {}\t |grad|^2: {}".format(ucb_x, np.linalg.norm(grad_app)**2))
            if np.linalg.norm(grad_app) <= self.opt_tol:
                break
            x, adapted = node.adapt(x + (1/np.sqrt(k)) * grad_app)
            if adapted:
                break
            k+=1
        #print("mu: {} sigma: {}".format(mu_x, sigma_x))
        node.x = x
        self.node2idx[tuple(x)] =  self.node2idx[tuple(x0)]
        self.means[node_idx] = mu_x
        self.variances[node_idx] = sigma_x
        self.I[leaf_idx] = np.min([
            ucb_x,
            self.means[self._get_node_idx(node.father)] + self.beta * np.sqrt(self.variances[self._get_node_idx(node.father)]) + self._compute_V(node.father.level)
        ]) + self._compute_V(node.level) if node.level > 0 else ucb_x + self._compute_V(0)
        self.leaf_set[leaf_idx] = node
        return node


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
