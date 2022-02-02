import numpy as np
from adabkb.utils import diagonal_dot

from adabkb.options import OptimizerOptions




class AbsOptimizer:
    def __init__(self, options: OptimizerOptions = None):
        self.options = options
        self.node2idx = {}
        self.father_ucbs = {}
        self.num_nodes = 0
        self.logdet = 1
        self.beta = 1

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
    def unique_arms_pulled(self):
        return np.count_nonzero(self.pulled_arms_count)

    @property
    def dict_size(self):
        return self.dict_arms_count.sum()

    @property
    def dict_size_unique(self):
        return np.count_nonzero(self.dict_arms_count)
        
    @property
    def early_stopping(self):
        return self.options.early_stopping

    @property
    def v_1(self):
        return self.options.v_1
    
    @property
    def dot(self):
        return self.options.kernel

    @property
    def rho(self):
        return self.options.rho

    @property
    def gfun(self):
        return self.options.kernel.confidence_function

    @property
    def verbose(self):
        return self.options.verbose


    def _get_node_idx(self, node):
        return self.node2idx[tuple(node.x)]

    def _compute_V(self, h):
        """Given a level \\(h \\geq 0\\), it compute \\(V_h\\) s.t. \\(\\forall i\\)
        \\[\\sup\\limits_{x, x^{\\prime} \\in X_{h,i}} |f(x) - f(x^\\prime)| \\leq V_h\\]

        Returns
        -------
        Vh : float
            upper bound on the function variation in cell at level h
        """
        return self.fnorm * self.gfun(self.v_1 * (self.rho**h) ) 
        
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