from numpy.core.fromnumeric import mean
from numpy.core.numeric import indices
from scipy.linalg import solve_triangular, svd, qr, LinAlgError
import numpy as np

from adabkb.utils import  diagonal_dot, stable_invert_root, PartitionTreeNode

from pytictoc import TicToc
from cmaes import CMA

from adabkb.optimizer import AbsOptimizer
from adabkb.options import OptimizerOptions


class SafeAdaBKB(AbsOptimizer):


    def __init__(self, jmin: float, beta: float, options: OptimizerOptions = None):
        super().__init__(options=options)
        self.jmin = jmin
        self.beta = beta

    def initialize(self, search_space, N, hmax, y0):
        root = PartitionTreeNode(search_space, N, None, 0, 0, self.options.expand_fun)
        self.leaf_set = [root]
        self.safe_idx = [0]
        self.hmax = hmax
        self.X = []
        self.Y = []