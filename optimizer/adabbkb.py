from scipy.linalg import solve_triangular, svd, qr, qr_update, LinAlgError
import numpy as np
from options import OptimizerOptions

from sklearn.gaussian_process.kernels import Kernel

from utils import GreedyExpansion, diagonal_dot, stable_invert_root, PartitionTreeNode

from pytictoc import TicToc
import itertools as it
import time

from optimizer import AdaBKB

class AdaBBKB(AdaBKB):
    def __init__(self, dot_fun: Kernel, ratio_threshold : float = 2.0, options: OptimizerOptions = None):
        self.ratio_threshold = ratio_threshold
        super().__init__(dot_fun, options)
