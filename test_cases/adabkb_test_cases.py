import unittest as ut
import numpy as np


from options import OptimizerOptions
from utils import GreedyExpansion
from optimizer import AdaBKB

from sklearn.gaussian_process.kernels import RBF



class AdaBKBTestCase(ut.TestCase):

    def initialization_test_case(self):
        test_fun = lambda x: - (x**2)
        search_space = np.array([[-1.0, 2.0]])
        sigma = 1.0
        N = 3
        lam=1e-10
        noise_var = lam**2
        kernel = RBF(sigma)
        gfun = lambda x : (np.sqrt(2)/sigma) * x
        opt = OptimizerOptions(GreedyExpansion(), gfun, lam=lam, noise_var=noise_var, delta=0.0005, verbose=True)
        optimizer = AdaBKB(kernel, opt)
        leaf_set = optimizer.initialize(test_fun,search_space, N, budget=30)
        self.assertEqual(len(leaf_set),N)
