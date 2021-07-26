import unittest as ut
import numpy as np

import time

from adabkb.options import  OptimizerOptions
from adabkb.utils import GreedyExpansion
from adabkb.optimizer import SafeAdaBKB
from sklearn.gaussian_process.kernels import RBF

from benchmark_functions import Branin, Booth, SixHumpCamel, Rosenbrock,\
     Hartmann3, Ackley, Shekel, Hartmann6, Levy




class SafeAdaBKB_TestCase(ut.TestCase):
    def setUp(self):
        self.test_fun = lambda x: -(x**2)
        self.search_space = np.array([[-10.0, 25.0]])
        sigma = 15.0
        self.N = 3
        jmin = 1.0
        lam=1e-10
        noise_var = lam**2
        self.kernel = RBF(sigma)
        gfun = lambda x : (np.sqrt(2)/sigma) * x
        v_1 = self.N
        rho = 0.1
        self.opt = OptimizerOptions(GreedyExpansion(), gfun, v_1=v_1, rho=rho, lam=lam, noise_var=noise_var, delta=0.0005, verbose=False)
