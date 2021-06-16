import unittest as ut
import numpy as np


from options import OptimizerOptions
from utils import GreedyExpansion
from optimizer import AdaBKB

from sklearn.gaussian_process.kernels import RBF

from benchmark_functions import Branin

class AdaBKBTestCase(ut.TestCase):

    def setUp(self):
        self.test_fun = lambda x: -(x**2)
        self.search_space = np.array([[-10.0, 25.0]])
        sigma = 15.0
        self.N = 3
        lam=1e-10
        noise_var = lam**2
        self.kernel = RBF(sigma)
        gfun = lambda x : (np.sqrt(2)/sigma) * x
        v_1 = self.N
        rho = 0.1
        self.opt = OptimizerOptions(GreedyExpansion(), gfun, v_1=v_1, rho=rho, lam=lam, noise_var=noise_var, delta=0.0005, verbose=True)


    def initialization_test_case(self):
        optimizer = AdaBKB(self.kernel, self.opt)
        leaf_set = optimizer.initialize(self.test_fun,self.search_space, self.N, budget=30)
        self.assertEqual(len(leaf_set), self.N)
        self.assertEqual(len(optimizer.I), self.N)

    def single_step_test_case(self):
        optimizer = AdaBKB(self.kernel, self.opt)
        _ = optimizer.initialize(self.test_fun,self.search_space, self.N, budget=30)
        optimizer.step()

    def branin_test_case(self):
        optimizer = AdaBKB(self.kernel, self.opt)
        t = 0
        T = 500
        h_max = 6
        test_fun = Branin(noise_params=(0.01, np.random.RandomState(42)))
        optimizer.initialize(lambda x: -test_fun(x),test_fun.search_space, self.N, budget=T, h_max=h_max)
        while t < T:
            xt, idx = optimizer.step()
            yt = -test_fun(xt)
            print("[--] xt: {}\tyt: {}".format(xt, yt))
            optimizer.update_model(idx, yt)
            print("[--] t: {}/{}\tidx: {}\txt: {}\tyt: {}\tlfset size: {}".format(t, T, idx, xt, yt, len(optimizer.leaf_set)))
            t += 1
        

