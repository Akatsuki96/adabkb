import unittest as ut
import numpy as np


from options import OptimizerOptions
from utils import GreedyExpansion
from optimizer import AdaBKB, AdaBBKB

from sklearn.gaussian_process.kernels import RBF

from benchmark_functions import Branin, Booth, SixHumpCamel, Rosenbrock, Hartmann3, Ackley, Shekel, Hartmann6
from sklearn.gaussian_process.kernels import PairwiseKernel

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
        self.opt = OptimizerOptions(GreedyExpansion(), gfun, v_1=v_1, rho=rho, lam=lam, noise_var=noise_var, delta=0.0005, verbose=False)


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
            optimizer.update_model(idx, yt)
            t += 1
        fun = Branin()
        self.assertAlmostEqual(fun(xt), fun.global_min[1], delta=1.0)

    def booth_test_case(self):
        lam=1e-10
        noise_var = lam**2
        N = 7
        v_1 = N * np.sqrt(2)
        sigma = 1.50
        gfun = lambda x : (np.sqrt(2)/sigma) * x
        rho = 1/(N**2) 
        opt = OptimizerOptions(GreedyExpansion(), gfun, v_1=v_1, rho=rho, lam=lam, noise_var=noise_var, delta=0.0005, verbose=False)
        optimizer = AdaBKB(RBF(sigma), opt)
        t = 0
        T = 500
        h_max = 6
        test_fun = Booth(noise_params=(0.01, np.random.RandomState(42)))
        optimizer.initialize(lambda x: -test_fun(x),test_fun.search_space, N, budget=T, h_max=h_max)
        while t < T:
            xt, idx = optimizer.step()
            yt = -test_fun(xt)
            optimizer.update_model(idx, yt)
            t += 1
        fun = Booth()
        self.assertAlmostEqual(fun(xt), fun.global_min[1], delta=1.0)
        
    def six_hump_camel_test_case(self):
        lam=0.01
        noise_var = lam**2
        N = 3
        v_1 = N * np.sqrt(2)
        sigma = 0.7
        gfun = lambda x : (np.sqrt(2)/sigma) * x
        rho = 1/(N**2)
        opt = OptimizerOptions(GreedyExpansion(), gfun, v_1=v_1, rho=rho, lam=lam, noise_var=noise_var, delta=0.0005, verbose=False)
        optimizer = AdaBKB(RBF(sigma), opt)
        t = 0
        T = 500
        h_max = 6
        test_fun = SixHumpCamel(noise_params=(0.01, np.random.RandomState(42)))
        optimizer.initialize(lambda x: -test_fun(x),test_fun.search_space, N, budget=T, h_max=h_max)
        while t < T:
            xt, idx = optimizer.step()
            yt = -test_fun(xt)
            optimizer.update_model(idx, yt)
            t += 1
        fun = SixHumpCamel()
        self.assertAlmostEqual(fun(xt), fun.global_min[1], delta=1.0)

    def rosenbrock_2_test_case(self):
        lam=0.01
        noise_var = lam**2
        N = 5
        v_1 = N * np.sqrt(2)
        sigma = 0.017
        gfun = lambda x : (np.sqrt(2)/sigma) * x
        rho = 1/(N**2)
        opt = OptimizerOptions(GreedyExpansion(), gfun, v_1=v_1, rho=rho, lam=lam, noise_var=noise_var, delta=0.25, verbose=False)
        optimizer = AdaBKB(RBF(sigma), opt)
        t = 0
        T = 500
        h_max = 7
        test_fun = Rosenbrock(2, noise_params=(0.01, np.random.RandomState(42)))
        optimizer.initialize(lambda x: -test_fun(x),test_fun.search_space, N, budget=T, h_max=h_max)
        while t < T:
            xt, idx = optimizer.step()
            yt = -test_fun(xt)
            optimizer.update_model(idx, yt)
            t += 1
        fun = Rosenbrock(2)
        self.assertAlmostEqual(fun(xt), fun.global_min[1], delta=2.0)

    def hartmann_3_test_case(self):
        lam=0.01
        noise_var = lam**2
        N = 2
        v_1 = N * np.sqrt(2)
        sigma = 0.15
        gfun = lambda x : (np.sqrt(2)/sigma) * x
        rho = 1/(N**2)
        opt = OptimizerOptions(GreedyExpansion(), gfun, v_1=v_1, rho=rho, lam=lam, noise_var=noise_var, delta=0.000025, verbose=False)
        optimizer = AdaBKB(RBF(sigma), opt)
        t = 0
        T = 500
        h_max = 6
        test_fun = Hartmann3(noise_params=(0.01, np.random.RandomState(42)))
        optimizer.initialize(lambda x: -test_fun(x),test_fun.search_space, N, budget=T, h_max=h_max)
        while t < T:
            xt, idx = optimizer.step()
            yt = -test_fun(xt)
            optimizer.update_model(idx, yt)
            t += 1
        fun = Hartmann3()
        self.assertAlmostEqual(fun(xt), fun.global_min[1], delta=2.0)

    def ackley_3_test_case(self):
        lam=0.01
        noise_var = lam**2
        N = 3
        v_1 =  N * np.sqrt(3)
        sigma = 5.7
        gfun = lambda x : (np.sqrt(2)/sigma) * x
        rho = N**(- 1/3)
        opt = OptimizerOptions(GreedyExpansion(), gfun, v_1=v_1, rho=rho, lam=lam, noise_var=noise_var, delta=0.000025, verbose=False)
        optimizer = AdaBKB(RBF(sigma), opt)
        t = 0
        T = 500
        h_max = 6
        test_fun = Ackley(3,noise_params=(0.01, np.random.RandomState(42)))
        optimizer.initialize(lambda x: -test_fun(x),test_fun.search_space, N, budget=T, h_max=h_max)
        while t < T:
            xt, idx = optimizer.step()
            yt = -test_fun(xt)
            optimizer.update_model(idx, yt)
            t += 1
        fun = Ackley(3)
        self.assertAlmostEqual(fun(xt), fun.global_min[1], delta=5.0)

    def ackley_5_test_case(self):
        lam=0.01
        noise_var = lam**2
        N = 3
        v_1 =  N * np.sqrt(5)
        sigma = 5.7
        gfun = lambda x : (np.sqrt(2)/sigma) * x
        rho = N**(- 1/5)
        opt = OptimizerOptions(GreedyExpansion(), gfun, v_1=v_1, rho=rho, lam=lam, noise_var=noise_var, delta=0.000025, verbose=False)
        optimizer = AdaBKB(RBF(sigma), opt)
        t = 0
        T = 500
        h_max = 6
        test_fun = Ackley(5,noise_params=(0.01, np.random.RandomState(42)))
        optimizer.initialize(lambda x: -test_fun(x),test_fun.search_space, N, budget=T, h_max=h_max)
        while t < T:
            xt, idx = optimizer.step()
            yt = -test_fun(xt)
            optimizer.update_model(idx, yt)
            t += 1
        fun = Ackley(5)
        self.assertAlmostEqual(fun(xt), fun.global_min[1], delta=5.0)

    def hartmann6_test_case(self):
        lam=0.01
        noise_var = lam**2
        N = 2
        v_1 =  N * np.sqrt(6)
        sigma = 1.10
        gfun = lambda x : (np.sqrt(2/sigma) * x)
        rho = 1/ N#N**(- 1/3)
        opt = OptimizerOptions(GreedyExpansion(), gfun, v_1=v_1, rho=rho, lam=lam, noise_var=noise_var, delta=1e-5, verbose=False)
        optimizer = AdaBKB(RBF(sigma), opt)
        t = 0
        T = 700
        h_max = 10
        test_fun = Hartmann6(noise_params=(0.01, np.random.RandomState(42)))
        optimizer.initialize(lambda x: -test_fun(x),test_fun.search_space, N, budget=T, h_max=h_max)
        while t < T:
            xt, idx = optimizer.step()
            yt = -test_fun(xt)
            optimizer.update_model(idx, yt)
            t += 1
        fun = Hartmann6()
        self.assertAlmostEqual(fun(xt), fun.global_min[1], delta=1.0)


class AdaBBKBTestCase(ut.TestCase):

    def setUp(self):
        rnd_state = np.random.RandomState(42)
        d = 1
        pwkernel = PairwiseKernel(metric='linear')
        w_star = rnd_state.randn(d).reshape(1,-1)
        lam=0.01

        self.test_fun = lambda x: -(x**2 + rnd_state.randn()*lam)#pwkernel(x, w_star)
        self.search_space = np.array([[-100.0, 100.0]]).reshape(-1, 2)
        sigma = 1.0
        self.N = 3
        noise_var = lam**2
        self.kernel = RBF(sigma)
        gfun = lambda x : (np.sqrt(2)/sigma) * x
        v_1 = self.N
        rho = self.N ** (-1)
        self.opt = OptimizerOptions(GreedyExpansion(), gfun, v_1=v_1, 
                            rho=rho, lam=lam, noise_var=noise_var, delta=0.0005, verbose=True)


    def temp_test(self):
        optimizer = AdaBBKB(self.kernel,ratio_threshold=1.0, options=self.opt)
        optimizer.initialize(self.test_fun,self.search_space, self.N, budget=30, h_max=6)
        for t in range(0, 100):
            xs, batch = optimizer.step()
            print("[--] batch size: {}".format(len(batch)))
            ys = []
            for x in xs:
                ys.append( self.test_fun(x)[0])
            optimizer.update_model(batch, ys)
        print([node.level for node in optimizer.leaf_set])
        print("[--] x*: {}".format(xs))