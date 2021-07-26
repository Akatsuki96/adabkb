import unittest as ut
import numpy as np

import time

from adabkb.options import  OptimizerOptions
from adabkb.utils import GreedyExpansion
from adabkb.optimizer import AdaBKB, AdaBBKB, AdaBBKBWP, AdaBBKBWP_2

from sklearn.gaussian_process.kernels import RBF

from benchmark_functions import Branin, Booth, SixHumpCamel, Rosenbrock,\
     Hartmann3, Ackley, Shekel, Hartmann6, Levy
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
        v_1 = self.N
        rho = self.N ** (-1)
        sigma = 0.85
        gfun = lambda x : (np.sqrt(2)/sigma) * x     
        lam=0.01
        noise_var = lam**2
        opt = OptimizerOptions(GreedyExpansion(), gfun, v_1=v_1, rho=rho, lam=lam, noise_var=noise_var, delta=0.0005, verbose=False)
        optimizer = AdaBKB(RBF(sigma), self.opt)
        t = 0
        T = 500
        h_max = 6
        test_fun = Branin(noise_params=(0.01, np.random.RandomState(42)))
        optimizer.initialize(lambda x: -test_fun(x),test_fun.search_space, self.N, budget=T, h_max=h_max)
        while t < T:
            xt, idx = optimizer.step()
            yt = -test_fun(xt)
           # print("[--] xt: {}\t yt: {}".format(xt, yt))
            optimizer.update_model(idx, yt)
            t += 1
        fun = Branin()
       # print("[--] f(x_T) = {}\tf(x*) = {}".format(fun(xt), fun.global_min[1]))
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
        self.assertAlmostEqual(fun(xt), fun.global_min[1], delta=1.0)

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
        lam=1e-10

        self.test_fun = lambda x: -(x**2 )#pwkernel(x, w_star)
        self.search_space = np.array([[-1.0, 10.0]]).reshape(-1, 2)
        sigma = 15.0
        self.N = 3
        noise_var = lam**2
        self.kernel = RBF(sigma)
        self.gfun = lambda x : (np.sqrt(2)/sigma) * x
        v_1 = self.N
        rho = 0.1#self.N ** (-1)
        self.opt = OptimizerOptions(GreedyExpansion(), self.gfun, v_1=v_1, 
                            rho=rho, lam=lam, noise_var=noise_var, delta=0.0005, verbose=False)


    def temp_test(self):
        optimizer = AdaBBKB(self.kernel,ratio_threshold=2.0, options=self.opt)
        optimizer.initialize(self.test_fun,self.search_space, self.N, budget=30, h_max=6)
        t = 0
        T = 500
        while t <= T:
            xs, batch = optimizer.step()
            ys = []
            for x in xs:
                ys.append( self.test_fun(x)[0])
            optimizer.update_model(batch, ys)
            t+=len(batch)

    def branin_test_case(self):
        optimizer = AdaBBKB(RBF(15.0), ratio_threshold=2.0, options=self.opt)
        t = 0
        T = 500
        h_max = 6
        test_fun = Branin(noise_params=(0.01, np.random.RandomState(42)))
        optimizer.initialize(lambda x: -test_fun(x),test_fun.search_space, self.N, budget=T, h_max=h_max)
        while t < T:
            xs, batch = optimizer.step()
            if t + len(batch) > T:
                xs = xs[:T - t]
                batch = batch[:T - t]
            ys = []
            for x in xs:
                yt = -test_fun(x)
                ys.append(yt)

            optimizer.update_model(batch, ys)
            t+=len(batch)

        fun = Branin()
        self.assertAlmostEqual(fun(xs[0]), fun(fun.global_min[0][0]),delta=1)

    def six_hump_camel_test_case(self):
        lam=1e-5
        noise_var = lam**2
        N = 3
        v_1 = N * np.sqrt(2)
        sigma = 0.7
        gfun = lambda x : (np.sqrt(2)/sigma) * x
        rho = 1/(N**2)
        opt = OptimizerOptions(GreedyExpansion(), gfun, v_1=v_1, rho=rho, lam=lam, noise_var=noise_var, delta=0.0005, verbose=False)
        optimizer = AdaBBKB(RBF(sigma), ratio_threshold=2.0,options=opt)
        t = 0
        T = 500
        h_max = 6
        test_fun = SixHumpCamel(noise_params=(0.01, np.random.RandomState(42)))
        optimizer.initialize(lambda x: -test_fun(x),test_fun.search_space, N, budget=T, h_max=h_max)
        while t < T:    
            xs, batch = optimizer.step()
            if t + len(batch) > T:
                xs = xs[:T - t]
                batch = batch[:T - t]
            ys = []
            for x in xs:
                yt = -test_fun(x)
                ys.append(yt)

            optimizer.update_model(batch, ys)
            t+=len(batch)
        fun = SixHumpCamel()
        self.assertAlmostEqual(fun(xs[0]), fun.global_min[1], delta=1.0)

    def hartmann_3_test_case(self):
        lam=0.01
        noise_var = lam**2
        N = 2
        v_1 = N * np.sqrt(2)
        sigma = 0.15
        gfun = lambda x : (np.sqrt(2)/sigma) * x
        rho = 1/(N**2)
        opt = OptimizerOptions(GreedyExpansion(), gfun, v_1=v_1, rho=rho, lam=lam, noise_var=noise_var, delta=0.000025, verbose=False)
        optimizer = AdaBBKB(RBF(sigma), ratio_threshold=2.0,options=opt)
        t = 0
        T = 500
        h_max = 6
        test_fun = Hartmann3(noise_params=(0.01, np.random.RandomState(42)))
        optimizer.initialize(lambda x: -test_fun(x),test_fun.search_space, N, budget=T, h_max=h_max)
        while t < T:    
            xs, batch = optimizer.step()
            if t + len(batch) > T:
                xs = xs[:T - t]
                batch = batch[:T - t]
            ys = []
            for x in xs:
                yt = -test_fun(x)
                ys.append(yt)

            optimizer.update_model(batch, ys)
            t+=len(batch)
        fun = Hartmann3()
        self.assertAlmostEqual(fun(xs[0]), fun.global_min[1], delta=1.0)

    def hartmann6_test_case(self):
        lam=0.01
        noise_var = lam**2
        N = 2
        v_1 =  N * np.sqrt(6)
        sigma = 1.10
        gfun = lambda x : (np.sqrt(2/sigma) * x)
        rho = 1/ N#N**(- 1/3)
        opt = OptimizerOptions(GreedyExpansion(), gfun, v_1=v_1, rho=rho, lam=lam, noise_var=noise_var, delta=1e-5, verbose=False)
        optimizer = AdaBBKB(RBF(sigma), ratio_threshold=2.0, options=opt)
        t = 0
        T = 700
        h_max = 10
        test_fun = Hartmann6(noise_params=(0.01, np.random.RandomState(42)))
        optimizer.initialize(lambda x: -test_fun(x),test_fun.search_space, N, budget=T, h_max=h_max)
        while t < T:    
            xs, batch = optimizer.step()
            if t + len(batch) > T:
                xs = xs[:T - t]
                batch = batch[:T - t]
            ys = []
            for x in xs:
                yt = -test_fun(x)
                ys.append(yt)

            optimizer.update_model(batch, ys)
            t+=len(batch)
        fun = Hartmann6()
        self.assertAlmostEqual(fun(xs[0]), fun.global_min[1], delta=1.5)

class AdaBBKBWPTestCase(ut.TestCase):
    def setUp(self):
        rnd_state = np.random.RandomState(42)
        lam=1e-10

        self.test_fun = lambda x: -(x**2 )#pwkernel(x, w_star)
        self.search_space = np.array([[-1.0, 1.0]]).reshape(-1, 2)
        sigma = 1.0
        self.N = 3
        noise_var = lam**2
        self.kernel = RBF(sigma)
        self.gfun = lambda x : (np.sqrt(2)/sigma) * x
        v_1 = self.N
        rho = self.N ** (-1)
        self.opt = OptimizerOptions(GreedyExpansion(), self.gfun, v_1=v_1, 
                            rho=rho, lam=lam, noise_var=noise_var, delta=1e-10, verbose=False)


    def temp_test(self):
        optimizer = AdaBBKBWP(self.kernel,ratio_threshold=2.0, options=self.opt)
        optimizer.initialize(self.test_fun,self.search_space, self.N, budget=30, h_max= 6)
        t = 0
        T = 100
        tot_time = time.time()
        while t <= T:
            xs, batch = optimizer.step()
            if batch is None:
                print("[--] t: {}/{}\tBest found: {} => {}".format(t, T, xs, self.test_fun(xs[0])))
                break
            ys = []
            for x in xs:
                yt = self.test_fun(x)[0]
            #    print("\t[--] t: {}/{}\tx: {}\ty: {}".format(t, T, x, yt))
                ys.append(yt)

            optimizer.update_model(batch, ys)
            t+=len(batch)
#            print("[--] Leaf set size: {}".format(len(optimizer.leaf_set)))
        print("[--] Num pruning: {}".format(optimizer.num_pruning))
        print("[--] x: {} => {}\t time: {}".format(xs[-1], self.test_fun(xs[-1]), time.time() - tot_time))

    def temp_test_no_hmax(self):
        optimizer = AdaBBKBWP(self.kernel,ratio_threshold=2.0, options=self.opt)
        optimizer.initialize(self.test_fun,self.search_space, self.N, budget=30, h_max= None)
        t = 0
        T = 100
        tot_time = time.time()
        while t <= T:
            xs, batch = optimizer.step()
            if batch is None:
                print("[--] t: {}/{}\tBest found: {} => {}".format(t, T, xs, self.test_fun(xs[0])))
                break
            ys = []
            for x in xs:
                yt = self.test_fun(x)[0]
          #      print("\t[--] t: {}/{}\tx: {}\ty: {}".format(t, T, x, yt))
                ys.append(yt)

            optimizer.update_model(batch, ys)
            t+=len(batch)
         #   print("[--] Leaf set size: {}".format(len(optimizer.leaf_set)))
        print("[--] Num pruning: {}".format(optimizer.num_pruning))
        print("[--] x: {} => {}\t time: {}".format(xs[-1], self.test_fun(xs[-1]), time.time() - tot_time))

    def branin_test_case(self):
        v_1 = self.N * np.sqrt(2)
        rho = self.N ** (-1/2)
        sigma = 0.85
        gfun = lambda x : (np.sqrt(2)/sigma) * x     
        lam=0.1
        noise_var = lam**2
        opt = OptimizerOptions(GreedyExpansion(), gfun, v_1=v_1, rho=rho, lam=lam, noise_var=noise_var, delta=1e-10, verbose=False)
        optimizer = AdaBBKBWP(RBF(sigma),ratio_threshold=2.0,options= opt)
        t = 0
        T = 1000
        h_max = 6
        test_fun = Branin(noise_params=(lam, np.random.RandomState(42)))
        optimizer.initialize(lambda x: -test_fun(x),test_fun.search_space, self.N, budget=T, h_max=None)
        while t < T:
            xs, batch = optimizer.step()
            if batch is None:
                print("[--] t: {}/{}\tBest node: {}".format(t, T, xs[0]))
                break
            if t + len(batch) > T:
                xs = xs[:T - t]
                batch = batch[:T - t]
            ys = []
            for x in xs:
                yt = -test_fun(x)
                ys.append(yt)
                print("[--] xt: {}\tyt: {}".format(x, yt))

            optimizer.update_model(batch, ys)
            t+=len(batch)
        print("[--] N. pruned leafs: {}\t Leaf set size: {}".format(optimizer.num_pruning, len(optimizer.leaf_set)))


    def hartmann_3_test_case(self):
        lam=0.001
        noise_var = lam**2
        N = 2
        v_1 = N * np.sqrt(2)
        sigma = 0.15
        gfun = lambda x : (np.sqrt(2)/sigma) * x
        rho = 1/(N**2)
        opt = OptimizerOptions(GreedyExpansion(), gfun, v_1=v_1, rho=rho, lam=lam, noise_var=noise_var, delta=0.00025, verbose=False)
        optimizer = AdaBBKBWP(RBF(sigma), ratio_threshold=2.0,options=opt)
        t = 0
        T = 700
        h_max = None
        test_fun = Hartmann3(noise_params=(lam, np.random.RandomState(42)))
        optimizer.initialize(lambda x: -test_fun(x),test_fun.search_space, N, budget=T, h_max=h_max)
        while t < T:    
            xs, batch = optimizer.step()
            if batch is None:
                print("[--] t: {}/{}\tBest node: {}".format(t, T, xs))
                xs = [xs]
                break
            if t + len(batch) > T:
                xs = xs[:T - t]
                batch = batch[:T - t]
            ys = []
            for x in xs:
                yt = -test_fun(x)
                ys.append(yt)
                print("[--] t: {}/{}\txt: {}\tyt: {}".format(t, T, x, yt))

            optimizer.update_model(batch, ys)
            t+=len(batch)
        print("[--] N. pruned leafs: {}\t Leaf set size: {}".format(optimizer.num_pruning, len(optimizer.leaf_set)))
        fun = Hartmann3()
        self.assertAlmostEqual(fun(xs[0]), fun.global_min[1], delta=0.55)

    def hartmann_6_test_case(self):
        lam=0.01
        noise_var = lam**2
        N = 2
        v_1 =  N #* np.sqrt(6)
        sigma = 0.1
        gfun = lambda x : (np.sqrt(2/sigma) * x)
        rho = 1/N
        opt = OptimizerOptions(GreedyExpansion(), gfun, v_1=v_1, rho=rho, lam=lam, noise_var=noise_var, delta=1e-5, verbose=False)
        optimizer = AdaBBKBWP(RBF(sigma), ratio_threshold=2.0, options=opt)
        t = 0
        T = 1000
        h_max = None
        test_fun = Hartmann6(noise_params=(lam, np.random.RandomState(42)))
        optimizer.initialize(lambda x: -test_fun(x),test_fun.search_space, N, budget=T, h_max=h_max)
        while t < T:    
            xs, batch = optimizer.step()
            if batch is None:
                print("[--] t: {}/{}\tBest node: {}".format(t, T, xs))
                xs = [xs]
                break
            if t + len(batch) > T:
                xs = xs[:T - t]
                batch = batch[:T - t]
            ys = []
            for x in xs:
                yt = -test_fun(x)
                ys.append(yt)
                print("[--] t: {}/{}\txt: {}\tyt: {}\tlset size: {}".format(t, T, x, yt, len(optimizer.leaf_set)))

            optimizer.update_model(batch, ys)
            t+=len(batch)
        print("[--] N. pruned leafs: {}\t Leaf set size: {}".format(optimizer.num_pruning, len(optimizer.leaf_set)))
        fun = Hartmann6()
        self.assertAlmostEqual(fun(xs[0]), fun.global_min[1], delta=1.0)

    def levy_8_test_case(self):
        lam=0.1
        noise_var = lam**2
        N = 5
        v_1 =  N * np.sqrt(6)
        sigma = 2.0
        gfun = lambda x : (np.sqrt(2/sigma) * x)
        rho = (1/N) #1/(N**8)
        opt = OptimizerOptions(GreedyExpansion(), gfun, v_1=v_1, rho=rho, lam=lam, noise_var=noise_var, delta=0.25, verbose=False)
        optimizer = AdaBBKBWP(RBF(sigma), ratio_threshold=2.0, options=opt)
        t = 0
        T = 700
        h_max = None
        test_fun = Levy(d=6,noise_params=(lam, np.random.RandomState(42)))
        optimizer.initialize(lambda x: -test_fun(x),test_fun.search_space, N, budget=T, h_max=h_max)
        while t < T:    
            xs, batch = optimizer.step()
            if batch is None:
                print("[--] t: {}/{}\tBest node: {}".format(t, T, xs))
                xs = [xs]
                break
            if t + len(batch) > T:
                xs = xs[:T - t]
                batch = batch[:T - t]
            ys = []
            for x in xs:
                yt = -test_fun(x)
                ys.append(yt)
                print("[--] t: {}/{}\txt: {}\tyt: {}\tlset size: {}".format(t, T, x, yt, len(optimizer.leaf_set)))

            optimizer.update_model(batch, ys)
            t+=len(batch)
        print("[--] N. pruned leafs: {}\t Leaf set size: {}".format(optimizer.num_pruning, len(optimizer.leaf_set)))
        fun = Levy(d=6)
        print("[--] xt: {}\tyt: {}".format(xs[0], fun(xs[0])))
        
class AdaBBKBWP_2TestCase(ut.TestCase):
    def setUp(self):
        rnd_state = np.random.RandomState(42)
        lam=1e-10

        self.test_fun = lambda x: -(x**2 )#pwkernel(x, w_star)
        self.search_space = np.array([[-1.0, 1.0]]).reshape(-1, 2)
        sigma = 1.0
        self.N = 3
        noise_var = lam**2
        self.kernel = RBF(sigma)
        self.gfun = lambda x : (np.sqrt(2)/sigma) * x
        v_1 = self.N
        rho = self.N ** (-1)
        self.opt = OptimizerOptions(GreedyExpansion(), self.gfun, v_1=v_1, 
                            rho=rho, lam=lam, noise_var=noise_var, delta=1e-10, verbose=False)


    def temp_test(self):
        optimizer = AdaBBKBWP_2(self.kernel,ratio_threshold=2.0, options=self.opt)
        optimizer.initialize(self.test_fun,self.search_space, self.N, budget=30, h_max= 6)
        t = 0
        T = 100
        tot_time = time.time()
        while t <= T:
            xs, batch = optimizer.step()
            if batch is None:
                print("[--] t: {}/{}\tBest found: {} => {}".format(t, T, xs, self.test_fun(xs[0])))
                break
            ys = []
            for x in xs:
                yt = self.test_fun(x)[0]
            #    print("\t[--] t: {}/{}\tx: {}\ty: {}".format(t, T, x, yt))
                ys.append(yt)

            optimizer.update_model(batch, ys)
            t+=len(batch)
         #   print("[--] Leaf set size: {}".format(len(optimizer.leaf_set)))
        print("[--] Num pruning: {}".format(optimizer.num_pruning))
        print("[--] x: {} => {}\t time: {}".format(xs[-1], self.test_fun(xs[-1]), time.time() - tot_time))

    def temp_test_no_hmax(self):
        optimizer = AdaBBKBWP_2(self.kernel,ratio_threshold=2.0, options=self.opt)
        optimizer.initialize(self.test_fun,self.search_space, self.N, budget=30, h_max= None)
        t = 0
        T = 100
        tot_time = time.time()
        while t <= T:
            xs, batch = optimizer.step()
            if batch is None:
                print("[--] t: {}/{}\tBest found: {} => {}".format(t, T, xs, self.test_fun(xs[0])))
                break
            ys = []
            for x in xs:
                yt = self.test_fun(x)[0]
#                print("\t[--] t: {}/{}\tx: {}\ty: {}".format(t, T, x, yt))
                ys.append(yt)

            optimizer.update_model(batch, ys)
            t+=len(batch)
        #    print("[--] Leaf set size: {}".format(len(optimizer.leaf_set)))
        print("[--] Num pruning: {}".format(optimizer.num_pruning))
        print("[--] x: {} => {}\t time: {}".format(xs[-1], self.test_fun(xs[-1]), time.time() - tot_time))