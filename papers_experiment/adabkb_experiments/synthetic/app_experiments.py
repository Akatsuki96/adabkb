import os
import time
import numpy as np
import itertools as it
from adabkb.benchmark_functions import *
from sklearn.gaussian_process.kernels import RBF
from adabkb.options import OptimizerOptions

from adabkb.optimizer import AdaBKB
from adabkb.other_methods import Bkb, AdaGpucb, GPUCB 

np.random.seed(12)

trials = 5
alpha = 1.0
T = 700
D = 6
delta = 1e-5
fnorm = 1.0
seed = 12
qbar = 5
time_threshold = 600


DELIM = "-----"

ack2_fun = Ackley(2, (0.01, np.random.RandomState(seed)))
bra_fun = Branin((0.01, np.random.RandomState(seed)))
bea_fun = Beale((0.01, np.random.RandomState(seed)))
boh_fun = Bohachevsky((0.01, np.random.RandomState(seed)))
shk_fun = Shekel((0.01, np.random.RandomState(seed)))
ros_fun = Rosenbrock(2, (0.01, np.random.RandomState(seed)))
tri2_fun = Trid(2, (0.01, np.random.RandomState(seed))) 
hart3_fun = Hartmann3((0.01, np.random.RandomState(seed)))
tri4_fun = Trid(4, (0.01, np.random.RandomState(seed))) 

def end_trace(fpath):
    with open(fpath, "a") as f:
        f.write("{}\n".format(DELIM))

def write_log(fpath, yreg, tm, lset_size, cpruned, could_stop):
    with open(fpath, "a") as f:
        f.write("{},{},{},{},{}\n".format(yreg, tm, lset_size, cpruned, could_stop))


def write_exp_info(config):
    os.makedirs("./out/{}".format(config['fun'].name), exist_ok=True)
    with open("./out/{}/info".format(config['fun'].name),"w") as f:
        f.write("[--] trials: {}\n".format(config['trials']))
        f.write("[--] T: {}\n".format(config['T']))
        f.write("[--] search space: {}\n".format(list(config['search_space'])))
        f.write("[++] BKB params\n")
        f.write("\t[--] sigma: {}\n".format(config['bkb_params']['sigma']))
        f.write("\t[--] lam: {}\n".format(config['bkb_params']['lam']))
        f.write("[++] Ada-BKB params\n")
        f.write("\t[--] sigma: {}\n".format(config['adabkb_params']['sigma']))
        f.write("\t[--] lam: {}\n".format(config['adabkb_params']['lam']))
        f.write("\t[--] N: {}\n".format(config['adabkb_params']['N']))
        f.write("\t[--] hmax: {}\n".format(config['adabkb_params']['hmax']))


def bkb_test(config):
    os.makedirs("./out/{}/BKB".format(config['fun'].name), exist_ok=True)
    trials = config['trials']
    T = config['T']
    fun = config['fun']
    nfree_fun = config['nfree_fun']
    ctimes = []
    cregs = []
    bkb_config = config['bkb_params']
    arm_set = bkb_config['arm_set']
    rnd_state = bkb_config['random_state']
    for _ in range(trials):
        ct = []
        creg = []
        tot_time = time.time()
        bkb = Bkb(bkb_config['lam'], bkb_config['kernel'], bkb_config['fnorm'], bkb_config['lam']**2,\
            bkb_config['delta'], bkb_config['qbar'])
        #    def initialize(self, arm_set, index_init, y_init, random_state: np.random.RandomState, dict_init=None):
        ind_init = np.random.choice(range(len(arm_set)), bkb_config['init_samples'], replace=False)
        y_init = np.array([-fun(x) for x in arm_set[ind_init]])
        bkb.initialize(arm_set, ind_init, y_init, rnd_state)
        for t in range(T):
            tm = time.time()
            ind, _ = bkb.predict()
            xt = arm_set[ind[0]]
            yt = -fun(xt)
            bkb.update(ind, [yt], rnd_state)
            tm = time.time() - tm
            print("[--] t: {}/{}\txt: {}\tyt: {}".format(t, T, xt, -yt))
            write_log("./out/{}/BKB/trace.log".format(fun.name), nfree_fun(xt), tm, 0, 0, False)
            creg.append(nfree_fun(xt) - fun.global_min[1])
            ct.append(tm)
            if time.time() - tot_time > time_threshold:
                break
        ctimes.append(ct)
        cregs.append(creg)
        end_trace("./out/{}/BKB/trace.log".format(fun.name))

def adabkb_test(config):
    os.makedirs("./out/{}/AdaBKB".format(config['fun'].name), exist_ok=True)
    trials = config['trials']
    T = config['T']
    fun = config['fun']
    nfree_fun = config['nfree_fun']
    ctimes = []
    cregs = []
    adabkb_config = config['adabkb_params']
    for _ in range(trials):
        ct = []
        creg = []
        tot_time = time.time()
        adabkb = AdaBKB(adabkb_config['kernel'], adabkb_config['options'])
        #    def initialize(self, search_space, N : int = 2, h_max : int = 100):
        adabkb.initialize(fun.search_space, adabkb_config['N'], adabkb_config['hmax'])
        for t in range(T):
            tm = time.time()
            xt, node_id = adabkb.step()
            yt = -fun(xt)
            adabkb.update_model(node_id, yt)
            tm = time.time() - tm
            print("[--] t: {}/{}\txt: {}\tyt: {}".format(t, T, xt, -yt))
            write_log("./out/{}/AdaBKB/trace.log".format(fun.name), nfree_fun(xt), tm, len(adabkb.leaf_set), adabkb.pruned, adabkb.estop)
            creg.append(nfree_fun(xt) - fun.global_min[1])
            ct.append(tm)
            if time.time() - tot_time > time_threshold or len(adabkb.leaf_set) < 1:
                break
        ctimes.append(ct)
        cregs.append(creg)
        end_trace("./out/{}/AdaBKB/trace.log".format(fun.name))

def gpucb_test(config):
    os.makedirs("./out/{}/GPUCB".format(config['fun'].name), exist_ok=True)
    trials = config['trials']
    T = config['T']
    fun = config['fun']
    nfree_fun = config['nfree_fun']
    ctimes = []
    cregs = []
    gpucb_config = config['gpucb_params']
    arm_set = gpucb_config['arm_set']
    rnd_state = gpucb_config['random_state']
    for _ in range(trials):
        ct = []
        creg = []
        tot_time = time.time()
        gpucb = GPUCB(gpucb_config['kernel'], gpucb_config['sigma'],\
            gpucb_config['lam'], gpucb_config['lam']**2,\
            gpucb_config['delta'], gpucb_config['a'],\
            gpucb_config['b'], gpucb_config['r'])
        ind_init = np.random.choice(range(len(arm_set)), gpucb_config['init_samples'], replace=False)
        y_init = np.array([-fun(x) for x in arm_set[ind_init]])
        #    def initialize(self, arm_set, index_init, y_init, rnd_state):
        gpucb.initialize(arm_set, ind_init, y_init, rnd_state)
        for t in range(T):
            tm = time.time()
            ind, _ = gpucb.predict()
            xt = arm_set[ind[0]]
            yt = -fun(xt)
            gpucb.update(ind, [yt], rnd_state)
            tm = time.time() - tm
            print("[--] t: {}/{}\txt: {}\tyt: {}".format(t, T, xt, -yt))
            write_log("./out/{}/GPUCB/trace.log".format(fun.name), nfree_fun(xt), tm, 0, 0, False)
            creg.append(nfree_fun(xt) - fun.global_min[1])
            ct.append(tm)
            if time.time() - tot_time > time_threshold:
                break
        ctimes.append(ct)
        cregs.append(creg)
        end_trace("./out/{}/GPUCB/trace.log".format(fun.name))

def adagpucb_test(config):
    os.makedirs("./out/{}/AdaGPUCB".format(config['fun'].name), exist_ok=True)
    trials = config['trials']
    T = config['T']
    fun = config['fun']
    nfree_fun = config['nfree_fun']
    ctimes = []
    cregs = []
    adagpucb_config = config['adagpucb_params']
    for _ in range(trials):
        ct = []
        creg = []
        #AdaGPUCB
        #(self, sigma, d: int = 1,\
        #C1: float = 1.0, lam = 1e-12, noise_var = 1e-12, fnorm = 1.0, delta = 0.5,\
        #      expand_fun: ExpansionProcedure = GreedyExpansion())
        adagpucb = AdaGpucb(adagpucb_config['sigma'], adagpucb_config['g'], adagpucb_config['v_1'],\
            adagpucb_config['rho'], fun.search_space.shape[0],\
            adagpucb_config['C1'], adagpucb_config['lam'], adagpucb_config['lam']**2,\
            adagpucb_config['fnorm'], adagpucb_config['delta'])
        #    def initialize(self, search_space, N : int = 2, h_max : int = 100):
        #    def initialize(self, target_fun, root):
        adagpucb.run(lambda x: -fun(x), nfree_fun, fun.search_space, adagpucb_config['N'], T,\
             real_best=fun.global_min[1], hmax = adagpucb_config['hmax'], time_threshold= time_threshold, 
             out_dir="./out/{}/AdaGPUCB/".format(fun.name))
        end_trace("./out/{}/AdaGPUCB/trace.log".format(fun.name))
        #adagpucb.initialize(fun, adagpucb_config['N'], adagpucb_config['hmax'])
        #for t in range(T):
        #    tm = time.time()
        #    xt, node_id = adabkb.step()
        #    yt = -fun(xt)
        #    adabkb.update_model(node_id, yt)
        #    tm = time.time() - tm
        #    print("[--] t: {}/{}\txt: {}\tyt: {}".format(t, T, xt, -yt))
        #    write_log("./out/{}/AdaBKB/trace.log".format(fun.name), nfree_fun(xt), tm, len(adabkb.leaf_set), adabkb.pruned, adabkb.estop)
        #    creg.append(nfree_fun(xt) - fun.global_min[1])
        #    ct.append(tm)
        #ctimes.append(ct)
        #cregs.append(creg)
        #end_trace("./out/{}/AdaBKB/trace.log".format(fun.name))



def get_hartmann6_config():
    sigma = 0.50
    lam = 1e-3
    C1 = 2.0#5e-6 #6e-6
    N = 5 
    v_1 =  N * np.sqrt(1/2)
    rho = N ** (-1/np.sqrt(2))
    hmax = int(np.log(T))#/ (2 * alpha * np.log(1/rho)))
    gfun = lambda x : (1/sigma) * x
    opt = OptimizerOptions(gfun, v_1 = v_1, rho = rho,\
        sigma = sigma, lam = lam, noise_var=lam**2, delta=delta,\
        fnorm=fnorm, qbar=qbar, seed=seed)
    arm_set = np.array(
        list(it.product(*[np.linspace(d[0], d[1], 10) for d in hman_fun.search_space]))
    ).reshape(-1, 6)
    config = {
        'trials' : trials,
        'T' : T,
        'fun' : hman_fun,
        'nfree_fun' : Hartmann6(),
        'search_space': hman_fun.search_space,
        'adabkb_params' : {
            'sigma' : sigma,
            'lam' : lam,
            'alpha' : alpha,
            'kernel' : RBF(sigma),
            'options' : opt,
            'N' : N,
            'hmax' : hmax
        },
        'bkb_params' : {
            'sigma' : sigma,
            'lam' : lam,
            'kernel' : RBF(sigma),
            'fnorm' : fnorm,
            'noise_variance' : lam**2,
            'delta' : delta,
            'random_state' : np.random.RandomState(12),
            'init_samples' : 2,
            'arm_set' : arm_set,
            'qbar' : qbar
        },
        'gpucb_params' :{
            'sigma' : sigma,
            'lam' : lam,
            'kernel' : RBF(sigma),
            'fnorm' : fnorm,
            'noise_variance' : lam**2,
            'delta' : delta,
            'a' : 0.1,
            'init_samples' : 2,
            'arm_set' : arm_set,
            'b' : 0.5,
            'random_state': np.random.RandomState(12),
            'r' : 1.01
        },
        'adagpucb_params':{
            'sigma' : sigma,
            'lam' : lam,
            'C1' : C1,
            'g' : gfun,
            'v_1' :  v_1,
            'rho' : rho,
            'delta' : delta,
            'fnorm' : fnorm,
            'N' : N,
            'hmax' : hmax
        }

    }
    return config

def get_branin_config():
    sigma = 0.50
    lam = 1e-3
    C1 = 1.0#5e-6 #6e-6
    N = 3
    v_1 =  N * np.sqrt(1/2)
    rho = N ** (-1/np.sqrt(2))
    hmax = 5
    gfun = lambda x : (1/sigma) * x
    opt = OptimizerOptions(gfun, v_1 = v_1, rho = rho,\
        sigma = sigma, lam = lam, noise_var=lam**2, delta=delta,\
        fnorm=fnorm, qbar=qbar, seed=seed)
    arm_set = np.array(
        list(it.product(*[np.linspace(d[0], d[1], 15) for d in bra_fun.search_space]))
    ).reshape(-1, 2)
    config = {
        'trials' : trials,
        'T' : T,
        'fun' : bra_fun,
        'nfree_fun' : Branin(),
        'search_space': bra_fun.search_space,
        'adabkb_params' : {
            'sigma' : sigma,
            'lam' : lam,
            'alpha' : alpha,
            'kernel' : RBF(sigma),
            'options' : opt,
            'N' : N,
            'hmax' : hmax
        },
        'bkb_params' : {
            'sigma' : sigma,
            'lam' : lam,
            'kernel' : RBF(sigma),
            'fnorm' : fnorm,
            'noise_variance' : lam**2,
            'delta' : delta,
            'random_state' : np.random.RandomState(12),
            'init_samples' : 3,
            'arm_set' : arm_set,
            'qbar' : qbar
        },
        'gpucb_params' :{
            'sigma' : sigma,
            'lam' : lam,
            'kernel' : RBF(sigma),
            'fnorm' : fnorm,
            'noise_variance' : lam**2,
            'delta' : delta,
            'a' : 0.1,
            'init_samples' : 2,
            'arm_set' : arm_set,
            'b' : 0.5,
            'random_state': np.random.RandomState(12),
            'r' : 1.01
        },
        'adagpucb_params':{
            'sigma' : sigma,
            'lam' : lam,
            'C1' : C1,
            'g' : gfun,
            'v_1' :  v_1,
            'rho' : rho,
            'delta' : delta,
            'fnorm' : fnorm,
            'N' : N,
            'hmax' : hmax
        }

    }
    return config


def get_beale_config():
    sigma = 1.0
    lam = 1e-3
    C1 = 1.0#5e-6 #6e-6
    N = 3
    v_1 =  N * np.sqrt(1/2)
    rho = N ** (-1/np.sqrt(2))
    hmax = 5
    gfun = lambda x : (1/sigma) * x
    opt = OptimizerOptions(gfun, v_1 = v_1, rho = rho,\
        sigma = sigma, lam = lam, noise_var=lam**2, delta=delta,\
        fnorm=fnorm, qbar=qbar, seed=seed)
    arm_set = np.array(
        list(it.product(*[np.linspace(d[0], d[1], 15) for d in bea_fun.search_space]))
    ).reshape(-1, 2)
    config = {
        'trials' : trials,
        'T' : T,
        'fun' : bea_fun,
        'nfree_fun' : Beale(),
        'search_space': bea_fun.search_space,
        'adabkb_params' : {
            'sigma' : sigma,
            'lam' : lam,
            'alpha' : alpha,
            'kernel' : RBF(sigma),
            'options' : opt,
            'N' : N,
            'hmax' : hmax
        },
        'bkb_params' : {
            'sigma' : sigma,
            'lam' : lam,
            'kernel' : RBF(sigma),
            'fnorm' : fnorm,
            'noise_variance' : lam**2,
            'delta' : delta,
            'random_state' : np.random.RandomState(12),
            'init_samples' : 3,
            'arm_set' : arm_set,
            'qbar' : qbar
        },
        'gpucb_params' :{
            'sigma' : sigma,
            'lam' : lam,
            'kernel' : RBF(sigma),
            'fnorm' : fnorm,
            'noise_variance' : lam**2,
            'delta' : delta,
            'a' : 0.1,
            'init_samples' : 2,
            'arm_set' : arm_set,
            'b' : 0.5,
            'random_state': np.random.RandomState(12),
            'r' : 1.01
        },
        'adagpucb_params':{
            'sigma' : sigma,
            'lam' : lam,
            'C1' : C1,
            'g' : gfun,
            'v_1' :  v_1,
            'rho' : rho,
            'delta' : delta,
            'fnorm' : fnorm,
            'N' : N,
            'hmax' : hmax
        }

    }
    return config

def get_ackley2_config():
    sigma = 3.50 #0.5
    lam = 1e-3
    C1 = 1.0#5e-6 #6e-6
    N = 3
    v_1 =  N * np.sqrt(1/2)
    rho = N ** (-1/np.sqrt(2))
    hmax = 7
    gfun = lambda x : (1/sigma) * x
    opt = OptimizerOptions(gfun, v_1 = v_1, rho = rho,\
        sigma = sigma, lam = lam, noise_var=lam**2, delta=delta,\
        fnorm=fnorm, qbar=qbar, seed=seed)
    arm_set = np.array(
        list(it.product(*[np.linspace(d[0], d[1], 15) for d in ack2_fun.search_space]))
    ).reshape(-1, 2)
    config = {
        'trials' : trials,
        'T' : T,
        'fun' : ack2_fun,
        'nfree_fun' : Ackley(2),
        'search_space': ack2_fun.search_space,
        'adabkb_params' : {
            'sigma' : sigma,
            'lam' : lam,
            'alpha' : alpha,
            'kernel' : RBF(sigma),
            'options' : opt,
            'N' : N,
            'hmax' : hmax
        },
        'bkb_params' : {
            'sigma' : sigma,
            'lam' : lam,
            'kernel' : RBF(sigma),
            'fnorm' : fnorm,
            'noise_variance' : lam**2,
            'delta' : delta,
            'random_state' : np.random.RandomState(12),
            'init_samples' : 3,
            'arm_set' : arm_set,
            'qbar' : qbar
        },
        'gpucb_params' :{
            'sigma' : sigma,
            'lam' : lam,
            'kernel' : RBF(sigma),
            'fnorm' : fnorm,
            'noise_variance' : lam**2,
            'delta' : delta,
            'a' : 0.1,
            'init_samples' : 3,
            'arm_set' : arm_set,
            'b' : 0.5,
            'random_state': np.random.RandomState(12),
            'r' : 1.01
        },
        'adagpucb_params':{
            'sigma' : sigma,
            'lam' : lam,
            'C1' : C1,
            'g' : gfun,
            'v_1' :  v_1,
            'rho' : rho,
            'delta' : delta,
            'fnorm' : fnorm,
            'N' : N,
            'hmax' : hmax
        }

    }
    return config


def get_shekel4_config():
    sigma = 1.75
    sigma_disc = 3.0
    lam = 1e-3
    C1 = 1.0#5e-6 #6e-6
    N = 9
    d = 4
    v_1 =  N * np.sqrt(1/d)
    rho = N ** (-1/np.sqrt(d))
    hmax = 6
    gfun = lambda x : (1/sigma) * x
    opt = OptimizerOptions(gfun, v_1 = v_1, rho = rho,\
        sigma = sigma, lam = lam, noise_var=lam**2, delta=delta,\
        fnorm=fnorm, qbar=qbar, seed=seed)
    arm_set = np.array(
        list(it.product(*[np.linspace(d[0], d[1], 15) for d in shk_fun.search_space]))
    ).reshape(-1, 4)
    config = {
        'trials' : trials,
        'T' : T,
        'fun' : shk_fun,
        'nfree_fun' : Shekel(),
        'search_space': shk_fun.search_space,
        'adabkb_params' : {
            'sigma' : sigma,
            'lam' : lam,
            'alpha' : alpha,
            'kernel' : RBF(sigma),
            'options' : opt,
            'N' : 9,
            'hmax' : hmax 
        },
        'bkb_params' : {
            'sigma' : sigma_disc,
            'lam' : lam,
            'kernel' : RBF(sigma_disc),
            'fnorm' : fnorm,
            'noise_variance' : lam**2,
            'delta' : delta,
            'random_state' : np.random.RandomState(12),
            'init_samples' : 3,
            'arm_set' : arm_set,
            'qbar' : qbar
        },
        'gpucb_params' :{
            'sigma' : sigma_disc,
            'lam' : lam,
            'kernel' : RBF(sigma_disc),
            'fnorm' : fnorm,
            'noise_variance' : lam**2,
            'delta' : delta,
            'a' : 0.1,
            'init_samples' : 3,
            'arm_set' : arm_set,
            'b' : 0.5,
            'random_state': np.random.RandomState(12),
            'r' : 1.01
        },
        'adagpucb_params':{
            'sigma' : sigma,
            'lam' : lam,
            'C1' : C1,
            'g' : gfun,
            'v_1' :  4 * np.sqrt(1/d),
            'rho' : 4**(-1/d),
            'delta' : delta,
            'fnorm' : fnorm,
            'N' : 4,
            'hmax' : hmax
        }

    }
    return config



def get_bohachevsky_config():
    sigma = 1.70
    sigma_disc = 5.70
    lam = 1e-3
    C1 = 1.0#5e-6 #6e-6
    N = 3
    v_1 =  N * np.sqrt(1/2)
    rho = N ** (-1/np.sqrt(2))
    hmax = 9
    gfun = lambda x : (1/sigma) * x
    opt = OptimizerOptions(gfun, v_1 = v_1, rho = rho,\
        sigma = sigma, lam = lam, noise_var=lam**2, delta=delta,\
        fnorm=fnorm, qbar=qbar, seed=seed)
    arm_set = np.array(
        list(it.product(*[np.linspace(d[0], d[1], 15) for d in boh_fun.search_space]))
    ).reshape(-1, 2)
    config = {
        'trials' : trials,
        'T' : T,
        'fun' : boh_fun,
        'nfree_fun' : Bohachevsky(),
        'search_space': boh_fun.search_space,
        'adabkb_params' : {
            'sigma' : sigma,
            'lam' : lam,
            'alpha' : alpha,
            'kernel' : RBF(sigma),
            'options' : opt,
            'N' : N,
            'hmax' : hmax
        },
        'bkb_params' : {
            'sigma' : sigma_disc,
            'lam' : lam,
            'kernel' : RBF(sigma_disc),
            'fnorm' : fnorm,
            'noise_variance' : lam**2,
            'delta' : delta,
            'random_state' : np.random.RandomState(12),
            'init_samples' : 10,
            'arm_set' : arm_set,
            'qbar' : qbar
        },
        'gpucb_params' :{
            'sigma' : sigma_disc,
            'lam' : lam,
            'kernel' : RBF(sigma_disc),
            'fnorm' : fnorm,
            'noise_variance' : lam**2,
            'delta' : delta,
            'a' : 0.1,
            'init_samples' : 10,
            'arm_set' : arm_set,
            'b' : 0.5,
            'random_state': np.random.RandomState(12),
            'r' : 1.01
        },
        'adagpucb_params':{
            'sigma' : sigma,
            'lam' : lam,
            'C1' : C1,
            'g' : gfun,
            'v_1' :  v_1,
            'rho' : rho,
            'delta' : delta,
            'fnorm' : fnorm,
            'N' : N,
            'hmax' : hmax
        }

    }
    return config


def get_levy8_config():
    sigma = 2.5
    lam = 1e-3
    C1 = 1.0#5e-6 #6e-6
    init_samples = int(np.sqrt(T))
    N = 3
    v_1 =  N * np.sqrt(1/8)
    rho = N ** (-1/np.sqrt(8))
    hmax = int(np.log(T)/ (2 * alpha * np.log(1/rho))) #5
    gfun = lambda x : (1/sigma) * x
    opt = OptimizerOptions(gfun, v_1 = v_1, rho = rho,\
        sigma = sigma, lam = lam, noise_var=lam**2, delta=delta,\
        fnorm=fnorm, qbar=qbar, seed=seed)
    arm_set = np.array(
        list(it.product(*[np.linspace(d[0], d[1], 5) for d in lev8_fun.search_space]))
    ).reshape(-1, 8)
    config = {
        'trials' : trials,
        'T' : T,
        'fun' : lev8_fun,
        'nfree_fun' : Levy(8),
        'search_space': lev8_fun.search_space,
        'adabkb_params' : {
            'sigma' : sigma,
            'lam' : lam,
            'alpha' : alpha,
            'kernel' : RBF(sigma),
            'options' : opt,
            'N' : N,
            'hmax' : hmax
        },
        'bkb_params' : {
            'sigma' : sigma,
            'lam' : lam,
            'kernel' : RBF(sigma),
            'fnorm' : fnorm,
            'noise_variance' : lam**2,
            'delta' : delta,
            'random_state' : np.random.RandomState(12),
            'init_samples' : init_samples,
            'arm_set' : arm_set,
            'qbar' : qbar
        },
        'gpucb_params' :{
            'sigma' : sigma,
            'lam' : lam,
            'kernel' : RBF(sigma),
            'fnorm' : fnorm,
            'noise_variance' : lam**2,
            'delta' : delta,
            'a' : 0.1,
            'init_samples' : init_samples,
            'arm_set' : arm_set,
            'b' : 0.5,
            'random_state': np.random.RandomState(12),
            'r' : 1.01
        },
        'adagpucb_params':{
            'sigma' : sigma,
            'lam' : lam,
            'C1' : C1,
            'g' : gfun,
            'v_1' :  v_1,
            'rho' : rho,
            'delta' : delta,
            'fnorm' : fnorm,
            'N' : N,
            'hmax' : hmax
        }

    }
    return config


def get_rosenbrock2_config():
    sigma = 0.70
    sigma_disc = 0.7
    lam = 1e-7
    C1 = 1.0#5e-6 #6e-6
    N = 11 
    v_1 =  N * np.sqrt(1/2)
    rho = N ** (-1/np.sqrt(2))
    hmax = 10#int(np.log(T)/ (2 * alpha * np.log(1/rho)))
    gfun = lambda x : (1/sigma) * x
    opt = OptimizerOptions(gfun, v_1 = v_1, rho = rho,\
        sigma = sigma, lam = lam, noise_var=lam**2, delta=delta,\
        fnorm=fnorm, qbar=qbar, seed=seed)
    arm_set = np.array(
        list(it.product(*[np.linspace(d[0], d[1], 15) for d in ros_fun.search_space]))
    ).reshape(-1, 2)
    config = {
        'trials' : trials,
        'T' : T,
        'fun' : ros_fun,
        'nfree_fun' : Rosenbrock(2),
        'search_space': ros_fun.search_space,
        'adabkb_params' : {
            'sigma' : sigma,
            'lam' : lam,
            'alpha' : alpha,
            'kernel' : RBF(sigma),
            'options' : opt,
            'N' : N,
            'hmax' : hmax
        },
        'bkb_params' : {
            'sigma' : sigma_disc,
            'lam' : lam,
            'kernel' : RBF(sigma_disc),
            'fnorm' : fnorm,
            'noise_variance' : lam**2,
            'delta' : delta,
            'random_state' : np.random.RandomState(12),
            'init_samples' : 5,
            'arm_set' : arm_set,
            'qbar' : qbar
        },
        'gpucb_params' :{
            'sigma' : sigma_disc,
            'lam' : lam,
            'kernel' : RBF(sigma_disc),
            'fnorm' : fnorm,
            'noise_variance' : lam**2,
            'delta' : delta,
            'a' : 0.1,
            'init_samples' : 5,
            'arm_set' : arm_set,
            'b' : 0.5,
            'random_state': np.random.RandomState(12),
            'r' : 1.01
        },
        'adagpucb_params':{
            'sigma' : sigma,
            'lam' : lam,
            'C1' : C1,
            'g' : gfun,
            'v_1' :  v_1,
            'rho' : rho,
            'delta' : delta,
            'fnorm' : fnorm,
            'N' : N,
            'hmax' : hmax
        }

    }
    return config

def get_trid2_config():
    sigma = 1.50 #0.5
    sigma_disc = 1.5
    lam = 1e-3
    C1 = 1.0#5e-6 #6e-6
    N = 5
    v_1 =  N * np.sqrt(1/2)
    rho = N ** (-1/np.sqrt(2))
    hmax = 7
    gfun = lambda x : (1/sigma) * x
    opt = OptimizerOptions(gfun, v_1 = v_1, rho = rho,\
        sigma = sigma, lam = lam, noise_var=lam**2, delta=delta,\
        fnorm=fnorm, qbar=qbar, seed=seed)
    arm_set = np.array(
        list(it.product(*[np.linspace(d[0], d[1], 15) for d in tri2_fun.search_space]))
    ).reshape(-1, 2)
    config = {
        'trials' : trials,
        'T' : T,
        'fun' : tri2_fun,
        'nfree_fun' : Trid(2),
        'search_space': tri2_fun.search_space,
        'adabkb_params' : {
            'sigma' : sigma,
            'lam' : lam,
            'alpha' : alpha,
            'kernel' : RBF(sigma),
            'options' : opt,
            'N' : N,
            'hmax' : hmax
        },
        'bkb_params' : {
            'sigma' : sigma_disc,
            'lam' : lam,
            'kernel' : RBF(sigma_disc),
            'fnorm' : fnorm,
            'noise_variance' : lam**2,
            'delta' : delta,
            'random_state' : np.random.RandomState(12),
            'init_samples' : 3,
            'arm_set' : arm_set,
            'qbar' : qbar
        },
        'gpucb_params' :{
            'sigma' : sigma_disc,
            'lam' : lam,
            'kernel' : RBF(sigma_disc),
            'fnorm' : fnorm,
            'noise_variance' : lam**2,
            'delta' : delta,
            'a' : 0.1,
            'init_samples' : 3,
            'arm_set' : arm_set,
            'b' : 0.5,
            'random_state': np.random.RandomState(12),
            'r' : 1.01
        },
        'adagpucb_params':{
            'sigma' : sigma,
            'lam' : lam,
            'C1' : C1,
            'g' : gfun,
            'v_1' :  v_1,
            'rho' : rho,
            'delta' : delta,
            'fnorm' : fnorm,
            'N' : N,
            'hmax' : hmax
        }

    }
    return config

def get_hartmann3_config():
    sigma = 0.50 #0.5
    lam = 1e-3
    C1 = 1.0#5e-6 #6e-6
    N = 3
    v_1 =  N * np.sqrt(1/3)
    rho = N ** (-1/np.sqrt(3))
    hmax = 7
    gfun = lambda x : (1/sigma) * x
    opt = OptimizerOptions(gfun, v_1 = v_1, rho = rho,\
        sigma = sigma, lam = lam, noise_var=lam**2, delta=delta,\
        fnorm=fnorm, qbar=qbar, seed=seed)
    arm_set = np.array(
        list(it.product(*[np.linspace(d[0], d[1], 15) for d in hart3_fun.search_space]))
    ).reshape(-1, 3)
    config = {
        'trials' : trials,
        'T' : T,
        'fun' : hart3_fun,
        'nfree_fun' : Hartmann3(),
        'search_space': hart3_fun.search_space,
        'adabkb_params' : {
            'sigma' : sigma,
            'lam' : lam,
            'alpha' : alpha,
            'kernel' : RBF(sigma),
            'options' : opt,
            'N' : N,
            'hmax' : hmax + 1
        },
        'bkb_params' : {
            'sigma' : sigma,
            'lam' : lam,
            'kernel' : RBF(sigma),
            'fnorm' : fnorm,
            'noise_variance' : lam**2,
            'delta' : delta,
            'random_state' : np.random.RandomState(12),
            'init_samples' : 3,
            'arm_set' : arm_set,
            'qbar' : qbar
        },
        'gpucb_params' :{
            'sigma' : sigma,
            'lam' : lam,
            'kernel' : RBF(sigma),
            'fnorm' : fnorm,
            'noise_variance' : lam**2,
            'delta' : delta,
            'a' : 0.1,
            'init_samples' : 3,
            'arm_set' : arm_set,
            'b' : 0.5,
            'random_state': np.random.RandomState(12),
            'r' : 1.01
        },
        'adagpucb_params':{
            'sigma' : sigma,
            'lam' : lam,
            'C1' : C1,
            'g' : gfun,
            'v_1' :  v_1,
            'rho' : rho,
            'delta' : delta,
            'fnorm' : fnorm,
            'N' : N,
            'hmax' : hmax
        }

    }
    return config


def get_trid4_config():
    sigma = 10.75
    sigma_disc = 10.75
    lam = 1e-3
    C1 = 1.0#5e-6 #6e-6
    N = 13
    v_1 =  2 * np.sqrt(1/4)
    rho = 2 ** (-1/np.sqrt(4))
    hmax = 7 #int(np.log(T)/(2*alpha*np.log(1/rho)))
    gfun = lambda x : (1/sigma) * x
    opt = OptimizerOptions(gfun, v_1 = v_1, rho = rho,\
        sigma = sigma, lam = lam, noise_var=lam**2, delta=delta,\
        fnorm=fnorm, qbar=qbar, seed=seed)
    arm_set = np.array(
        list(it.product(*[np.linspace(d[0], d[1], 15) for d in tri4_fun.search_space]))
    ).reshape(-1, 4)
    config = {
        'trials' : trials,
        'T' : T,
        'fun' : tri4_fun,
        'nfree_fun' : Trid(4),
        'search_space': tri4_fun.search_space,
        'adabkb_params' : {
            'sigma' : sigma,
            'lam' : lam,
            'alpha' : alpha,
            'kernel' : RBF(sigma),
            'options' : opt,
            'N' : N,
            'hmax' : hmax
        },
        'bkb_params' : {
            'sigma' : sigma_disc,
            'lam' : lam,
            'kernel' : RBF(sigma_disc),
            'fnorm' : fnorm,
            'noise_variance' : lam**2,
            'delta' : delta,
            'random_state' : np.random.RandomState(12),
            'init_samples' : int(np.sqrt(T)),
            'arm_set' : arm_set,
            'qbar' : qbar
        },
        'gpucb_params' :{
            'sigma' : sigma_disc,
            'lam' : lam,
            'kernel' : RBF(sigma_disc),
            'fnorm' : fnorm,
            'noise_variance' : lam**2,
            'delta' : delta,
            'a' : 0.1,
            'init_samples' : int(np.sqrt(T)),
            'arm_set' : arm_set,
            'b' : 0.5,
            'random_state': np.random.RandomState(12),
            'r' : 1.01
        },
        'adagpucb_params':{
            'sigma' : sigma,
            'lam' : lam,
            'C1' : C1,
            'g' : gfun,
            'v_1' :  v_1,
            'rho' : rho,
            'delta' : delta,
            'fnorm' : fnorm,
            'N' : 5,
            'hmax' : hmax ## + 2
        }

    }
    return config

ack5_fun = Ackley(5, (0.01, np.random.RandomState(seed)))

def get_ackley5_config():
    sigma = 5.0
    sigma_disc = 50.0
    lam = 1e-3
    C1 = 1.0
    N = 3
    v_1 =  N * np.sqrt(1/3)
    rho = N ** (-1/np.sqrt(3))
    hmax = 6 #int(np.log(T)/(2*alpha*np.log(1/rho)))
    gfun = lambda x : (1/sigma) * x
    opt = OptimizerOptions(gfun, v_1 = v_1, rho = rho,\
        sigma = sigma, lam = lam, noise_var=lam**2, delta=delta,\
        fnorm=fnorm, qbar=qbar, seed=seed)
    arm_set = np.array(
        list(it.product(*[np.linspace(d[0], d[1], 10) for d in ack5_fun.search_space]))
    ).reshape(-1, 5)
    config = {
        'trials' : trials,
        'T' : T,
        'fun' : ack5_fun,
        'nfree_fun' : Ackley(5),
        'search_space': ack5_fun.search_space,
        'adabkb_params' : {
            'sigma' : sigma,
            'lam' : lam,
            'alpha' : alpha,
            'kernel' : RBF(sigma),
            'options' : opt,
            'N' : N,
            'hmax' : hmax + 1
        },
        'bkb_params' : {
            'sigma' : sigma_disc,
            'lam' : lam,
            'kernel' : RBF(sigma_disc),
            'fnorm' : fnorm,
            'noise_variance' : lam**2,
            'delta' : delta,
            'random_state' : np.random.RandomState(12),
            'init_samples' : 2,
            'arm_set' : arm_set,
            'qbar' : qbar
        },
        'gpucb_params' :{
            'sigma' : sigma_disc,
            'lam' : lam,
            'kernel' : RBF(sigma_disc),
            'fnorm' : fnorm,
            'noise_variance' : lam**2,
            'delta' : delta,
            'a' : 0.1,
            'init_samples' : 2,#int(np.sqrt(T)),
            'arm_set' : arm_set,
            'b' : 0.5,
            'random_state': np.random.RandomState(12),
            'r' : 2.01
        },
        'adagpucb_params':{
            'sigma' : sigma,
            'lam' : lam,
            'C1' : C1,
            'g' : gfun,
            'v_1' :  v_1,
            'rho' : rho,
            'delta' : delta,
            'fnorm' : fnorm,
            'N' : N,
            'hmax' : hmax
        }

    }
    return config

lev6_fun = Levy(6, (0.01, np.random.RandomState(seed)))
def get_levy6_config():
    sigma = 5.0
    lam = 1e-3
    C1 = 1.0
    N = 5
    d = 6
    v_1 =  N * np.sqrt(1/d)
    rho = N ** (-1/np.sqrt(d))
    hmax = 7#6 #int(np.log(T)/(2*alpha*np.log(1/rho)))
    gfun = lambda x : (1/sigma) * x
    opt = OptimizerOptions(gfun, v_1 = v_1, rho = rho,\
        sigma = sigma, lam = lam, noise_var=lam**2, delta=delta,\
        fnorm=fnorm, qbar=qbar, seed=seed)
    arm_set = np.array(
        list(it.product(*[np.linspace(d[0], d[1], 10) for d in lev6_fun.search_space]))
    ).reshape(-1, d)
    config = {
        'trials' : trials,
        'T' : T,
        'fun' : lev6_fun,
        'nfree_fun' : Levy(6),
        'search_space': lev6_fun.search_space,
        'adabkb_params' : {
            'sigma' : sigma,
            'lam' : lam,
            'alpha' : alpha,
            'kernel' : RBF(sigma),
            'options' : opt,
            'N' : N,
            'hmax' : hmax 
        },
        'bkb_params' : {
            'sigma' : sigma,
            'lam' : lam,
            'kernel' : RBF(sigma),
            'fnorm' : fnorm,
            'noise_variance' : lam**2,
            'delta' : delta,
            'random_state' : np.random.RandomState(12),
            'init_samples' : 3,#int(np.sqrt(T)),
            'arm_set' : arm_set,
            'qbar' : qbar
        },
        'gpucb_params' :{
            'sigma' : sigma,
            'lam' : lam,
            'kernel' : RBF(sigma),
            'fnorm' : fnorm,
            'noise_variance' : lam**2,
            'delta' : delta,
            'a' : 0.1,
            'init_samples' : 3,#int(np.sqrt(T)),
            'arm_set' : arm_set,
            'b' : 0.5,
            'random_state': np.random.RandomState(12),
            'r' : 1.01
        },
        'adagpucb_params':{
            'sigma' : sigma,
            'lam' : lam,
            'C1' : C1,
            'g' : gfun,
            'v_1' :  v_1,
            'rho' : rho,
            'delta' : delta,
            'fnorm' : fnorm,
            'N' : N,
            'hmax' : hmax
        }

    }
    return config

ras8_fun = Rastrigin(8, (0.01, np.random.RandomState(seed)))
def get_ras8_config():
    sigma = 7.0
    sigma_dist = 7.0
    lam = 1e-3
    C1 = 1.0
    N = 3
    d = 8
    v_1 =  N * np.sqrt(1/d)
    rho = N ** (-1/np.sqrt(d))
    hmax = 10#6 #int(np.log(T)/(2*alpha*np.log(1/rho)))
    gfun = lambda x : (1/sigma) * x
    opt = OptimizerOptions(gfun, v_1 = v_1, rho = rho,\
        sigma = sigma, lam = lam, noise_var=lam**2, delta=delta,\
        fnorm=fnorm, qbar=qbar, seed=seed)
    arm_set = np.array(
        list(it.product(*[np.linspace(d[0], d[1], 5) for d in ras8_fun.search_space]))
    ).reshape(-1, d)
    config = {
        'trials' : trials,
        'T' : T,
        'fun' : ras8_fun,
        'nfree_fun' : Rastrigin(8),
        'search_space': ras8_fun.search_space,
        'adabkb_params' : {
            'sigma' : sigma,
            'lam' : lam,
            'alpha' : alpha,
            'kernel' : RBF(sigma),
            'options' : opt,
            'N' : N,
            'hmax' : hmax 
        },
        'bkb_params' : {
            'sigma' : sigma_dist,
            'lam' : lam,
            'kernel' : RBF(sigma_dist),
            'fnorm' : fnorm,
            'noise_variance' : lam**2,
            'delta' : delta,
            'random_state' : np.random.RandomState(12),
            'init_samples' : 2,
            'arm_set' : arm_set,
            'qbar' : qbar
        },
        'gpucb_params' :{
            'sigma' : sigma_dist,
            'lam' : lam,
            'kernel' : RBF(sigma_dist),
            'fnorm' : fnorm,
            'noise_variance' : lam**2,
            'delta' : delta,
            'a' : 0.1,
            'init_samples' : 2,
            'arm_set' : arm_set,
            'b' : 0.5,
            'random_state': np.random.RandomState(12),
            'r' : 1.01
        },
        'adagpucb_params':{
            'sigma' : sigma,
            'lam' : lam,
            'C1' : C1,
            'g' : gfun,
            'v_1' :  v_1,
            'rho' : rho,
            'delta' : delta,
            'fnorm' : fnorm,
            'N' : N,
            'hmax' : hmax
        }

    }
    return config


dix10_fun = DixonPrice(10, (0.01, np.random.RandomState(seed)))
def get_dix10_config():
    sigma = 2.0
    sigma_disc = 7.70
    lam = 1e-3
    C1 = 1.0
    N = 5
    d = 10
    v_1 =  N * np.sqrt(1/d)
    rho = N ** (-1/np.sqrt(d))
    hmax = 10#6 #int(np.log(T)/(2*alpha*np.log(1/rho)))
    gfun = lambda x : (1/sigma) * x
    opt = OptimizerOptions(gfun, v_1 = v_1, rho = rho,\
        sigma = sigma, lam = lam, noise_var=lam**2, delta=delta,\
        fnorm=fnorm, qbar=qbar, seed=seed)
    arm_set = np.array(
        list(it.product(*[np.linspace(d[0], d[1], 5) for d in dix10_fun.search_space]))
    ).reshape(-1, d)
    config = {
        'trials' : trials,
        'T' : T,
        'fun' : dix10_fun,
        'nfree_fun' : DixonPrice(10),
        'search_space': dix10_fun.search_space,
        'adabkb_params' : {
            'sigma' : sigma,
            'lam' : lam,
            'alpha' : alpha,
            'kernel' : RBF(sigma),
            'options' : opt,
            'N' : N,
            'hmax' : hmax 
        },
        'bkb_params' : {
            'sigma' : sigma_disc,
            'lam' : lam,
            'kernel' : RBF(sigma_disc),
            'fnorm' : fnorm,
            'noise_variance' : lam**2,
            'delta' : 0.015,
            'random_state' : np.random.RandomState(12),
            'init_samples' : 3,#int(np.sqrt(T)),
            'arm_set' : arm_set,
            'qbar' : qbar
        },
        'gpucb_params' :{
            'sigma' : sigma_disc,
            'lam' : lam,
            'kernel' : RBF(sigma_disc),
            'fnorm' : fnorm,
            'noise_variance' : lam**2,
            'delta' : delta,
            'a' : 0.1,
            'init_samples' : 3,#int(np.sqrt(T)),
            'arm_set' : arm_set,
            'b' : 0.5,
            'random_state': np.random.RandomState(12),
            'r' : 1.01
        },
        'adagpucb_params':{
            'sigma' : sigma,
            'lam' : lam,
            'C1' : C1,
            'g' : gfun,
            'v_1' :  v_1,
            'rho' : rho,
            'delta' : delta,
            'fnorm' : fnorm,
            'N' : N,
            'hmax' : hmax
        }

    }
    return config

def execute_experiments(configs):
    for config in configs:
      #  adabkb_test(config) 
      #  adagpucb_test(config)
        bkb_test(config)
        gpucb_test(config)
      


if __name__ == '__main__':
    os.makedirs("./out/", exist_ok=True)
    
    #branin_config = get_branin_config()
    #ackley2_config = get_ackley2_config()
    #beale_config = get_beale_config()
    #bohachevsky_config = get_bohachevsky_config()
    #shekel_config = get_shekel4_config()
    #rosenbrock2_config = get_rosenbrock2_config()
    #trid2_config = get_trid2_config()
    #trid4_config = get_trid4_config()
    #lev6_config = get_levy6_config()
    #ras8_config = get_ras8_config()
    #dix10_config = get_dix10_config()
    ack5_config = get_ackley5_config()

    execute_experiments ([
    #   branin_config, 
    #   ackley2_config, 
    #   beale_config, 
    #   bohachevsky_config, 
    #   rosenbrock2_config, 
    #   shekel_config, 
    #    trid4_config,
        ack5_config,
    #    lev6_config,
    #    ras8_config,
    #dix10_config
     ])
