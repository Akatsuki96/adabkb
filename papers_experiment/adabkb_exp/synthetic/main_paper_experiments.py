import os
import sys

sys.path.insert(0, '../')


import time
import numpy as np
import itertools as it
from benchmark_functions import SixHumpCamel, Hartmann6, Levy, Bkb, AdaGpucb, GPUCB 
from sklearn.gaussian_process.kernels import RBF
from adabkb.options import OptimizerOptions
from adabkb.kernels import GaussianKernel

from adabkb.optimizer import AdaBKB

np.random.seed(12)

trials = 5
alpha = 1.0
T = 700
D = 6
delta = 1e-5
fnorm = 1.0
seed = 12
qbar = 3
time_threshold = 1200


DELIM = "-----"

hman_fun = Hartmann6((0.01, np.random.RandomState(seed)))
shcam_fun = SixHumpCamel((0.01, np.random.RandomState(seed)))
lev8_fun = Levy(8, (0.01, np.random.RandomState(seed)))

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
        adabkb = AdaBKB(adabkb_config['options'])
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
            if time.time() - tot_time > time_threshold:
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
    C1 = 0.50#5e-6 #6e-6
    N = 3
    v_1 =  N * np.sqrt(1/6)
    rho = N ** (-1/np.sqrt(6))
    hmax = 6#int(np.log(T))#/ (2 * alpha * np.log(1/rho)))
    gfun = lambda x : (1/sigma) * x
    opt = OptimizerOptions(GaussianKernel(sigma), v_1 = v_1, rho = rho,\
        sigma = sigma, lam = lam, delta=delta,\
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
            'kernel' : GaussianKernel(sigma),
            'options' : opt,
            'N' : N,
            'hmax' : 12
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

def get_sixhumpcamel_config():
    sigma = 0.50
    lam = 1e-3
    C1 = 1.0#5e-6 #6e-6
    N = 5
    v_1 =  N * np.sqrt(1/2)
    rho = N ** (-1/np.sqrt(2))
    hmax = 6#int(np.log(T))#/ (2 * alpha * np.log(1/rho)))
    gfun = lambda x : (1/sigma) * x
    opt = OptimizerOptions(GaussianKernel(sigma), v_1 = v_1, rho = rho,\
        sigma = sigma, lam = lam,  delta=delta,\
        fnorm=fnorm, qbar=qbar, seed=seed)
    arm_set = np.array(
        list(it.product(*[np.linspace(d[0], d[1], 15) for d in shcam_fun.search_space]))
    ).reshape(-1, 2)
    config = {
        'trials' : trials,
        'T' : T,
        'fun' : shcam_fun,
        'nfree_fun' : SixHumpCamel(),
        'search_space': shcam_fun.search_space,
        'adabkb_params' : {
            'sigma' : sigma,
            'lam' : lam,
            'alpha' : alpha,
            'kernel' : GaussianKernel(sigma),
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
    opt = OptimizerOptions(GaussianKernel(sigma), v_1 = v_1, rho = rho,\
        sigma = sigma, lam = lam,  delta=delta,\
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
            'kernel' : GaussianKernel(sigma),
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



if __name__ == '__main__':
    os.makedirs("./out/", exist_ok=True)
    
    hartmann_config = get_hartmann6_config()
    sixhumpcamel_config = get_sixhumpcamel_config()
    lev8_config = get_levy8_config()

#    write_exp_info(sixhumpcamel_config)
#    adabkb_test(sixhumpcamel_config) 
#    adagpucb_test(sixhumpcamel_config)
#    bkb_test(sixhumpcamel_config)
#    gpucb_test(sixhumpcamel_config)


#    write_exp_info(hartmann_config)
    adabkb_test(hartmann_config) 
#    adagpucb_test(hartmann_config)
#    bkb_test(hartmann_config)
#    gpucb_test(hartmann_config)


#    write_exp_info(lev8_config)
#    adabkb_test(lev8_config) 
#    adagpucb_test(lev8_config)
#    bkb_test(lev8_config)
#    gpucb_test(lev8_config)
    