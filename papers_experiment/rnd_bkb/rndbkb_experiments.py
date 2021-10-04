import os
import time
import numpy as np
import itertools as it
from adabkb.benchmark_functions import *
from sklearn.gaussian_process.kernels import RBF
from adabkb.options import OptimizerOptions

from adabkb.optimizer import AdaBKB
from adabkb.other_methods import Bkb

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
hman6_fun = Hartmann6((0.01, np.random.RandomState(seed)))
ras8_fun = Rastrigin(8, (0.01, np.random.RandomState(seed)))

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


def rndbkb_test(config, arm_set_size):

    #arm_set_size = config['size']
    os.makedirs("./out/{}/RNDBKB_{}".format(config['fun'].name, arm_set_size), exist_ok=True)
    trials = config['trials']
    T = config['T']
    fun = config['fun']
    nfree_fun = config['nfree_fun']
    ctimes = []
    cregs = []
    bkb_config = config['rndbkb_params']
    search_space = fun.search_space
    rnd_state = bkb_config['random_state']
    arm_set = rnd_state.rand(arm_set_size, search_space.shape[0]) * (search_space[:, 1] - search_space[:, 0]) + search_space[:, 0]
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
            write_log("./out/{}/RNDBKB_{}/trace.log".format(fun.name, arm_set_size),\
             nfree_fun(xt), tm, 0, 0, False)
            creg.append(nfree_fun(xt) - fun.global_min[1])
            ct.append(tm)
            if time.time() - tot_time > time_threshold:
                break
        ctimes.append(ct)
        cregs.append(creg)
        end_trace("./out/{}/RNDBKB_{}/trace.log".format(fun.name, arm_set_size))

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
    
    config = {
        'trials' : trials,
        'T' : T,
        'size' : [int(x) for x in np.linspace(10, np.sqrt(15**4) , 5)],
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
        'rndbkb_params' : {
            'sigma' : sigma,
            'lam' : lam,
            'kernel' : RBF(sigma),
            'fnorm' : fnorm,
            'noise_variance' : lam**2,
            'delta' : delta,
            'random_state' : np.random.RandomState(12),
            'init_samples' : 3,
            'qbar' : qbar
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

    config = {
        'trials' : trials,
        'T' : T,
        'size' : [int(x) for x in np.linspace(10, np.sqrt(15**4) , 5)],
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
        'rndbkb_params' : {
            'sigma' : sigma_disc,
            'lam' : lam,
            'kernel' : RBF(sigma_disc),
            'fnorm' : fnorm,
            'noise_variance' : lam**2,
            'delta' : delta,
            'random_state' : np.random.RandomState(12),
            'init_samples' : 2,
            'qbar' : qbar
        }

    }
    return config


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
    #arm_set = np.array(
    #    list(it.product(*[np.linspace(d[0], d[1], 10) for d in hman_fun.search_space]))
    #).reshape(-1, 6)
    config = {
        'trials' : trials,
        'T' : T,
        'fun' : hman6_fun,
        'size' : [int(x) for x in np.linspace(10, np.sqrt(1e6), 5)],
        'nfree_fun' : Hartmann6(),
        'search_space': hman6_fun.search_space,
        'adabkb_params' : {
            'sigma' : sigma,
            'lam' : lam,
            'alpha' : alpha,
            'kernel' : RBF(sigma),
            'options' : opt,
            'N' : N,
            'hmax' : hmax
        },
        'rndbkb_params' : {
            'sigma' : sigma,
            'lam' : lam,
            'kernel' : RBF(sigma),
            'fnorm' : fnorm,
            'noise_variance' : lam**2,
            'delta' : delta,
            'random_state' : np.random.RandomState(12),
            'init_samples' : 2,
            'qbar' : qbar
        }

    }
    return config

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
        'size' : [int(x) for x in np.linspace(10, np.sqrt(5**8), 5)],
        'adabkb_params' : {
            'sigma' : sigma,
            'lam' : lam,
            'alpha' : alpha,
            'kernel' : RBF(sigma),
            'options' : opt,
            'N' : N,
            'hmax' : hmax 
        },
        'rndbkb_params' : {
            'sigma' : sigma_dist,
            'lam' : lam,
            'kernel' : RBF(sigma_dist),
            'fnorm' : fnorm,
            'noise_variance' : lam**2,
            'delta' : delta,
            'random_state' : np.random.RandomState(12),
            'init_samples' : 2,
            'qbar' : qbar
        }

    }
    return config

def execute_experiments(configs):
    for config in configs:
        adabkb_test(config)
        for size in config['size']: 
            rndbkb_test(config, size)
      


if __name__ == '__main__':
    os.makedirs("./out/", exist_ok=True)
    
    branin_config = get_branin_config()
    trid4_config = get_trid4_config()
    hart6_config = get_hartmann6_config()
    ras8_config = get_ras8_config()
    
    execute_experiments ([
     #   branin_config,
    #    trid4_config,
     #   hart6_config,
        ras8_config
    ])
