import os
import time
import numpy as np
import itertools as it
from adabkb.benchmark_functions.benchmark_functions import *
from adabkb.benchmark_functions.other_methods import *
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
qbar = 5
time_threshold = 600


F_values = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

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
        _extracted_from_write_exp_info_7(f, "[++] BKB params\n", config, 'bkb_params')
        _extracted_from_write_exp_info_7(
            f, "[++] Ada-BKB params\n", config, 'adabkb_params'
        )

        f.write("\t[--] N: {}\n".format(config['adabkb_params']['N']))
        f.write("\t[--] hmax: {}\n".format(config['adabkb_params']['hmax']))

# TODO Rename this here and in `write_exp_info`
def _extracted_from_write_exp_info_7(f, arg1, config, arg3):
    f.write(arg1)
    f.write("\t[--] sigma: {}\n".format(config[arg3]['sigma']))
    f.write("\t[--] lam: {}\n".format(config[arg3]['lam']))


def adabkb_test(config, F_value):

    #F_value = config['size']
    os.makedirs("./out/{}/AdaBKB_{}".format(config['fun'].name, F_value), exist_ok=True)
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
        adabkb_config['options'].fnorm = F_value
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
            write_log("./out/{}/AdaBKB_{}/trace.log".format(fun.name, F_value), nfree_fun(xt), tm, len(adabkb.leaf_set), adabkb.pruned, adabkb.estop)
            creg.append(nfree_fun(xt) - fun.global_min[1])
            ct.append(tm)
            if time.time() - tot_time > time_threshold or len(adabkb.leaf_set) < 1:
                break
        ctimes.append(ct)
        cregs.append(creg)
        end_trace("./out/{}/AdaBKB_{}/trace.log".format(fun.name,F_value))



def get_branin_config():
    sigma = 0.50
    lam = 1e-3
    N = 3
    v_1 =  N * np.sqrt(1/2)
    rho = N ** (-1/np.sqrt(2))
    hmax = 5
    gfun = lambda x : (1/sigma) * x
    opt = OptimizerOptions(GaussianKernel(sigma), v_1 = v_1, rho = rho,\
        sigma = sigma, lam = lam,  delta=delta,\
        fnorm=fnorm, qbar=qbar, seed=seed)
    return {
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
            'hmax' : hmax,
            'fnorm' : fnorm,
            'noise_variance' : lam**2,
            'delta' : delta,
            'random_state' : np.random.RandomState(12),
            'qbar' : qbar
        }

    }


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
    opt = OptimizerOptions(GaussianKernel(sigma), v_1 = v_1, rho = rho,\
        sigma = sigma, lam = lam,  delta=delta,\
        fnorm=fnorm, qbar=qbar, seed=seed)

    return {
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
        }
    }


def get_hartmann6_config():
    sigma = 0.50
    lam = 1e-3
    C1 = 2.0#5e-6 #6e-6
    N = 5
    v_1 =  N * np.sqrt(1/2)
    rho = N ** (-1/np.sqrt(2))
    hmax = int(np.log(T))#/ (2 * alpha * np.log(1/rho)))
    gfun = lambda x : (1/sigma) * x
    opt = OptimizerOptions(GaussianKernel(sigma), v_1 = v_1, rho = rho,\
        sigma = sigma, lam = lam,  delta=delta,\
        fnorm=fnorm, qbar=qbar, seed=seed)
    return {
        'trials' : trials,
        'T' : T,
        'fun' : hman6_fun,
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
        }
    }

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
    opt = OptimizerOptions(GaussianKernel(sigma), v_1 = v_1, rho = rho,\
        sigma = sigma, lam = lam,  delta=delta,\
        fnorm=fnorm, qbar=qbar, seed=seed)
    arm_set = np.array(
        list(it.product(*[np.linspace(d[0], d[1], 5) for d in ras8_fun.search_space]))
    ).reshape(-1, d)
    return {
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
        }
    }

def execute_experiments(configs):
    for config in configs:
        for f_value in F_values: 
            adabkb_test(config, f_value)
      


if __name__ == '__main__':
    os.makedirs("./out/", exist_ok=True)
    
    branin_config = get_branin_config()
    trid4_config = get_trid4_config()
    hart6_config = get_hartmann6_config()
    ras8_config = get_ras8_config()
    
    execute_experiments ([
        branin_config,
        trid4_config,
        hart6_config,
        ras8_config
    ])
