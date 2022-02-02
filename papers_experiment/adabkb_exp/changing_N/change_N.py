import os
import time
import numpy as np
import sys

sys.path.insert(0, '../')

import itertools as it
from benchmark_functions.benchmark_functions import *
from benchmark_functions.other_methods import *
from adabkb.kernels import GaussianKernel
from adabkb.options import OptimizerOptions

from adabkb.optimizer import AdaBKB
from adabkb.kernels import GaussianKernel

np.random.seed(12)

trials = 1
alpha = 1.0
T = 700
D = 6
delta = 1e-5
fnorm = 1.0
seed = 12
qbar = 5
time_threshold = 600

F = 1.0
F_values = [3, 5, 7, 9]

DELIM = "-----"

bra_fun = Branin((0.0001, np.random.RandomState(seed)))
bea_fun = Beale((0.01, np.random.RandomState(seed)))
boh_fun = Bohachevsky((0.01, np.random.RandomState(seed)))
tri4_fun = Trid(4, (0.0001, np.random.RandomState(seed))) 

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
    adabkb_config['N'] = F_value
    #adabkb_config['options'].v_1 = F_value * adabkb_config['options'].v_1
    #adabkb_config['options'].rho = F_value ** adabkb_config['options'].rho
    
    for _ in range(trials):
        ct = []
        creg = []
        tot_time = time.time()
        adabkb_config['options'].fnorm = 1.0#F_value
        adabkb_config['options'].v_1 = F_value * np.sqrt(fun.search_space.shape[0])
        adabkb_config['options'].rho = F_value **(-1/fun.search_space.shape[0]) 

        adabkb = AdaBKB(adabkb_config['options'])
        #    def initialize(self, search_space, N : int = 2, h_max : int = 100):
        adabkb.initialize(fun.search_space, adabkb_config['N'], adabkb_config['hmax'])
        for t in range(T):
            tm = time.time()
            xt, node_id = adabkb.step()
            yt = -fun(xt)
            adabkb.update_model(node_id, yt)
            tm = time.time() - tm
            print("[N = {}] t: {}/{}\txt: {}\tyt: {}\t|L_t|: {}".format(adabkb_config['N'], t, T, xt, -yt, len(adabkb.leaf_set)))
            write_log("./out/{}/AdaBKB_{}/trace.log".format(fun.name, F_value), nfree_fun(xt), tm, len(adabkb.leaf_set), adabkb.pruned, adabkb.estop)
            creg.append(nfree_fun(xt) - fun.global_min[1])
            ct.append(tm)
            if time.time() - tot_time > time_threshold or len(adabkb.leaf_set) < 1:
                break
        ctimes.append(ct)
        cregs.append(creg)
        end_trace("./out/{}/AdaBKB_{}/trace.log".format(fun.name,F_value))

def get_beale_config():
    sigma = 0.50
    lam = 1e-3
    hmax = 3
    opt = OptimizerOptions(GaussianKernel(sigma), v_1 = 1.0, rho = 1.0,\
        sigma = sigma, lam = lam,  delta=delta,\
        fnorm=fnorm, qbar=qbar, seed=seed)
    return {
        'trials' : trials,
        'T' : T,
        'fun' : bea_fun,
        'nfree_fun' : Beale(),
        'search_space': bea_fun.search_space,
        'adabkb_params' : {
            'sigma' : sigma,
            'lam' : lam,
            'alpha' : alpha,
            'kernel' : GaussianKernel(sigma),
            'options' : opt,
            'hmax' : hmax,
            'fnorm' : fnorm,
            'random_state' : np.random.RandomState(12),
        }

    }
def get_boh_config():
    sigma = 0.75 #0.5
    lam = 1e-3
    hmax = 2
    opt = OptimizerOptions(GaussianKernel(sigma), v_1 = 1.0, rho = 1.0,\
        sigma = sigma, lam = lam,  delta=delta,\
        fnorm=fnorm, qbar=qbar, seed=seed)
    return {
        'trials' : trials,
        'T' : 5000,
        'fun' : boh_fun,
        'nfree_fun' : Bohachevsky(),
        'search_space': boh_fun.search_space,
        'adabkb_params' : {
            'sigma' : sigma,
            'lam' : lam,
            'alpha' : alpha,
            'kernel' : GaussianKernel(sigma),
            'options' : opt,
            'hmax' : hmax,
            'fnorm' : fnorm,
            'random_state' : np.random.RandomState(12),
        }

    }

def get_branin_config():
    sigma = 0.50
    lam = 1e-3
    N = 3
    hmax = 6
    gfun = lambda x : np.sqrt(2)/(sigma) * x
    opt = OptimizerOptions(GaussianKernel(sigma), v_1 = 1.0, rho = 1.0,\
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
            'kernel' : GaussianKernel(sigma),
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
    sigma = 2.75
    sigma_disc = 10.75
    lam = 1e-8
    C1 = 1.0#5e-6 #6e-6
    N = 13
    v_1 =  1 * np.sqrt(1/4)
    rho = 1 ** (-1/np.sqrt(4))
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
            'kernel' : GaussianKernel(sigma),
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
    bea_config = get_beale_config()
    boh_config = get_boh_config()
    
    execute_experiments ([
        branin_config,
        trid4_config,
        bea_config,
        boh_config
    ])
