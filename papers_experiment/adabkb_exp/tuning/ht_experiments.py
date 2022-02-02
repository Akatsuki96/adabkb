import os

from metric import OneMinusAUC, CERR, MSE
from dataset import CASP, HTRU2, Magic

import torch
import time
import numpy as np
import itertools as it

from falkon import Falkon, InCoreFalkon
from falkon.kernels import GaussianKernel
from falkon.options import FalkonOptions

from adabkb.optimizer import AdaBKB
from benchmark_functions.other_methods import Bkb, AdaGpucb 
from adabkb.options import OptimizerOptions
from adabkb.utils import GreedyExpansion

from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split


DELIM = "--------------\n"
T = 700
qbar = 3
seed = 12
trials = 5
time_threshold = 1200

def get_falkon_options(sigma, lam, M):
    return {
        'kernel': GaussianKernel(sigma), 
        'penalty': lam,
        'seed' : seed,
        'maxiter' : 10, 
        'M': M, 
        'options': FalkonOptions(cg_tolerance=1e-4, keops_active="no")
        }
    
def build_falkon(sigma, lam, M, incore=True):
    flk_options = get_falkon_options(sigma, lam, M)
    if incore:
        return InCoreFalkon(**flk_options)
    return Falkon(**flk_options)

def test_config(Xtr, ytr, Xts, yts, config, metric, incore=True):
    flk = build_falkon(torch.from_numpy(config[0]), config[1], config[2], incore)
    flk.fit(Xtr.cuda(), ytr.cuda())
    y_pred = flk.predict(Xts.cuda())
    return metric(yts, y_pred.cpu())

def get_optimizer_options(gfun, lam, delta, fnorm, qbar, seed):
    options = {
        'expand_fun' : GreedyExpansion(),
        'gfun' : gfun,
        'lam' : lam,
        'noise_var' : lam**2,
        'delta' : delta,
        'fnorm' : fnorm,
        'qbar' : qbar,
        'seed' : seed
    }
    return OptimizerOptions(**options)

def get_target_function(Xtr, ytr, metric, num_sig, flk_lam, M, incore=True):
    def target_function(x):        
        flk = build_falkon(torch.from_numpy(x[0:num_sig]), flk_lam, M, incore)

        Xtrain, Xval, Ytrain, Yval = train_test_split(Xtr, ytr, test_size=0.3, random_state=1213)

        flk.fit(Xtrain.cuda(), Ytrain.cuda())
        predictions = flk.predict(Xval.cuda())
        return -metric(Yval, predictions.cpu())
    return target_function

def adagpucb_test(dataset, config):
    adagpucb_config = config['adagpucb_options']
    out_path = "./out/{}/AdaGPUCB".format(dataset.name)
    os.makedirs(out_path, exist_ok=True)
    kernel = RBF(adagpucb_config['sigma'])
    test_err = []
    for _ in range(trials):
        opt = get_optimizer_options(adagpucb_config['g'], adagpucb_config['lam'], adagpucb_config['delta'],\
            adagpucb_config['fnorm'], qbar, seed)
        adagpucb = AdaGpucb(adagpucb_config['sigma'], adagpucb_config['g'], adagpucb_config['v_1'],\
            adagpucb_config['rho'], config['search_space'].shape[0],\
            adagpucb_config['C1'], adagpucb_config['lam'], adagpucb_config['lam']**2,\
            adagpucb_config['fnorm'], adagpucb_config['delta'])

        Xtr, ytr, Xts, yts = dataset.__split__()
        target = get_target_function(Xtr, ytr, MSE(), config['d'], config['flk_lam'], config['M'], incore=True) 
        _, xt, _ = adagpucb.run(target, target, config['search_space'], adagpucb_config['N'], T,\
             real_best=0.0, hmax = adagpucb_config['hmax'], time_threshold= time_threshold, 
             out_dir="./out/{}/AdaGPUCB/".format(dataset.name))
        with open(out_path + "/trace.log", "a") as f:
            f.write(DELIM)
        test_err.append(test_config(Xtr, ytr, Xts, yts, [xt, config['flk_lam'], config['M']], MSE(), incore=True))
    with open(out_path + "/results", "a") as f:
        f.write("AdaGPUCB: {} +/- {}".format(np.array(test_err).mean(), np.array(test_err).std()))


def adabkb_test(dataset, config):
    adabkb_config = config['adabkb_options']
    out_path = "./out/{}/AdaBKB".format(dataset.name)
    os.makedirs(out_path, exist_ok=True)
    kernel = RBF(adabkb_config['sigma'])
    test_err = []
    num_pulled = 0
    for _ in range(trials):
        opt = get_optimizer_options(adabkb_config['g'], adabkb_config['lam'], adabkb_config['delta'],\
            adabkb_config['fnorm'], qbar, seed)
        adabkb = AdaBKB(kernel, opt)
        adabkb.initialize(config['search_space'], adabkb_config['N'], adabkb_config['hmax'])
        Xtr, ytr, Xts, yts = dataset.__split__()
        target = get_target_function(Xtr, ytr, MSE(), config['d'], config['flk_lam'], config['M'], incore=True) 
        for t in range(T):
            it_time = time.time()
            xt, idx = adabkb.step()
            yt = target(xt)
            print("[--] t: {}/{}\txt: {}\tyt: {}\t|Lt|: {}\tdnode: {}".format(t, T, xt, yt, len(adabkb.leaf_set), adabkb.pulled_arms_matrix.shape[0]))
            adabkb.update_model(idx, yt)
            it_time = time.time() - it_time
            with open(out_path + "/trace.log", "a") as f:
                f.write("{},{},{},{},{}\n".format(yt, it_time, len(adabkb.leaf_set), adabkb.pruned, adabkb.estop))
        num_pulled += adabkb.pulled_arms_matrix.shape[0]
        with open(out_path + "/trace.log", "a") as f:
            f.write(DELIM)
        test_err.append(test_config(Xtr, ytr, Xts, yts, [xt, config['flk_lam'], config['M']], MSE(), incore=True))
    
    num_pulled = int(np.array(num_pulled).mean())
    with open(out_path + "/results", "a") as f:
        f.write("AdaBKB: {} +/- {}\n".format(np.array(test_err).mean(), np.array(test_err).std()))
        f.write("Num pulled: {}".format(num_pulled))
    return num_pulled

def bkb_test(dataset, config):
    bkb_config = config['bkb_options']
    out_path = "./out/{}/BKB".format(dataset.name)
    os.makedirs(out_path, exist_ok=True)
    kernel = RBF(bkb_config['sigma'])
    search_space = bkb_config['search_space']
    init_samples = bkb_config['init_samples']
    rnd_state = np.random.RandomState(seed)
    test_err = []
    for _ in range(trials):
        tot_time = time.time()
        Xtr, ytr, Xts, yts = dataset.__split__()
        target = get_target_function(Xtr, ytr, MSE(), config['d'], config['flk_lam'], config['M'], incore=True) 
        bkb = Bkb(bkb_config['lam'], kernel, bkb_config['fnorm'], bkb_config['lam']**2, bkb_config['delta'], qbar)
        Xinit = np.random.choice(range(search_space.shape[0]), init_samples, replace=False)
        Yinit = []
        for ind in Xinit:
            print("x[init]: ", search_space[ind])
            Yinit.append(target(search_space[ind]))
        bkb.initialize(search_space, Xinit, np.array(Yinit).reshape(-1), rnd_state)
        for t in range(T):
            it_time = time.time()
            idx, _ = bkb.predict()
            xt = search_space[idx[0]]
            yt = target(search_space[idx[0]])
            print("[--] t: {}/{}\txt: {}\tyt: {}".format(t, T, xt, yt))
            bkb.update(idx, [yt], rnd_state)
            it_time = time.time() - it_time
            with open(out_path + "/trace.log", "a") as f:
                f.write("{},{},{},{},{}\n".format(yt, it_time, 0, 0, False))
            if time.time() - tot_time > time_threshold:
                break
        with open(out_path + "/trace.log", "a") as f:
            f.write(DELIM)
        test_err.append(test_config(Xtr, ytr, Xts, yts, [xt, config['flk_lam'], config['M']], MSE(), incore=True))
    with open(out_path + "/results", "a") as f:
        f.write("BKB: {} +/- {}".format(np.array(test_err).mean(), np.array(test_err).std()))

def rndbkb_test(dataset, config, sz):
    bkb_config = config['rndbkb_options']
    out_path = "./out/{}/RandomBKB".format(dataset.name)
    os.makedirs(out_path, exist_ok=True)
    kernel = RBF(bkb_config['sigma'])
    search_space = bkb_config['search_space']
    init_samples = bkb_config['init_samples']
    rnd_state = np.random.RandomState(seed)
    search_space = search_space[np.random.choice(range(search_space.shape[0]), sz, replace=False)]
    test_err = []
    for _ in range(trials):
        tot_time = time.time()
        Xtr, ytr, Xts, yts = dataset.__split__()
        target = get_target_function(Xtr, ytr, MSE(), config['d'], config['flk_lam'], config['M'], incore=True) 
        bkb = Bkb(bkb_config['lam'], kernel, bkb_config['fnorm'], bkb_config['lam']**2, bkb_config['delta'], qbar)
        Xinit = np.random.choice(range(search_space.shape[0]), init_samples, replace=False)
        Yinit = []
        for ind in Xinit:
            print("x[init]: ", search_space[ind])
            Yinit.append(target(search_space[ind]))
        bkb.initialize(search_space, Xinit, np.array(Yinit).reshape(-1), rnd_state)
        for t in range(T):
            it_time = time.time()
            idx, _ = bkb.predict()
            xt = search_space[idx[0]]
            yt = target(search_space[idx[0]])
            print("[--] t: {}/{}\txt: {}\tyt: {}".format(t, T, xt, yt))
            bkb.update(idx, [yt], rnd_state)
            it_time = time.time() - it_time
            with open(out_path + "/trace.log", "a") as f:
                f.write("{},{},{},{},{}\n".format(yt, it_time, 0, 0, False))
            if time.time() - tot_time > time_threshold:
                break
        with open(out_path + "/trace.log", "a") as f:
            f.write(DELIM)
        test_err.append(test_config(Xtr, ytr, Xts, yts, [xt, config['flk_lam'], config['M']], MSE(), incore=True))
    with open(out_path + "/results", "a") as f:
        f.write("RandomBKB: {} +/- {}".format(np.array(test_err).mean(), np.array(test_err).std()))



def get_htru_config():
    flk_lam = 1e-5
    M = 1000
    sigma = 10.0
    lam = 1e-9
    d = 8
    C1 = 1.0
    hmax = 6
    N = 3
    delta = 1e-5
    fnorm = 1.0
    g = lambda x: (1/sigma) * x
    v_1 = N * np.sqrt(d)
    rho = N**(-1/d)
    search_space = np.array([[1e-5, 2.0] for _ in range(d)]).reshape(-1, 2)
    disc = np.array(list(it.product(*[np.linspace(d[0], d[1], 5) for d in search_space]))).reshape(-1, d)
    return {
        'search_space': search_space,
        'd' : d,
        'M' : M,
        'flk_lam' : flk_lam,
        'adabkb_options' : {
            'sigma' : sigma,
            'lam' : lam,
            'v_1' : v_1,
            'g' : g,
            'rho' : rho,
            'hmax' : hmax,
            'N' : N,
            'delta' : delta,
            'fnorm' : fnorm
        },
        'adagpucb_options' : {
            'sigma' : sigma,
            'lam' : lam,
            'v_1' : v_1,
            'g' : g,
            'rho' : rho,
            'C1' : C1,
            'hmax' : hmax,
            'N' : N,
            'delta' : delta,
            'fnorm' : fnorm
        },
        'bkb_options' :{
            'sigma' : sigma,
            'lam' : lam,
            'hmax' : hmax,
            'init_samples': 2,
            'search_space' : disc,
            'N' : N,
            'delta' : delta,
            'fnorm' : fnorm 
        },
        'rndbkb_options' :{
            'sigma' : sigma,
            'lam' : lam,
            'hmax' : hmax,
            'init_samples': 2,
            'search_space' : disc,
            'N' : N,
            'delta' : delta,
            'fnorm' : fnorm 
        }
    }

def get_casp_config():
    flk_lam = 1e-6
    M = 2000
    sigma = 5.0
    lam = 1e-9
    d = 9
    C1 = 1e-4
    hmax = 7
    N = 5
    delta = 1e-5
    fnorm = 1.0
    g = lambda x: (1/sigma) * x
    v_1 = N * np.sqrt(3)
    rho = N**(-1/3)
    search_space = np.array([[1e-5, 2.0] for _ in range(d)]).reshape(-1, 2)
    disc= np.array(list(it.product(*[np.linspace(d[0], d[1], 5) for d in search_space]))).reshape(-1, d)
    return {
        'search_space': search_space,
        'd' : d,
        'M' : M,
        'flk_lam' : flk_lam,
        'adabkb_options' : {
            'sigma' : sigma,
            'lam' : lam,
            'v_1' : v_1,
            'g' : g,
            'rho' : rho,
            'hmax' : hmax,
            'N' : N,
            'delta' : delta,
            'fnorm' : fnorm
        },
        'adagpucb_options' : {
            'sigma' : sigma,
            'lam' : lam,
            'v_1' : v_1,
            'g' : g,
            'rho' : rho,
            'C1' : C1,
            'hmax' : hmax,
            'N' : N,
            'delta' : delta,
            'fnorm' : fnorm
        },
        'bkb_options' :{
            'sigma' : sigma,
            'lam' : lam,
            'hmax' : hmax,
            'init_samples': 2,
            'search_space' : disc,
            'N' : N,
            'delta' : delta,
            'fnorm' : fnorm 
        },
        'rndbkb_options' :{
            'sigma' : sigma,
            'lam' : lam,
            'hmax' : hmax,
            'init_samples': 2,
            'search_space' : disc,
            'N' : N,
            'delta' : delta,
            'fnorm' : fnorm 
        }
    }

def get_magic_config():
    flk_lam = 1e-6
    M = 2000
    sigma = 5.0
    lam = 1e-9
    d = 10
    C1 = 1.0
    hmax = 6
    N = 3
    delta = 1e-5
    fnorm = 1.0
    g = lambda x: (1/sigma) * x
    v_1 = N * np.sqrt(4)
    rho = N**(-1/4)
    search_space = np.array([[1e-5, 10.0] for _ in range(d)]).reshape(-1, 2)
    disc = np.array(list(it.product(*[np.linspace(d[0], d[1], 5) for d in search_space]))).reshape(-1, d)
    return {
        'search_space': search_space,
        'd' : d,
        'M' : M,
        'flk_lam' : flk_lam,
        'adabkb_options' : {
            'sigma' : sigma,
            'lam' : lam,
            'v_1' : v_1,
            'g' : g,
            'rho' : rho,
            'hmax' : hmax,
            'N' : N,
            'delta' : delta,
            'fnorm' : fnorm
        },
        'adagpucb_options' : {
            'sigma' : sigma,
            'lam' : lam,
            'v_1' : v_1,
            'g' : g,
            'rho' : rho,
            'C1' : C1,
            'hmax' : hmax,
            'N' : N,
            'delta' : delta,
            'fnorm' : fnorm
        },
        'bkb_options' :{
            'sigma' : sigma,
            'lam' : lam,
            'hmax' : hmax,
            'init_samples': 2,
            'search_space' : disc,
            'N' : N,
            'delta' : delta,
            'fnorm' : fnorm,
        },
        'rndbkb_options' :{
            'sigma' : sigma,
            'lam' : lam,
            'hmax' : hmax,
            'init_samples': 2,
            'search_space' : disc,
            'N' : N,
            'delta' : delta,
            'fnorm' : fnorm 
        }
    }


def write_log_info(dataset, config):
    adabkb_config = config['adabkb_options']
    adagpucb_config = config['adagpucb_options']
    with open("./out/{}/exp_info".format(dataset.name), "w") as f:
        f.write("[--] search space: {}\n".format(config['search_space']))
        f.write("[--] D: {}\n".format(config['d']))
        f.write("[++] Falkon fixed params:\n")
        f.write("\t[--] lambda: {}\n".format(config['flk_lam']))
        f.write("\t[--] M: {}\n".format(config['M']))
        f.write("[++] Ada-BKB params:\n")
        f.write("\t[--] sigma: {}\n".format(adabkb_config['sigma']))
        f.write("\t[--] lam: {}\n".format(adabkb_config['lam']))
        f.write("\t[--] v1: {}\n".format(adabkb_config['v_1']))
        f.write("\t[--] rho: {}\n".format(adabkb_config['rho']))
        f.write("\t[--] hmax: {}\n".format(adabkb_config['hmax']))
        f.write("\t[--] N: {}\n".format(adabkb_config['N']))
        f.write("\t[--] delta: {}\n".format(adabkb_config['delta']))
        f.write("\t[--] fnorm: {}\n".format(adabkb_config['fnorm']))
        f.write("[++] AdaGP-UCB params:\n")
        f.write("\t[--] sigma: {}\n".format(adagpucb_config['sigma']))
        f.write("\t[--] lam: {}\n".format(adagpucb_config['lam']))
        f.write("\t[--] v1: {}\n".format(adagpucb_config['v_1']))
        f.write("\t[--] rho: {}\n".format(adagpucb_config['rho']))
        f.write("\t[--] hmax: {}\n".format(adagpucb_config['hmax']))
        f.write("\t[--] N: {}\n".format(adagpucb_config['N']))
        f.write("\t[--] delta: {}\n".format(adagpucb_config['delta']))
        f.write("\t[--] fnorm: {}\n".format(adagpucb_config['fnorm']))
        f.write("\t[--] C1: {}\n".format(adagpucb_config['C1']))

        



htru2_path = "/data/mrando/HTRU2/HTRU_2.csv"
casp_path = "/data/mrando/CASP/CASP.csv"
magic_path = "/data/mrando/Magic04/magic04.data"

htru = HTRU2(htru2_path)
casp = CASP(casp_path)
magic = Magic(magic_path)

if __name__ == '__main__':
    print("Falkon test")
    #htru_config = get_htru_config()
    #sz = adabkb_test(htru, htru_config)
    #adagpucb_test(htru, htru_config)
    #rndbkb_test(htru, htru_config, sz)
    #bkb_test(htru, htru_config)
    

    casp_config = get_casp_config()
   # sz = adabkb_test(casp, casp_config)
    adagpucb_test(casp, casp_config)
    #rndbkb_test(casp, casp_config, sz)
    #bkb_test(casp, casp_config)
    
    #magic_config = get_magic_config()
    #sz = adabkb_test(magic, magic_config)
    #adagpucb_test(magic, magic_config)
    #rndbkb_test(magic, magic_config, sz)
    #bkb_test(magic, magic_config)
    