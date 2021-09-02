from batchbkb.bkb_lib import BKB
from sklearn.gaussian_process.kernels import RBF
from adabkb import AdaBKB
from adabkb.options import OptimizerOptions
from benchmark_functions import Ackley, Hartmann3, Hartmann6, Levy
from numpy.random import RandomState

import time
import numpy as np
import matplotlib.pyplot as plt

import os

def plot_comparison(title, xlabel, ylabel, adabkb, rbkbs):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.plot(range(adabkb.shape[0]), adabkb, '-',label="adabkb")
    for (lab, rbkb) in rbkbs:
        ax.plot(range(rbkb.shape[0]), rbkb, '-', label=lab)
    ax.legend(loc="upper right")

def plot_regrets(adabkb_regret, random_bkb_regrets):
    plot_comparison("Average regret", "time step", "average regret", adabkb_regret, random_bkb_regrets)

def plot_ctime(adabkb_ctime, random_bkb_ctime):
    plot_comparison("Cumulative time", "time step", "time(s)", adabkb_ctime, random_bkb_ctime)


rnd_state = RandomState(42)
lam = 0.01
d = 2
sigma = 2.0
noise_var = lam**2
N = 2
F = 1.0
qbar = 2.0
delta = 1e-5
fun = Hartmann3((lam, rnd_state)) #Ackley(d, (lam, rnd_state))#Levy(d, (lam, rnd_state)) 
main_path = "./out/{}".format(fun.name)
T = 700
hmax = 50
os.makedirs(main_path, exist_ok=True)
with open(main_path + "/experiment_info.log", "w") as f:
    f.write("[++] function: {}\n".format(fun.name))
    f.write("\t[--] d: {}\n".format(fun.search_space.shape[0]))
    f.write("\t[--] minimizers: {}\n".format(fun.global_min[0]))
    f.write("\t[--] minimum: {}\n".format(fun.global_min[1]))
    f.write("\t[--] search space: {}\n".format(fun.search_space))
    f.write("[++] parameters\n")
    f.write("\t[--] sigma: {}\n".format(sigma))
    f.write("\t[--] lambda: {} (variance: {})\n".format(lam, noise_var))
    f.write("\t[--] |f|: {}\tqbar: {}\n".format(F,qbar))
    f.write("\t[--] delta: {}\n".format(delta))
    f.write("\t[--] T: {}\thmax: {}\n".format(T, hmax))

def generate_random_sspace(search_space, nrand, state):
    return state.rand(nrand, search_space.shape[0]) * (search_space[:,1] - search_space[:, 0]) + (search_space[:, 0])

def execute_adabkb(fun):
    gfun = lambda x : (1/sigma) * x 
    v_1 = N * np.sqrt(2)#fun.search_space.shape[0])
    rho = N**(- 1/2) 
    os.makedirs(main_path + "/adabkb", exist_ok=True)
    options = OptimizerOptions(gfun, v_1 = v_1, rho = rho, lam = lam, noise_var=noise_var,\
        delta=delta, fnorm=F, qbar=qbar, seed=42)
    adabkb = AdaBKB(dot_fun=RBF(sigma), options=options)
    #    def initialize(self, target_fun, search_space, N : int = 2, budget : int = 1000, h_max : int = 100):
    adabkb.initialize(lambda x: -fun(x), fun.search_space, N, T, hmax)
    reg = []
    ctime = []
    for t in range(T):
        tm = time.time()
        xt, idx = adabkb.step()
        yt = -fun(xt)
        #print("[--] xt: {}\tyt: {}".format(xt, yt))
        adabkb.update_model(idx, yt)
        ctime.append(time.time() - tm)
        reg.append(np.abs(fun.global_min[1] + yt))
    ctime = np.cumsum(ctime)
    print(len(list(adabkb.node2idx.keys())))
    reg = np.cumsum(reg)
    reg = np.array([reg[i]/(i+1) for i in range(reg.shape[0])])
    return ctime, reg

def execute_rndbkb(fun, num_random):
    os.makedirs(main_path + "/rndbkb_{}".format(num_random), exist_ok=True)
    X = generate_random_sspace(fun.search_space, num_random, rnd_state)
    print("[--] search space shape: {}".format(X.shape))
    ind_init = rnd_state.randint(0,X.shape[0]-1,2)
    y_init = []
    for x in X[ind_init].reshape(-1, fun.search_space.shape[0]):
        y_init.append(-fun(x))
    bkb = BKB(dot = dot, lam= lam, noise_variance = noise_var, fnorm = F, delta= delta, qbar=qbar)
    regret = []
    ctime = []
    bkb.initialize(X, ind_init, y_init)
    for t in range(T):
        tm = time.time()
        ind, _ = bkb.predict()
        y = -fun(X[ind].reshape(-1,1))
        bkb.update(ind, [y], rnd_state)
        ctime.append(time.time() - tm)    
        regret.append(np.abs(fun.global_min[1] + y))
    regret = np.cumsum(regret)
    ctime = np.cumsum(ctime)
    regret = np.array([regret[i]/(i+1) for i in range(regret.shape[0])])
    return ctime, regret
dot = RBF(sigma)
ns = [20, 50, 111, 150, 200]#int(x) for x in np.linspace(10, 2000, 5)]
for s in ns:
    ctime, reg = execute_rndbkb(fun, s)
    with open(main_path + "/rndbkb_{}/log".format(s), "w") as f:
        for i in range(ctime.shape[0]):
            f.write("{},{}\n".format(ctime[i], reg[i]))

ctime, reg = execute_adabkb(fun)
with open(main_path + "/adabkb/log", "w") as f:
    for i in range(ctime.shape[0]):
        f.write("{},{}\n".format(ctime[i], reg[i]))
