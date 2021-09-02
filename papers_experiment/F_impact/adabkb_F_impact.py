from sklearn.gaussian_process.kernels import RBF
from adabkb import AdaBKB
from adabkb.options import OptimizerOptions
from benchmark_functions import Branin, Ackley, Hartmann3, Hartmann6, Levy
from numpy.random import RandomState

import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os

sns.set_theme(style="darkgrid")

def plot_comparison(title, xlabel, ylabel, adabkb, fname, legend_loc, yscale=None):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if yscale is not None:
        ax.set_yscale(yscale)
    for (lab, data) in adabkb:
        print(data)
        ax.plot(range(data.shape[0]), data, '-', label=lab)
    ax.legend(loc=legend_loc)
    plt.savefig(fname)
    plt.close(fig)

def plot_regrets(adabkb_regret, fun_name, path):
    plot_comparison("{}: average regret".format(fun_name), "$t$", "average regret", adabkb_regret, path + "/avg_regret.png", "upper right", yscale="log")

def plot_ctime(adabkb_ctime, fun_name, path):
    plot_comparison("{}: cumulative time".format(fun_name), "$t$", "time(s)", adabkb_ctime, path + "/ctime.png", "upper left")


rnd_state = RandomState(42)
lam = 0.01
d = 2
sigma = 2.0
noise_var = lam**2
N = 2
qbar = 2.0
delta = 1e-5
#Branin((lam, rnd_state)) 
fun = Hartmann3((lam, rnd_state)) #Ackley(d, (lam, rnd_state))#Levy(d, (lam, rnd_state)) 
main_path = "./out/{}".format(fun.name)
T = 700
hmax = 10
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
    f.write("\t[--] qbar: {}\n".format(qbar))
    f.write("\t[--] delta: {}\n".format(delta))
    f.write("\t[--] T: {}\thmax: {}\n".format(T, hmax))

def execute_adabkb(fun, F):
    gfun = lambda x : (1/sigma) * x 
    v_1 = N * np.sqrt(2)#fun.search_space.shape[0])
    rho = N**(- 1/2) 
    #os.makedirs(main_path + "/adabkb", exist_ok=True)
    options = OptimizerOptions(gfun, v_1 = v_1, rho = rho, lam = lam, noise_var=noise_var,\
        delta=delta, fnorm=F, qbar=qbar, seed=42)
    adabkb = AdaBKB(dot_fun=RBF(sigma), options=options)
    #    def initialize(self, target_fun, search_space, N : int = 2, budget : int = 1000, h_max : int = 100):
    adabkb.initialize(fun.search_space, N, hmax)
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

dot = RBF(sigma)

F = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

for fnorm in F:
    os.makedirs(main_path + "/adabkb_{}".format(fnorm), exist_ok=True)
    ctime, reg = execute_adabkb(fun,fnorm)
    with open(main_path + "/adabkb_{}/log".format(fnorm), "w") as f:
        for i in range(ctime.shape[0]):
            f.write("{},{}\n".format(ctime[i], reg[i]))

ctimes = []
regs = []    
for fnorm in F:
    with open(main_path + "/adabkb_{}/log".format(fnorm), "r") as f:
        read_data = f.read().split("\n")[:-1]
        tm = np.array([float(line.split(",")[0]) for line in read_data], dtype=np.float64).reshape(-1)
        r = np.array([float(line.split(",")[1]) for line in read_data], dtype=np.float64).reshape(-1)
    ctimes.append(("$F={}$".format(fnorm), tm))
    regs.append(("$F={}$".format(fnorm), r))
    
plot_regrets(regs, fun.name, main_path)
plot_ctime(ctimes, fun.name, main_path)