import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import argparse as ap
import seaborn as sns

font_scale = 1.70
line_width = 4
marker_size = 6 

sns.reset_defaults()
sns.set_context("notebook", font_scale=font_scale,
                rc={"lines.linewidth": line_width,
                    "lines.markersize": marker_size}
                )
sns.set_style("whitegrid")

COLORS = {
    'random' : 'black',
    'adabkb' : 'orange',
    'bkb' : 'brown',
    'gpucb' : 'blue',
    'adagpucb' : 'purple',
    'cmaes' : 'gray',
    'turbo' : 'green'
}

def to_minlen(lst, num_exp = 1):
    ris = []
    min_len = np.min([len(l) for l in lst])
    for l in lst:
        if len(l) > min_len:
            ris.append(l[:min_len])
        else:
            ris.append(l)
    return np.array(ris)#.reshape(num_exp, min_len)

def mean_confidence_interval(data, confidence=0.95):
    cinv = st.t.interval(0.95, data.shape[1]-1, loc=data.mean(axis=0), scale=data.std(axis=0))
    return cinv
    
def compute_average_regret(regrets):
    avg_regs = np.cumsum(regrets, axis=1) / np.array([np.arange(1,regrets.shape[1] + 1) for _ in range(regrets.shape[0])])
    return avg_regs.mean(axis=0), mean_confidence_interval(avg_regs)

def compute_cumulative_time(times):
    ctimes = np.cumsum(times, axis=1)
    return ctimes.mean(axis=0), mean_confidence_interval(ctimes)

def compute_cumulative_lsize(lssize):
    clssize = np.cumsum(lssize, axis=1)
    return clssize.mean(axis=0), mean_confidence_interval(clssize)


def get_adabkb_data(path, gmin, f_list):
    reg_results = []
    ctime_results = []
    for F in f_list:
        reg, ctime = [], []
        with open(path+"{}/log".format(F), "r") as f:
            parts = f.read().split("\n")[:-1]
        for line in parts:
            splitted = line.split(",")
            reg.append(float(splitted[1]) - gmin)
            ctime.append(float(splitted[0]))
        reg = np.cumsum(reg) / np.arange(1, len(reg) + 1)
        ctime = np.cumsum(ctime)
        reg_results.append(("F = {}".format(F), reg))
        ctime_results.append(("F = {}".format(F), ctime))
    return reg_results, ctime_results    



def plot_regret(path, fun, regs):
    fig, ax = plt.subplots()
    ax.set_title("{}: Average Regret".format(fun))
    ax.set_xlabel("$t$")
    ax.set_ylabel("avg $R_t$")
    for (label, reg) in regs:
        ax.plot(range(len(reg)), reg, '-', label=label) 
    ax.legend(loc = "upper right")
    ax.set_yscale("log")
    plt.savefig(path +"average_regret.png")
    plt.close(fig)

def plot_ctime(path, fun, ctimes):
    fig, ax = plt.subplots()
    ax.set_title("{}: Cumulative Time".format(fun))
    ax.set_xlabel("$t$")
    ax.set_ylabel("$s$")
    for (label, ctime) in ctimes:
        ax.plot(range(len(ctime)), ctime, '-', label=label) 
    ax.legend(loc = "lower right")
    ax.set_yscale("log")
    plt.savefig(path + "cumulative_time.png")
    plt.close(fig)

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("funname", type=str)
    parser.add_argument("path", type=str)
    parser.add_argument("global_min", type=float)
    args = parser.parse_args()
    F = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

    avg_regs, ctime = get_adabkb_data(args.path + "/adabkb_", args.global_min, F)
    plot_regret(args.path+"/", args.funname, avg_regs)
    plot_ctime(args.path+"/", args.funname, ctime)
