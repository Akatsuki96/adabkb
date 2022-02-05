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


def plot_adabkb_data(path, gmin):
    DELIM = "-----"
    with open(path, "r") as f:
        parts = f.read().split(DELIM)[:-1]
    regrets, times, ls_size, estop = [], [], [], []
    for part in parts[:1]:
        reg, tm, lsize = [], [], []
        es = -1
        c = 0
        for line in part.split("\n")[:-1]:
            ln = line.split(",")
            if ln[0] != '':
                reg.append(float(ln[0]) - gmin)
                tm.append(float(ln[1]))
                lsize.append(int(ln[2]))
                if es == -1 and ln[4]=="True":
                    es = c
            c+=1
       # print("ES: ",es)
        estop.append(es)
        regrets.append(reg)
        times.append(tm)
        ls_size.append(lsize)
    csize = np.array(ls_size).mean(axis=0)
    csize_conf = mean_confidence_interval(np.array(ls_size))#compute_cumulative_lsize(np.array(ls_size))
    avgreg, avgreg_conf = compute_average_regret(np.array(regrets))
    ctimes, ctimes_conf = compute_cumulative_time(np.array(times))
    mean_estop = np.array(estop).mean()
    return avgreg, avgreg_conf, ctimes, ctimes_conf, csize, csize_conf, mean_estop#, csize_conf, mean_estop


def plot_adagpucb_data(path, gmin):
    DELIM = "-----"
    with open(path, "r") as f:
        parts = f.read().split(DELIM)[:-1]
    regrets, times, ls_size = [], [], []
    for part in parts[-2:-1]:
        reg, tm, lsize = [], [], []

        for line in part.split("\n")[:-1]:
            ln = line.split(",")
            if ln[0] != '':
                reg.append(float(ln[0]) - gmin)
                tm.append(float(ln[1]))
                lsize.append(int(ln[2]))
       # print("ES: ",es)
        regrets.append(reg)
        times.append(tm)
        ls_size.append(lsize)
    csize = np.array(ls_size).mean(axis=0)
    csize_conf = mean_confidence_interval(np.array(ls_size)) #compute_cumulative_lsize(np.array(ls_size))
    avgreg, avgreg_conf = compute_average_regret(np.array(regrets))
    ctimes, ctimes_conf = compute_cumulative_time(np.array(times))
    return avgreg, avgreg_conf, ctimes, ctimes_conf, csize, csize_conf


def plot_gpucb_bkb_data(path, gmin):
    DELIM = "-----"
    with open(path, "r") as f:
        parts = f.read().split(DELIM)[:-1]
    regrets, times = [], []
    for part in parts:
        reg, tm, lsize = [], [], []

        for line in part.split("\n")[:-1]:
            ln = line.split(",")
            if ln[0] != '':
                reg.append(float(ln[0]) - gmin)
                tm.append(float(ln[1]))
        regrets.append(reg)
        times.append(tm)
    regrets = to_minlen(np.array(regrets))
    times = to_minlen(np.array(times))
    avgreg, avgreg_conf = compute_average_regret(regrets)
    ctimes, ctimes_conf = compute_cumulative_time(times)
    return avgreg, avgreg_conf, ctimes, ctimes_conf


def plot_ris(title, fun_name, datas, path, xlab, ylab, out, loc, cut=None, yscale = None):
    fig, ax = plt.subplots()
    ax.set_title("{}: {}".format(title, fun_name))
    ax.set_xlabel(xlab)#"$t$")
    ax.set_ylabel(ylab)#"avg $R_t$")
    if yscale is not None:
        ax.set_yscale(yscale)
    for (label, data, conf) in datas:
        ax.plot(range(len(data)), data, '-', c=COLORS[label], label=label)
        if conf is not None:
            ax.fill_between(range(len(data)), conf[1], conf[0], alpha=0.6, color=COLORS[label])
    if cut is not None and cut>0:
        ax.axvline(x=cut,linestyle='--',color="red", alpha=0.4)
    ax.legend(loc=loc)#"upper right")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    plt.savefig(path + out, transparent=True, bbox_inches="tight")#"/average_regret.png")
    plt.close(fig)

def plot_regret(fun_name, datas, cut, path):
    plot_ris("Average Regret", fun_name, datas, path, "$t$", "avg $R_t$", "/average_regret.pdf", "upper right", cut)

def plot_ctime(fun_name, datas, cut, path):
    plot_ris("Cumulative Time", fun_name, datas, path, "$t$", "time (s)", "/cumulative_time.pdf", "lower right", cut, "log")

def plot_lset(fun_name, datas, cut, path):
    plot_ris("Leaf set Size", fun_name, datas, path, "$t$", "$|L_t|$", "/leafset_size.pdf", "upper right", cut, "log")


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("funname", type=str)
    parser.add_argument("path", type=str)
    parser.add_argument("global_min", type=float)
    args = parser.parse_args()
    adabkb_results = plot_adabkb_data(args.path + "/{}/trace.log".format("AdaBKB"), args.global_min)
  #  adagpucb_results = plot_adagpucb_data(args.path + "/{}/trace.log".format("AdaGPUCB"), args.global_min)
  #  bkb_results = plot_gpucb_bkb_data(args.path + "/{}/trace.log".format("BKB"), args.global_min)
  #  gpucb_results = plot_gpucb_bkb_data(args.path + "/{}/trace.log".format("GPUCB"), args.global_min)

    print("gmin: {}".format(args.global_min))
    
    reg_datas =[
#        ("adagpucb", adagpucb_results[0], adagpucb_results[1]),
#        ("bkb", bkb_results[0], bkb_results[1]),
#        ("gpucb", gpucb_results[0], gpucb_results[1]),
        ("adabkb", adabkb_results[0], adabkb_results[1])
    ]
    time_data =[
 #       ("adagpucb", adagpucb_results[2], adagpucb_results[3]),
 #       ("bkb", bkb_results[2], bkb_results[3]),
 #       ("gpucb", gpucb_results[2], gpucb_results[3]),
        ("adabkb", adabkb_results[2], adabkb_results[3])
    ]
    lset_data = [
 #       ("adagpucb", adagpucb_results[4], adagpucb_results[5]),
        ("adabkb", adabkb_results[4], adabkb_results[5])
    ]
    #def plot_ris(title, fun_name, datas, cut, path, xlab, ylab, out, loc, yscale = None):
    plot_regret(args.funname, reg_datas, adabkb_results[-1], args.path)
    plot_ctime(args.funname, time_data, adabkb_results[-1], args.path)
    plot_lset(args.funname, lset_data, adabkb_results[-1], args.path)
