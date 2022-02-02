import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import argparse as ap
import seaborn as sns

font_scale = 1.50
line_width = 4
marker_size = 6 

sns.reset_defaults()
sns.set_context("notebook", font_scale=font_scale,
                rc={"lines.linewidth": line_width,
                    "lines.markersize": marker_size}
                )
sns.set_style("whitegrid")

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

def get_adabkb_data(path, gmin):
    DELIM = "-----"
    with open(path, "r") as f:
        parts = f.read().split(DELIM)[:-1]
    regrets, times, ls_size, estop = [], [], [], []
    for part in parts:
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




def get_rndbkb_data(path, gmin):
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
    for (label, data, conf, color) in datas:
        ax.plot(range(len(data)), data, '-', c=color, label=label)
        if conf is not None:
            ax.fill_between(range(len(data)), conf[1], conf[0], alpha=0.3, color=color)
    if cut is not None and cut>0:
        ax.axvline(x=cut,linestyle='--',color="red", alpha=0.4)
    ax.legend(loc=loc)#"upper right")
    fig.tight_layout()
    plt.savefig(path + out, transparent=True, bbox_inches="tight")#"/average_regret.pdf")
    plt.close(fig)

def plot_regret(fun_name, datas, cut, path):
    plot_ris("Average Regret", fun_name, datas, path, "$t$", "avg $R_t$", "/average_regret.pdf", "best", cut, "log")

def plot_ctime(fun_name, datas, cut, path):
    plot_ris("Cumulative Time", fun_name, datas, path, "$t$", "time (s)", "/cumulative_time.pdf", "lower right", cut, "log")


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("funname", type=str)
    parser.add_argument("path", type=str)
    parser.add_argument("global_min", type=float)
    parser.add_argument("max_points", type=float)

    args = parser.parse_args()

    adabkb_results = get_adabkb_data(args.path + "/{}/trace.log".format("AdaBKB"), args.global_min)
    
    print("gmin: {}".format(args.global_min))
    
    reg_datas =[
   #     ("adabkb", adabkb_results[0], adabkb_results[1], "orange"),
#        ("bkb", bkb_results[0], bkb_results[1]),
    ]
    time_data =[
#        ("adabkb", adabkb_results[2], adabkb_results[3], "orange"),
#        ("bkb", bkb_results[2], bkb_results[3]),
    ]

    max_size = args.max_points
    sizes = [int(x) for x in np.linspace(10, np.sqrt(max_size), 5)]

    colors = ["green", "blue", "purple", "brown", "gray", "black"]

    i=0
    for size in sizes:
        rndbkb_results = get_rndbkb_data(args.path + "/{}_{}/trace.log".format("RNDBKB", size), args.global_min)
        reg_datas.append(("rndbkb[size={}]".format(size), rndbkb_results[0], rndbkb_results[1], colors[i]))
        time_data.append(("rndbkb[size={}]".format(size), rndbkb_results[2], rndbkb_results[3], colors[i]))
        i+=1

    reg_datas.append( ("adabkb", adabkb_results[0], adabkb_results[1], "orange"))
    time_data.append( ("adabkb", adabkb_results[2], adabkb_results[3], "orange"))

    plot_regret(args.funname, reg_datas, adabkb_results[-1], args.path)
    plot_ctime(args.funname, time_data, adabkb_results[-1], args.path)
