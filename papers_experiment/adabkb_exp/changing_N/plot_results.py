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
    return st.t.interval(
        0.95, data.shape[1] - 1, loc=data.mean(axis=0), scale=data.std(axis=0)
    )
    
def compute_average_regret(regrets):
    avg_regs = np.cumsum(regrets, axis=1) / np.array([np.arange(1,regrets.shape[1] + 1) for _ in range(regrets.shape[0])])
    return avg_regs.mean(axis=0), mean_confidence_interval(avg_regs)

def compute_cumulative_time(times):
    ctimes = np.cumsum(times, axis=1)
    return ctimes.mean(axis=0), mean_confidence_interval(ctimes)


def get_rndbkb_data(path, gmin):
    DELIM = "-----"
    with open(path, "r") as f:
        parts = f.read().split(DELIM)[:-1]
    regrets, times, lset_size = [], [], []
    for part in parts:
        reg, tm, lsize = [], [], []

        for line in part.split("\n")[:-1]:
            ln = line.split(",")
            if ln[0] != '':
                reg.append(float(ln[0]) - gmin)
                tm.append(float(ln[1]))
                lsize.append(int(ln[2]))
        regrets.append(reg)
        times.append(tm)
        lset_size.append(lsize)
    regrets = to_minlen(np.array(regrets))
    times = to_minlen(np.array(times))
    lset_size = to_minlen(np.array(lset_size))

    csize = np.array(lset_size).mean(axis=0)
    csize_conf = mean_confidence_interval(lset_size)
    avgreg, avgreg_conf = compute_average_regret(regrets)
    ctimes, ctimes_conf = compute_cumulative_time(times)
    return avgreg, avgreg_conf, ctimes, ctimes_conf, csize, csize_conf


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
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    plt.savefig(path + out, transparent=True, bbox_inches="tight", pad_inches=0)#"/average_regret.pdf")
    plt.close(fig)

def plot_regret(fun_name, datas, cut, path):
    plot_ris("Average Regret", fun_name, datas, path, "$t$", "avg $R_t$", "/average_regret.pdf", "upper right", cut, "log")

def plot_ctime(fun_name, datas, cut, path):
    plot_ris("Cumulative Time", fun_name, datas, path, "$t$", "time (s)", "/cumulative_time.pdf", "lower right", cut, "log")

def plot_lset(fun_name, datas, cut, path):
    plot_ris("Leaf set Size", fun_name, datas, path, "$t$", "$|L_t|$", "/leaf_set_size.pdf", "upper right", cut, "log")


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("funname", type=str)
    parser.add_argument("path", type=str)
    parser.add_argument("global_min", type=float)

    args = parser.parse_args()
    
    reg_datas, time_data, lset_data = [], [], []
    sizes = [3, 5, 7, 9]

    colors = ["green", "blue", "purple", "brown", "gray", "black"]

    i=0
    for size in sizes:
        rndbkb_results = get_rndbkb_data(args.path + "/{}_{}/trace.log".format("AdaBKB", size), args.global_min)
        reg_datas.append(("adabkb[N={}]".format(size), rndbkb_results[0], rndbkb_results[1], colors[i]))
        time_data.append(("adabkb[N={}]".format(size), rndbkb_results[2], rndbkb_results[3], colors[i]))
        lset_data.append(("adabkb[N={}]".format(size), rndbkb_results[4], rndbkb_results[5], colors[i]))
        i+=1

    plot_regret(args.funname, reg_datas, None, args.path)
    plot_ctime(args.funname, time_data, None, args.path)
    plot_lset(args.funname, lset_data, None, args.path)
