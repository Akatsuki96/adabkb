import numpy as np
import matplotlib.pyplot as plt
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



direct = "./out/Ackley_30/AdaGPUCB"

expand_time = []


with open("{}/etime.log".format(direct), "r") as f:
    lines = f.readlines()
    for elem in lines:
        print(elem)
        if elem != "":
            expand_time.append(float(elem))

expand_time = np.cumsum(expand_time)

fig, ax =plt.subplots()
ax.set_title("Ackley 30: Cumulative expansion time")
ax.plot(range(expand_time.shape[0]), expand_time, "-", c="purple", label="AdaGPUCB")
ax.set_xlabel("$\\tau$")
ax.set_ylabel("time ($s$)")
ax.legend(loc="best")
plt.savefig("{}/etime.png".format("./out/Ackley_30/AdaGPUCB"), transparent=True, bbox_inches="tight", pad_inches=0)
plt.close(fig)