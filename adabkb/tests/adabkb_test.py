import numpy as np
from adabkb.optimizer import AdaBKB
from adabkb.kernels import GaussianKernel
from adabkb.options import OptimizerOptions


import time
import matplotlib.pyplot as plt

print("---------------------------------------------------------------------------------------")
init_time = time.time()

fun = lambda x: -2*x**2 - 3*np.sin(x)

search_space = np.array([[-1.0, 3.0]]).reshape(-1, 2)

X = np.linspace(-3, 6, 5000).reshape(-1, 1)
y = fun(X)

model_options = {
    'kernel' : GaussianKernel(5.),
    'd' : 1,
    'lam' : 1e-3,
    'noise_variance' : 1e-4,
    'F' : 1.0,
    'qbar' : 1.0,
    'delta' : 0.0025,
    'seed' : 12
}
N = 3
v_1 = N * 2# np.sqrt(2) 
rho = 1/N
h_max = 5

T= 700

options = OptimizerOptions(model_options, v_1 = v_1, rho=rho, N = N, h_max = h_max)
optimizer = AdaBKB(search_space, options)

Xobs, yobs, lset_size = [], [], []

init_time = time.time() - init_time
print("[--] INIT TIME: {}".format(round(init_time,5)))

mean_it_time = []

for t in range(T):
    it_time = time.time()
    node, idx = optimizer.step()
    yt = fun(node.x)
    lset_size.append(optimizer.leaf_set.shape[0])
    node_idx = optimizer.get_node_idx(optimizer.leaf_set[idx])
    print("[--] xt: {}\tyt: {}\tmu: {}\tstd: {}\tlevel: {}\tbeta: {}\tVh: {}\t|L|: {}".format(node.x, yt, round(optimizer.means[node_idx], 2), round(optimizer.stds[node_idx],2), node.level, round(optimizer.beta, 5), round(optimizer.Vh[node.level],5), optimizer.leaf_set.shape[0]))
    print("[--] emb size: {}".format(optimizer.model.embedding_size))
    optimizer.update([idx], node.x, yt[0])
    it_time = time.time() - it_time
    mean_it_time.append(it_time)
    Xobs.append(node.x)
    yobs.append(yt)
    if optimizer.leaf_set.shape[0] == 0:
        print("[--] best observed: {}".format(optimizer.best_lcb))
        break
    
print("[--] Mean time per iteration: {}s".format(round(np.mean(mean_it_time), 4)))
print("[--] Tot time: {}s".format(round(np.sum(mean_it_time), 4)))

mu, var = optimizer.model.predict(X) 
    
fig, ax = plt.subplots()
ax.set_title("AdaBKB after some evaluations")
ax.plot(X, y, "-", color="red", label="$f(x)$")
ax.plot(X, mu, "-", color="black", label="$\\tilde{\mu}(x)$")
ax.plot(Xobs, yobs, "o", color="orange")
ax.fill_between(X.reshape(-1), mu - optimizer.beta * np.sqrt(var), mu + optimizer.beta * np.sqrt(var), color = "black", alpha=0.4)
ax.plot(optimizer.best_lcb[0], fun(optimizer.best_lcb[0]), 'o', c="green")
#ax.set_ylim([-5, 5])

plt.savefig("adabkb_fun.pdf")

fig, ax = plt.subplots()
ax.set_title("AdaBKB lset size")
ax.plot(range(len(lset_size)), lset_size, "-", color="black")

plt.savefig("adabkb_lset.pdf")
