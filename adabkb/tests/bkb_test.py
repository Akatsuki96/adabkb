import numpy as np
from adabkb.surrogate import BKB
from adabkb.kernels import GaussianKernel

import matplotlib.pyplot as plt

fun = lambda x: np.sin(x) + np.cos(3*x)

X = np.linspace(-3, 6, 5000).reshape(-1, 1)
y = fun(X)


Xb = np.array([5.0]).reshape(-1, 1)
yb = fun(Xb)

#    def __init__(self, kernel, lam : float = 1.0, noise_variance : float = 1.0, F : float = 1.0, qbar : float = 1.0, delta : float = 0.5):
kernel = GaussianKernel(7.)
model = BKB(kernel=kernel, d=1, lam=1e-5, noise_variance=1.0)
model.initialize(Xb)
print("[--] model.X: {}\tmodel.y: {}".format(model.X, model.y))

model.full_update([0], yb.reshape(-1))

print("[--] model.X: {}\tmodel.y: {}\tmodel.mean: {}\tmodel.var: {}".format(model.X, model.y, model.means, model.variances))
model.extend_arm_set(np.array([[-1.0], [9.0]]))
model.full_update([1], np.array([fun(-1.0)]))
model.full_update([0], np.array([fun(5.0)]))

print("[--] model.X: {}\tmodel.y: {}\tmodel.mean: {}\tmodel.var: {}".format(model.X, model.y, model.means, model.variances))
print("[--] model.ucbs: {}\tmodel.lcbs: {}\tembsize: {}".format(model.ucbs, model.lcbs, model.embedding_size))

model.extend_arm_set(np.array([[0.50], [2.0]]))
model.full_update([3], np.array([fun(0.50)]))
model.full_update([4], np.array([fun(2.0)]))
print("[--] model.ucbs: {}\tmodel.lcbs: {}\tembsize: {}".format(model.ucbs, model.lcbs, model.embedding_size))

print("-"*100)
mu, var = model.predict(X)

l = [5.0, -1.0, 0.5, 2.0]

fig, ax = plt.subplots()
ax.plot(X, y, '-', c="red", label="$f(x)$")
ax.plot(X, mu, '-', c="black", label="$\\tilde{\mu(x)}$")
ax.plot(l, [fun(x) for x in l], 'x', c="orange")
ax.fill_between(X.reshape(-1), mu - np.sqrt(var), mu + np.sqrt(var), alpha=0.6, color="black")
plt.savefig("./bkb_test.pdf")
