
from adabkb import AdaBKB
from batch_bkb.bkb_lib import BKB

from benchmark_functions import Branin
from numpy.random import RandomState
from adabkb.options import OptimizerOptions

from sklearn.gaussian_process.kernels import RBF

target = [Branin(0.01, RandomState(12))]
#    def __init__(self, lam=1., dot=None, fnorm=1., noise_variance=1., delta=.5, qbar=1, verbose=0):
lam = 0.01
sigma = 1.0
F = 1
delta = 0.025
qbar = 2.0
bkb_options = {
    'lam': lam,
    'dot': RBF(sigma),
    'fnorm':F,
    'noise_variance' : lam**2,
    'delta' : delta,
    'qbar' : qbar
}

adabkb_opt = AdaBKB 
