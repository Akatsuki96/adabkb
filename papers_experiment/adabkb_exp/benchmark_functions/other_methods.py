import numpy as np
from scipy.linalg import solve_triangular, svd, qr, qr_update, LinAlgError, cholesky
from copy import deepcopy

from sklearn.utils.extmath import fast_logdet
#from utils import diagonal_dot, stable_invert_root
import time
from adabkb.utils import *
from .utils import flatten_list
import torch
import gpytorch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

def totensor(X):
    return torch.tensor(X, dtype=torch.float64)

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



class Bkb(object):
    def __init__(self, lam=1., dot=None, fnorm=1., noise_variance=1.,\
        delta=.5, qbar=1, verbose=0, force_cpu=True):
        self.lam = lam
        self.dot = dot
        self.fnorm = fnorm
        self.noise_variance = noise_variance
        self.delta = delta
        self.qbar = qbar

        self.t = 1
        self.beta = 1
        self.logdet = 1
        self.k = 1
        self.m = 1
        self.d = 1

        self.arm_set = np.zeros((self.k, self.d))
        self.arm_set_norms = np.zeros(self.k)
        self.arm_set_embedded = np.zeros((self.k, self.m))
        self.arm_set_embedded_norms = np.zeros((self.k, 1))
        self.y = np.zeros(self.k)

        self.pulled_arms_count = np.zeros(self.k)
        self.pulled_arms_y = np.zeros(self.k)

        self.dict_arms_count = np.zeros(self.k)
        self.dict_arms_matrix = np.zeros((self.k, self.d))

        self.A = np.zeros((self.m, self.m))
        self.w = np.zeros(self.m)
        self.Q = np.zeros((self.m, self.m))
        self.R = np.zeros((self.m, self.m))

        self.means = np.zeros(self.k)
        self.variances = np.zeros(self.k)
        self.conf_intervals = np.zeros(self.k)

        self.verbose = verbose
        self.force_cpu = force_cpu

    @property
    def pulled_arms_matrix(self):
        return self.arm_set_embedded[self.pulled_arms_count != 0, :]

    @property
    def unique_arms_pulled(self):
        return np.count_nonzero(self.pulled_arms_count)

    @property
    def dict_size(self):
        return self.dict_arms_count.sum()

    @property
    def dict_size_unique(self):
        return np.count_nonzero(self.dict_arms_count)

    @property
    def embedding_size(self):
        return self.m

    # update functions
    def _update_embedding(self):

        K_mm = self.dot(self.dict_arms_matrix)
        K_km = self.dot(self.arm_set, self.dict_arms_matrix)

        try:
            U, S, _ = svd(K_mm)
        except LinAlgError:
            U, S, _ = svd(K_mm, lapack_driver='gesvd')
            print("numerical problem, had to use other")

        U_thin, S_thin_inv_sqrt = stable_invert_root(U, S)
        self.arm_set_embedded = K_km.dot(U_thin * S_thin_inv_sqrt.T)
        self.arm_set_embedded_norms = np.linalg.norm(self.arm_set_embedded, axis=1)
        np.square(self.arm_set_embedded_norms, out=self.arm_set_embedded_norms)

        self.m = len(S_thin_inv_sqrt)
        assert(self.arm_set_embedded.shape == (self.k, self.m))
        assert(np.all(self.arm_set_embedded_norms
                      <= self.arm_set_norms + self.m**2 * np.finfo(self.arm_set_embedded_norms.dtype).eps))
        assert(np.all(np.isfinite(self.arm_set_embedded_norms)))

    def _reset_posterior(self, idx_to_update=None):
        #initialize A, P, and w
        pulled_arms_matrix = self.pulled_arms_matrix
        reweight_counts_vec = np.sqrt(self.pulled_arms_count[self.pulled_arms_count != 0].reshape(-1, 1))

        self.A = ((pulled_arms_matrix * reweight_counts_vec).T.dot(pulled_arms_matrix * reweight_counts_vec)
                  + self.lam * np.eye(self.m))

        self.Q, self.R = qr(self.A)

        self.w = solve_triangular(self.R, self.Q.T.dot(pulled_arms_matrix.T.dot(self.y[self.pulled_arms_count != 0])))

        self.means = self.arm_set_embedded.dot(self.w)

        assert np.all(np.isfinite(self.means))

        self._update_variances(idx_to_update)
        self.conf_intervals = self.beta * np.sqrt(self.variances)

    def _update_variances(self, idx_to_update=None):
        if idx_to_update is None:
            temp = solve_triangular(self.R,
                                    (self.arm_set_embedded.dot(self.Q)).T,
                                    overwrite_b=True,
                                    check_finite=False).T
            temp *= self.arm_set_embedded
            self.variances = (self.arm_set_norms - self.arm_set_embedded_norms) / self.lam + np.sum(temp, axis=1)
        else:
            temp = solve_triangular(self.R,
                                    (self.arm_set_embedded[idx_to_update, :].dot(self.Q)).T,
                                    overwrite_b=True,
                                    check_finite=False).T
            temp *= self.arm_set_embedded[idx_to_update, :]
            self.variances[idx_to_update] = (
                    (self.arm_set_norms[idx_to_update] - self.arm_set_embedded_norms[idx_to_update]) / self.lam
                    + np.sum(temp, axis=1)
            )
        assert np.all(self.variances >= 0.)
        assert np.all(np.isfinite(self.variances))

    def _update_beta(self):
        self.logdet = (self.variances * self.pulled_arms_count).sum() * np.log(self.pulled_arms_count.sum())
        self.beta = np.sqrt(self.lam) * self.fnorm + np.sqrt(self.noise_variance * (self.logdet + np.log(1 / self.delta)))
        assert np.isfinite(self.beta)

    # main loop functions
    def initialize(self, arm_set, index_init, y_init, random_state: np.random.RandomState, dict_init=None):
        self.k = arm_set.shape[0]
        self.t = len(index_init)
        self.m = len(np.unique(index_init))

        self.arm_set = arm_set
        self.arm_set_norms = diagonal_dot(self.arm_set, self.dot)

        #initialize X (represented using arm pull count) and y
        self.pulled_arms_count = np.zeros(self.k)
        self.y = np.zeros(self.k)
        for i, idx in enumerate(index_init):
            self.pulled_arms_count[idx] = self.pulled_arms_count[idx] + 1
            self.y[idx] = self.y[idx] + y_init[i]

        #initialize dict
        self.dict_arms_count = np.zeros(self.k)

        if dict_init is None:
            dict_init = index_init

        for idx in dict_init:
            self.dict_arms_count[idx] = self.dict_arms_count[idx] + 1

        self.dict_arms_matrix = self.arm_set[self.dict_arms_count != 0, :]
        self._update_embedding()
        self._reset_posterior()
        self._update_beta()

    def predict(self):
        ucbs = self.means + self.beta * np.sqrt(self.variances)
        assert np.all(np.isfinite(ucbs))
        chosen_arm_idx = np.argmax(ucbs)

        if self.verbose > 1:
            print(f'chosen {chosen_arm_idx} {ucbs[chosen_arm_idx]} {self.means[chosen_arm_idx]} {self.variances[chosen_arm_idx]}'
                  f'beta {self.beta} ucbs {self.means.max()} {self.means.min()} {self.variances.max()} {self.variances.min()}')

        return [chosen_arm_idx], ucbs

    def update(self, chosen_arm_idx_list, loss_list, random_state):
        self._update_pulled_arm_list_and_feedback(chosen_arm_idx_list, loss_list)

        self._resample_dict(random_state)
        self._update_embedding()
        self._reset_posterior()
        self._update_beta()

        return self

    # miscellaneous utils
    def _update_pulled_arm_list_and_feedback(self, chosen_arm_idx_list, loss_list):
        assert len(chosen_arm_idx_list) == len(loss_list)
        assert np.all(np.isfinite(loss_list))

        for i, chosen_arm_idx in enumerate(chosen_arm_idx_list):
            self.pulled_arms_count[chosen_arm_idx] = self.pulled_arms_count[chosen_arm_idx] + 1
            self.y[chosen_arm_idx] = self.y[chosen_arm_idx] + loss_list[i]
        self.t += len(chosen_arm_idx_list)

        assert self.t == self.pulled_arms_count.sum()

    def _resample_dict(self, random_state):
        resample_dict = random_state.rand(self.k) < (self.variances * self.pulled_arms_count * self.qbar)
        assert resample_dict.sum() > 0
        self.dict_arms_count = np.zeros(self.k)
        self.dict_arms_count[resample_dict] = 1
        self.dict_arms_matrix = self.arm_set[self.dict_arms_count != 0, :]


class GPUCB:

    def __init__(self, dot, sigma=1.0, lam=1e-12, noise_variance=1e-10, delta=0.50, a=0.1, b=0.5, r=1.01, training_iter = 5):
        self.dot = dot
        self.sigma = sigma
        self.delta = delta
        self.lam= lam
        self.a = a
        self.b = b
        self.r = r
        self.training_iter = training_iter
        self.X = []
        self.Y = []
        self.t= 0

    def optimize(self):
        self.gp.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam([{'params': self.gp.parameters()}], lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp)
        for i in range(self.training_iter):
            optimizer.zero_grad()
            output = self.gp(self.X)
            loss = -mll(output, self.Y)
            loss.backward()
            print("\t sigma: {}\t noise: {}\t loss: {}".format(self.gp.covar_module.lengthscale.item(), self.gp.likelihood.noise.item(), loss))
            optimizer.step()


    def initialize(self, arm_set, index_init, y_init, rnd_state):
        self.arm_set = torch.from_numpy(arm_set)
        Xinit = self.arm_set[index_init]
        self.t += self.arm_set.shape[0]
        self.X = Xinit#torch.from_numpy(Xinit)
        self.Y = torch.from_numpy(y_init).reshape(-1)
        self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.tensor(self.lam).reshape(-1,1))
        self.gp = ExactGPModel(self.X, self.Y, self.likelihood)
        params = {
            'covar_module.lengthscale' :     self.sigma,
        }
        self.gp.initialize(**params)
       # self.optimize()
        d = self.arm_set.shape[1]
        self.beta = torch.tensor([1.0])
      #  p2 = 2*d * np.log(np.square(self.t)* d*self.b*self.r*np.sqrt( np.log(4*d*self.a/self.delta)))
      #  self.beta = torch.tensor(2 * np.log(np.square(self.t)*2*np.square(np.pi)/(3*self.delta)) + p2)
        #print("[--] Params: {}".format(self.gp.parameters()))

    def predict(self):
        self.gp.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictive_distribution = self.likelihood(self.gp(self.arm_set))
            ucbs = predictive_distribution.mean + torch.sqrt(self.beta) * torch.sqrt(predictive_distribution.variance)
            chosen_idx = torch.argmax(ucbs)
            return [chosen_idx], ucbs

    def update(self, chosen_arm_idx_list, loss_list, random_state):
        chosen_idx = chosen_arm_idx_list[0]
        yt = loss_list[0]
        self.X = torch.cat([self.X, torch.tensor(self.arm_set[chosen_idx]).reshape(-1, self.X.shape[1])], dim=0)
        self.Y = torch.cat([self.Y, torch.tensor(yt).reshape(-1)], dim=0)
        params = {
            'covar_module.lengthscale' :     self.sigma,
       #
        }
     #   self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.tensor(self.lam).reshape(-1,1))
        self.gp = ExactGPModel(self.X, self.Y, self.likelihood)
        self.gp.initialize(**params)
       # self.optimize()
        self.t+= 1
        d = self.arm_set.shape[1]
       # p2 = 2*d * np.log(np.square(self.t)*d*self.b*self.r*np.sqrt( np.log(4*d*self.a/self.delta)))
       
class Gpucb(Bkb):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pulled_arms_list = np.zeros(1)
        self.y = np.zeros(1)
        self.K_lam_chol = np.zeros((1,1))
        self.K_kt = np.zeros((1,1))
        self.alpha = np.zeros(1)

        self.X = []
        self.Y = []
        self.Ymap = {}
        


    @property
    def pulled_arms_matrix(self):
        return self.arm_set[self.pulled_arms_list, :]

    def initialize(self, arm_set,  index_init, y_init, random_state: np.random.RandomState, dict_init=None):
        self.k = arm_set.shape[0]
        self.pulled_arms_count = np.zeros(self.k)
        self.t = len(index_init)
        for i, chosen_arm_idx in enumerate(index_init):
            self.pulled_arms_count[chosen_arm_idx] = self.pulled_arms_count[chosen_arm_idx] + 1

        self.arm_set = arm_set
        self.arm_set_norms = diagonal_dot(self.arm_set, self.dot)

        #initialize X (represented using arm pull count) and y
        self.pulled_arms_list = np.array(index_init)
        self.y = y_init.reshape(-1)

        #initialize weights alpha and beta
        self.K_kt = self.dot(self.arm_set, self.pulled_arms_matrix)
        self.K_lam_chol = cholesky(self.dot(self.pulled_arms_matrix) + self.lam * np.eye(self.t))
        self.alpha = solve_triangular(self.K_lam_chol, self.y, trans=1)
        mlog = fast_logdet(self.dot(self.pulled_arms_matrix) / self.lam + np.eye(self.t))
        assert np.isfinite(mlog)

        self.logdet = mlog
        self.beta = np.sqrt(self.lam) * self.fnorm + np.sqrt(self.noise_variance * (self.logdet + np.log(1 / self.delta)))

        self.means = self.K_kt.dot(self.alpha)
        assert np.all(np.isfinite(self.means))
        print("[--] yt: {}".format(self.y))
        print("[--] Gpucb: {}".format(self.means))
        self._update_variances()

    def predict(self):
        ucbs = self.means + self.beta * np.sqrt(self.variances)
        assert np.all(np.isfinite(ucbs))
        chosen_arm_idx = np.argmax(ucbs)
        return [chosen_arm_idx], ucbs

    def update(self, chosen_arm_idx_list, loss_list, random_state):
        self.pulled_arms_list = np.append(self.pulled_arms_list, np.array(chosen_arm_idx_list))
        self.y = np.append(self.y, loss_list)
        self.t += len(chosen_arm_idx_list)

        assert self.pulled_arms_list.shape == (self.t,)
        assert len(self.pulled_arms_matrix.shape) == 2
        assert self.pulled_arms_matrix.shape[0] == self.t
        assert self.y.shape == (self.t,)

        self._update_K_chol(chosen_arm_idx_list)

        self.alpha = solve_triangular(self.K_lam_chol, self.y, trans=1)
        self.means = self.K_kt.dot(self.alpha)
        self._update_variances()

        self.logdet += np.log(1 + self.variances[chosen_arm_idx_list]).sum()
        self.beta = np.sqrt(self.lam) * self.fnorm + np.sqrt(self.noise_variance * (self.logdet + np.log(1 / self.delta)))
        assert np.isfinite(self.beta)
       # print("[--] means: {}\tvar: {}".format(self.means, self.variances))
        return self

    def _update_variances(self, idx_to_update=None):

        if idx_to_update is None:
            tmp_mat = solve_triangular(self.K_lam_chol, self.K_kt.T, trans=1)
            tmp_mat *= tmp_mat
            self.variances = (self.arm_set_norms - np.sum(tmp_mat, axis=0)) / self.lam
        else:
            K_tx = self.dot(self.pulled_arms_matrix, self.arm_set[idx_to_update, :])
            tmp_mat = solve_triangular(self.K_lam_chol, K_tx, trans=1)
            tmp_mat *= tmp_mat
            self.variances[idx_to_update] = (self.arm_set_norms[idx_to_update] - np.sum(tmp_mat, axis=0)) / self.lam

        assert np.all(self.variances >= 0.)
        assert np.all(np.isfinite(self.variances))

    def _update_K_chol(self, chosen_arm_idx_list):
        b = len(chosen_arm_idx_list)

        K_kx = self.dot(self.arm_set, self.arm_set[chosen_arm_idx_list, :])
        self.K_kt = np.concatenate((self.K_kt, K_kx), axis=1)

        K_tx = K_kx[self.pulled_arms_list, :]

        # look at https: // en.wikipedia.org / wiki / Cholesky_decomposition  # Adding_and_removing_rows_and_columns
        # set A_13 and A_33 to zero (ignore them)
        A_12 = K_tx[:-b, :]
        A_22 = K_tx[-b:, :] + self.lam * np.eye(b)

        L_11 = self.K_lam_chol
        S_11 = L_11
        S_12 = solve_triangular(L_11, A_12, trans=1)
        S_22 = cholesky(A_22 - S_12.T.dot(S_12))

        self.K_lam_chol = np.block([[self.K_lam_chol, S_12], [np.zeros(S_12.T.shape), S_22]])

class AdaGpucb:
    def __init__(self, sigma, g, v_1, rho, d: int = 1,\
        C1: float = 1.0, lam = 1e-12, noise_var = 1e-12, fnorm = 1.0, delta = 0.5,\
              expand_fun: ExpansionProcedure = GreedyExpansion()):
       # self.dot = dot_fun
        self.d = d
        self.fnorm = fnorm
        self.delta = delta
        self.v_1 = v_1
        self.g = g
        self.rho = rho
        self.sigma = sigma
        self.lam = lam
        self.noise_variance = noise_var
        self.expand_fun = expand_fun
        self.C1 = C1
        self.Ymap = {}
        self.node2idx = {}
        self.father_ucbs = {}
        self.X = []
        self.pulled_arms_count = [0]
        self.Y = []#np.zeros(1)
        self.num_nodes = 0
        self.logdet = 1
        self.beta = 1
        self.K_lam_chol = np.zeros((1,1))
        self.K_kt = np.zeros((1,1))
        self.alpha = np.zeros(1)
      

    def _init_lsts_constants(self, xroot, horizon, hmax):
        self.iteration_time = []
        self.leaf_set_size = [1]
        self.cumulative_expansion_time = []
        self.cumulative_evaluation_time = []
        self.time_over_budget = []
        self.cumulative_regret = []
        self.cumulative_regret_abs = []
        alpha1 = np.sum([(2 ** (-i + 1))*np.sqrt(np.log(i)) for i in range(1, hmax + 1)])
        alpha2 = np.sum([(2 ** (-i + 1))*np.sqrt(i) for i in range(1, hmax + 1)])
        self.C3 = alpha1 + alpha2*np.sqrt(2*self.d*np.log(2))
        self.C2 = 2*np.log(2*np.square(self.C1)*np.square(np.pi)/6)
        self.C4 = self.C2 + 2*np.log(np.square(horizon)*np.square(np.pi)/6)
        self.node2idx[tuple(xroot)] = self.num_nodes
        self.num_nodes += 1
        self.pulled_arms_count[0] = 0
        self.beta = np.sqrt(self.delta + np.log(horizon*hmax)) 
    def _compute_V(self, h):
        gr = self.g(self.v_1 * (self.rho**h))
        return gr*(np.sqrt(2*self.delta + self.C4 + h*np.log(self.N) + 4*self.d*np.log(1/gr)) + self.C3) #*4

    def initialize(self, target_fun, root):
        yt = target_fun(root.x)
        self.X.append(root.x)
        self.Y.append(yt)
        self.Ymap[0] = yt
        self.pulled_arms_count[0] += 1
        # initialize likelihood and model
        self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.tensor([self.lam]))
        self.gp = ExactGPModel(totensor(self.X), totensor(self.Y), self.likelihood)
        params = {
            'covar_module.lengthscale' :     self.sigma,
        }
        self.gp.initialize(**params)
        self.gp.train()
        self.likelihood.train()
        self.gp.eval()
        self.likelihood.eval()
        #self.gp.fit(np.array(self.X).reshape(-1,self.d), np.array(self.Y).reshape(-1,1))
        pred = self.gp(totensor([root.x]))
        mu = pred.mean
        sigma = torch.sqrt(pred.variance)
        #mu, sigma = self.gp.predict(np.array([root.x]).reshape(-1,self.d), return_std=True)
        Vroot = self._compute_V(0)
        while self.beta * sigma > Vroot:
            yt = target_fun(root.x)
            self.X.append(root.x)
            self.Y.append(yt)
            self.Ymap[0] += yt
            self.gp = ExactGPModel(totensor(self.X).reshape(-1, self.d), totensor(self.Y).reshape(-1), self.likelihood)
        #self.gp.fit(np.array(self.X).reshape(-1,self.d), np.array(self.Y).reshape(-1,1))
            self.gp.train()
            self.likelihood.train()
            self.gp.eval()
            self.likelihood.eval()
            pred = self.gp(totensor([root.x]).reshape(-1,self.d))
            mu = pred.mean
            sigma = torch.sqrt(pred.variance)
        leaf_set = root.expand_node()
        for node in leaf_set:
            if tuple(node.x) not in self.node2idx:
                self.node2idx[tuple(node.x)] = self.num_nodes
                self.pulled_arms_count = np.append(self.pulled_arms_count, 0)
                self.num_nodes += 1
        self.fathers = {}
        self.fathers[self.node2idx[tuple(root.x)]] = (mu, sigma)
        self.fmeans, self.fstds = mu, sigma
        #self.gp.train()
        #self.likelihood.train()
        self.gp.eval()
        self.likelihood.eval()
        preds = self.gp(totensor([node.x for node in leaf_set]))
        self.means = preds.mean
        self.stds = torch.sqrt(preds.variance)
        #self.means, self.stds = self.gp.predict(tonp([node.x for node in leaf_set]).reshape(-1, self.d), return_std=True) 
        return leaf_set, self.Ymap[0]/self.pulled_arms_count[0]


    def _select_node(self, leaf_set):
        ucbs = []
        for i in range(self.means.reshape(-1).shape[0]):
            father_id = self.node2idx[tuple(leaf_set[i].father.x)]
            A = self.means[i].reshape(-1)+ self.beta*self.stds[i].reshape(-1)
            fmu, fsigma = self.fathers[father_id]
            B = fmu.reshape(-1) + self.beta*fsigma.reshape(-1) + self._compute_V(leaf_set[i].level-1)
            U = torch.min(totensor([A, B])) + self._compute_V(leaf_set[i].level)
            ucbs.append(U)
        
        best_idx = np.argmax(ucbs)
        return best_idx, self.means[best_idx].reshape(-1), self.stds[best_idx].reshape(-1), self._compute_V(leaf_set[best_idx].level )

    def _map_new_nodes(self, children):
        for node in children:
            if tuple(node.x) not in self.node2idx:
              #  print("NODE: ",node.x)
                self.node2idx[tuple(node.x)] = self.num_nodes
                self.pulled_arms_count = np.append(self.pulled_arms_count, 0)
                self.num_nodes += 1

    def _expansion(self, leaf_set, leaf_idx, mu, sigma):
        xt = leaf_set[leaf_idx]
        children = xt.expand_node()
        leaf_set[leaf_idx] = children
        leaf_set = flatten_list(leaf_set)
        self._map_new_nodes(children)
        self.fathers[self.node2idx[tuple(xt.x)]] = (mu, sigma)
        old_means = totensor(self.means).reshape(-1).clone()
        old_std = totensor(self.stds).reshape(-1).clone()
        self.means, self.stds = np.zeros(len(leaf_set)), np.zeros(len(leaf_set))
        c = 0
        for i in range(0, leaf_idx):
            self.means[i], self.stds[i] = old_means[i], old_std[i]
            c+=1
        c+=1
        for i in range(leaf_idx + self.N, len(leaf_set)):
            self.means[i], self.stds[i] = old_means[c], old_std[c]
            c += 1
  #              self.gp.train()
   #     self.likelihood.train()
        self.gp.eval()
        self.likelihood.eval()
        preds = self.gp(totensor([node.x for node in children]).reshape(-1, self.d))
        new_mu, new_sig = preds.mean, torch.sqrt(preds.variance)
        c = 0
        for i in range(leaf_idx, leaf_idx + self.N):
            self.means[i], self.stds[i] = new_mu[c], new_sig[c]
            c += 1
        return leaf_set

    def _update(self, leaf_set, node, yt):
        self.X.append(node.x)
        self.Y.append(yt)
        
        self.gp = ExactGPModel(totensor(self.X).reshape(-1,self.d), totensor(self.Y).reshape(-1), self.likelihood)
        params = {
            'covar_module.lengthscale' :     self.sigma,
        }
        self.gp.initialize(**params)
        #self.gp.fit(self.X, self.Y)
        self.gp.train()
        self.likelihood.train()
        self.gp.eval()
        self.likelihood.eval()
        preds = self.gp(totensor([node.x for node in leaf_set]).reshape(-1, self.d))
        self.means, self.stds = preds.mean, torch.sqrt(preds.variance)
        fathers_node = totensor(list(set([tuple(node.father.x) for node in leaf_set])))
       # self.means, self.stds = self.gp.predict(tonp([node.x for node in leaf_set]).reshape(-1, self.d), return_std=True) 
        preds = self.gp(fathers_node.reshape(-1,self.d) )
        mean_f, std_f = preds.mean, torch.sqrt(preds.variance)#self.gp.predict(fathers_node, return_std=True)
        for i in range(len(fathers_node)):
        #    print(self.node2idx)
            self.fathers[self.node2idx[tuple(fathers_node[i].detach().numpy())]] = (mean_f[i], std_f[i]) 


    def run(self, target_fun, clean_target, search_space, N, budget, real_best=0.0, hmax = None, time_threshold= None, out_dir="./"):
        assert N > 1 and (hmax is None or hmax > 1)
        if hmax is None:
            hmax = int(np.log(budget))
            

        self.N = N
        #self.gp = GaussianProcessRegressor(self.dot)
        root = PartitionTreeNode(search_space, N, None, 0, 0, self.expand_fun)
        
        tot_time = time.time()
        if time_threshold is None:
            time_threshold = 1200
        self._init_lsts_constants(root.x, budget, hmax)
        leaf_set, y_root = self.initialize(target_fun, root)
        best = (root.x, y_root)
        last_selected = root
        ne = 0
        #self.leaf_set_size.append(len(leaf_set))
        time_budget = time.time()
        while ne < budget:
            it_time = time.time()
            leaf_idx, mu, sigma, Vh = self._select_node(leaf_set)
            node = leaf_set[leaf_idx]
            if self.beta*sigma <= Vh and node.level < hmax:
                print("[AdaGPUCB] ne: {}\tSelected: {}\t conf: {}\t Vh: {}\th: {}/{}\tlf size: {}".format(ne, node.x, self.beta*sigma, Vh,node.level, hmax, len(leaf_set)))
                leaf_set = self._expansion(leaf_set, leaf_idx, mu, sigma)               
                #self.cumulative_evaluation_time.append(0)
                #self.cumulative_expansion_time.append(time.time() - it_time)
                etime = time.time() - it_time
                with open(out_dir+"etime.log", "a") as f:
                    f.write("{}\n".format(etime))
            else:
                yt = target_fun(node.x)
                print("[AdaGPUCB] ne: {}\txt: {}\tyt: {}\tlf size: {}".format(ne, node.x, -yt, len(leaf_set)))
                if self.node2idx[tuple(node.x)] not in self.Ymap:
                    self.Ymap[self.node2idx[tuple(node.x)]] = yt
                else:
                    self.Ymap[self.node2idx[tuple(node.x)]] += yt
                self.pulled_arms_count[self.node2idx[tuple(node.x)]] += 1
                avg_rew = self.Ymap[self.node2idx[tuple(node.x)]] / self.pulled_arms_count[self.node2idx[tuple(node.x)]] 
                if avg_rew > best[1]:
                    best = (node.x, avg_rew)

                #self.cumulative_regret.append(np.abs(best[1] - avg_rew))
                #self.cumulative_regret_abs.append(np.abs(real_best - avg_rew))
                self._update(leaf_set, node, yt)
                #self.time_over_budget.append(time.time() - time_budget)
                time_budget = time.time()
                ne += 1
                etime = time.time() - it_time
                #self.cumulative_expansion_time.append(0)
                #self.cumulative_evaluation_time.append(etime)
                with open(out_dir+"trace.log", "a") as f:
                    f.write("{},{},{},{},{}\n".format(clean_target(node.x), etime,len(leaf_set), 0, False))
                with open(out_dir+"etime.log", "a") as f:
                    f.write("{}\n".format(0))

                #with open(out_dir+"adagpucb_out.log", "a") as f:
                #    f.write("{},".format(self.Ymap[self.node2idx[tuple(node.x)]] /self.pulled_arms_count[self.node2idx[tuple(node.x)]] ))
            #self.iteration_time.append(time.time() - it_time)
            #self.leaf_set_size.append(len(leaf_set))
            last_selected = node.x
            print("[AdaGPUCB] Time from initialization: {} seconds".format(time.time() - tot_time))
            if time.time() - tot_time >= time_threshold:
                print("[!!] Interrupted: execution longer than {} seconds!".format(time_threshold))
                break
        #self.iteration_time = np.cumsum(self.iteration_time)
        #self.cumulative_regret = np.cumsum(self.cumulative_regret)
        #self.cumulative_regret_abs = np.cumsum(self.cumulative_regret_abs)
        #self.cumulative_expansion_time = np.cumsum(self.cumulative_expansion_time)
        #self.cumulative_evaluation_time = np.cumsum(self.cumulative_evaluation_time)
        #self.time_over_budget = np.cumsum(self.time_over_budget)
        return leaf_set, last_selected, best[0]

__all__ = ('Bkb', 'Gpucb', 'AdaGpucb', 'GPUCB')