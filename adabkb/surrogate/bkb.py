import numpy as np
from scipy.linalg import solve_triangular, svd, qr, qr_update, LinAlgError
from adabkb.utils import stable_invert_root, diagonal_dot

class BKB:
    
    def __init__(self, 
            kernel,
            d : int = 1,
            lam : float = 1.0,
            noise_variance : float = 1.0,
            F : float = 1.0, 
            qbar : float = 1.0, 
            delta : float = 0.5,
            seed : int = 12,
            beta : float = None):
        self.kernel = kernel
        self.lam = lam
        self.delta = delta
        self.noise_variance = noise_variance
        self.F = F
        self.qbar = qbar
        self.d = d
        self.random_state = np.random.RandomState(seed)
        self.X = np.zeros((1, d))
        self.Z = np.zeros((1, 1))
        self.y = np.zeros(1)
        self.X_norms = np.zeros(1)
        self.Z_norms = np.zeros((1, 1))
        self.means = np.zeros(1)
        self.variances = np.zeros(1)
        self.beta = 1.0 if beta is None else beta
        self.logdet = 1.0
        self.const_beta = beta is not None
        self.pulled_arms_count = np.zeros(1)
        self.pulled_arms_y = np.zeros(1)
        self.ucbs = np.zeros(1)
        self.lcbs = np.zeros(1)

        self.dict_arms_count = np.zeros(1)
        self.active_set = np.zeros((1, self.d))
        
        self.embedding_size = 1 

        self.A = np.zeros((1, 1))
        self.w = np.zeros(1)
        self.Q = np.zeros((1, 1))
        self.R = np.zeros((1, 1))
        
    @property
    def pulled_arms_matrix(self):
        return self.Z[self.pulled_arms_count != 0, :]

    @property
    def unique_arms_pulled(self):
        return np.count_nonzero(self.pulled_arms_count)

    @property
    def dict_size(self):
        return self.dict_arms_count.sum()

    @property
    def dict_size_unique(self):
        return np.count_nonzero(self.dict_arms_count)


    def initialize(self, x0):
        self.X[0] = x0
        self.X_norms = diagonal_dot(self.X, self.kernel)
        self.__update_embedding()

    def __update_arms(self, arms_idx, ys):
        self.y[arms_idx] += ys
        self.pulled_arms_count[arms_idx] += np.ones(ys.shape[0])

    def __resparsification(self):
  #      print("[--] X shape: {}".format(self.X.shape[0]))
        if self.X.shape[0] == 1:
            return
        resample_dict = self.random_state.rand(self.X.shape[0]) < (self.variances  * self.pulled_arms_count * self.qbar)
        assert resample_dict.sum() > 0
        dict_arms_count = np.zeros(self.X.shape[0])
        dict_arms_count[resample_dict] = 1
        self.active_set = self.X[dict_arms_count != 0, :]

    def __update_embedding(self):
        self.K_mm = self.kernel(self.active_set)
        self.K_km = self.kernel(self.X, self.active_set)
        try:
            U, S, _ = svd(self.K_mm)
        except LinAlgError:
            U, S, _ = svd(self.K_mm, lapack_driver='gesvd')
        self.U_thin, self.S_thin_inv_sqrt = stable_invert_root(U, S)
        
        self.Z = self.K_km.dot(self.U_thin * self.S_thin_inv_sqrt.T)
        self.Z_norms = np.square(np.linalg.norm(self.Z, axis = 1))
        self.embedding_size = len(self.S_thin_inv_sqrt)
    
    def __update_variances(self, idx_to_update):
        temp = (solve_triangular(self.R, (self.Z[idx_to_update, :].dot(self.Q)).T, 
                                 overwrite_b=True,
                                 check_finite=False).T * self.Z[idx_to_update, :]).sum(axis=1)
        self.variances[idx_to_update] = ((self.X_norms[idx_to_update] - self.Z_norms[idx_to_update]) / self.lam) + temp
        
        assert np.all(self.variances >= 0.)
        assert np.all(np.isfinite(self.variances))
    
    def __update_mean_variances(self, idx_to_update=None):
        pulled_arms_matrix = self.pulled_arms_matrix
        reweight_counts_vec = np.sqrt(self.pulled_arms_count[self.pulled_arms_count != 0].reshape(-1, 1))

        self.A = ((pulled_arms_matrix * reweight_counts_vec).T.dot(pulled_arms_matrix * reweight_counts_vec)
                + self.lam * np.eye(self.embedding_size))

        self.Q, self.R = qr(self.A)


        self.w = solve_triangular(self.R, self.Q.T.dot(pulled_arms_matrix.T.dot(self.y[self.pulled_arms_count != 0])))
        self.means[idx_to_update] = self.Z[idx_to_update, :].dot(self.w)
        assert np.all(np.isfinite(self.means))
        self.__update_variances(idx_to_update)

    def __update_beta(self):
        if self.const_beta:
            return
        self.logdet = (self.variances * self.pulled_arms_count).sum() * np.log(self.pulled_arms_count.sum())
        self.beta = np.sqrt(self.lam) * self.F + np.sqrt(self.noise_variance * (self.logdet + np.log(1 / self.delta)))
        assert np.isfinite(self.beta)
 
    def full_update(self, idx, ys):
        self.__update_arms(idx, ys)
        self.__resparsification()
        self.__update_embedding()
        self.__update_mean_variances(idx_to_update=idx)
        self.__update_beta()
        self.ucbs[idx] = self.means[idx] + self.beta * np.sqrt(self.variances[idx])
        self.lcbs[idx] = self.means[idx] - self.beta * np.sqrt(self.variances[idx])

    def update_emb(self, idx, ys):
        self.__update_arms(idx, ys)
        self.__resparsification()
        self.__update_embedding()
        
    def update_mean_variances(self, idx):
        self.__update_mean_variances(idx_to_update=idx)
        self.__update_beta()
        self.ucbs[idx] = self.means[idx] + self.beta * np.sqrt(self.variances[idx])
        self.lcbs[idx] = self.means[idx] - self.beta * np.sqrt(self.variances[idx])
        
        

    def extend_arm_set(self, new_arms):
        new_arms = new_arms.reshape(-1, self.X.shape[1])
        new_vec = np.zeros(new_arms.shape[0])
        self.X = np.concatenate((self.X, new_arms), axis=0)
        self.y = np.concatenate((self.y, new_vec))
   #     print("[--] X: {}".format(self.X))
        self.X_norms = diagonal_dot(self.X, self.kernel)
        self.means = np.concatenate((self.means, new_vec))
        self.variances = np.concatenate((self.variances, new_vec))
        self.pulled_arms_count = np.concatenate((self.pulled_arms_count, new_vec))
        self.dict_arms_count = np.concatenate((self.dict_arms_count, new_vec))
        self.ucbs = np.concatenate((self.ucbs, new_vec))
        self.lcbs = np.concatenate((self.lcbs, new_vec))
        self.K_km = self.kernel(self.X, self.active_set) 
        self.Z = self.K_km.dot(self.U_thin * self.S_thin_inv_sqrt.T)
        self.Z_norms = np.square(np.linalg.norm(self.Z, axis = 1))
        
        ind = list(range(self.X.shape[0]-new_arms.shape[0], self.X.shape[0]))
        
        self.__update_mean_variances(idx_to_update=ind)
        self.ucbs[ind] = self.means[ind] + self.beta * np.sqrt(self.variances[ind])
        self.lcbs[ind] = self.means[ind] - self.beta * np.sqrt(self.variances[ind])

    # Only for result visualization
    def __embedd(self, X):
        K_km = self.kernel(X, self.active_set)
        return K_km.dot(self.U_thin * self.S_thin_inv_sqrt.T)

    def predict(self, X):
        Z = self.__embedd(X)
 #       print("Z shape: ",Z.shape)
        temp = solve_triangular(self.R, (Z.dot(self.Q)).T, overwrite_b=False, check_finite=False).T
        temp *= Z
#        print("[--] temp: ", temp.shape)
        var = (diagonal_dot(X, self.kernel) - np.square(np.linalg.norm(Z, axis=1))) / self.lam  + np.sum(temp, axis=1)
        return Z.dot(self.w), var
        
#    def _embedd(self, X):
#        K_km = self.kernel(X, self.active_set)
#        return K_km.dot(self.U_thin * self.S_thin_inv_sqrt.T)
#        
#    def _update_emb(self):
#        K_mm = self.kernel(self.active_set)
#        try:
#            U, S, _ = svd(K_mm)
#        except LinAlgError:
#            U, S, _ = svd(K_mm, lapack_driver='gesvd')
#            print("numerical problem, had to use other")
#        self.U_thin, self.S_thin_inv_sqrt = stable_invert_root(U, S)
#        self.Z = self._embedd(self.X) 
#        self.Z_norm = np.square(np.linalg.norm(self.Z, axis=1))
#            
#    def _compute_variances(self):
#        temp = solve_triangular(self.R, (self.Z.dot(self.Q)).T, overwrite_b=True, check_finite=False).T
#        temp *= self.Z
#        print("[--] norms: {} {} Z: {} {}".format(self.X_norm.shape, self.Z_norm.shape, self.Z.shape, temp.shape))
#        self.variances = (self.X_norm - self.Z_norm) / self.lam + np.sum(temp, axis=1)
#            
#    def _compute_posterior(self):
#        self.A = self.Z.T.dot(self.Z) + self.lam * np.eye(self.Z.shape[1])
#        self.Q, self.R = qr(self.A, mode = "full")
#        self.w = solve_triangular(self.R, self.Q.T.dot(self.Z.T.dot(self.y)))
#
#      #  print(self.active_set)
#            
#    def fit(self, X : np.ndarray, y : np.ndarray):
#        self.X = X
#        self.y = y
#        self.active_set = self.X
#        self.X_norm = diagonal_dot(self.X, self.kernel)
#        self._resparification()
#        self._update_emb()
#        self._compute_posterior()
#        self._compute_variances()
#    
#    
#    
#    def update(self, x, y):
#        print("PRE: ",self.X.shape)
#        self.X = np.append(self.X, x, axis=0)
#        print("POST: ",self.X.shape)
#        self.y = np.append(self.y, y, axis=0)
#        self.X_norm = diagonal_dot(self.X, self.kernel)
#        self._resparification()
#        self._update_emb()
#        self._compute_posterior()
#    
#    def predict(self, X):
#        print(self.active_set.shape, X.shape)
#        Z = self._embedd(X)
#        print("Z shape: ",Z.shape)
#        temp = solve_triangular(self.R, (Z.dot(self.Q)).T, overwrite_b=False, check_finite=False).T
#        temp *= Z
#        print("[--] temp: ", temp.shape)
#        var = (diagonal_dot(X, self.kernel) - np.square(np.linalg.norm(Z, axis=1))) / self.lam  + np.sum(temp, axis=1)
#        return Z.dot(self.w), var