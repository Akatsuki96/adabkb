from sklearn.gaussian_process.kernels import RBF

class GaussianKernel(RBF):
    def __init__(self, length_scale=1, length_scale_bounds=...):
        super().__init__(length_scale=length_scale, length_scale_bounds=length_scale_bounds)

    def confidence_function(self, x):
        return (1/self.length_scale) * x