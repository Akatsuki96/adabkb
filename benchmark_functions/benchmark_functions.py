import numpy as np
from typing import Tuple

class BenchmarkFunction:
    """
    Generic benchmark function.

    Parameters
    ----------
    name
        string representing the name of the function
    search_space
        space of the parameters.
        A numpy array \\( d \\times 2 \\) where \\(d\\) are the dimensions of the search space.
        Each entry contains two scalars representing a lower and an upper bound, for instance
        <code>
        search_space = np.array([[0.0, 1.0], [0.0, 1.0]])
        </code>
        represents the square \\([0.0, 1.0]^2\\)
    global_min
        tuple containing a list of global minimizers of the function as first element and
        the global minimum as second element.
    noise_params : Tuple(float, numpy.random.RandomState)
        tuple composed by the standard deviation of a Gaussian distribution
        from which noise is sampled and a random state used to sample from a Gaussian.
    """

    def __init__(self,
        name : str,
        search_space : np.ndarray,
        global_min : Tuple,
        noise_params : Tuple= (0.0, None)):

        assert noise_params[0] >= 0.0 and search_space.shape[1] == 2 and search_space.shape[0] > 0
        self.name = name
        self.search_space = search_space
        self.dim = self.search_space.shape[0]
        self.noise_std = noise_params[0]
        self.global_min = global_min
        self.random_state = noise_params[1]

    def add_noise(self, f_val: float):
        """If noise_std is greater than 0, it adds a noise to the function evaluation f_val

        Parameters
        ----------
        f_val: float
            a real number (scalar).

        Returns
        -------
        noisy evaluation: float
            The evaluation plus the noise.

        """

        if self.noise_std == 0:
            noise = np.zeros(1)
        else:
            noise = self.random_state.normal(0, self.noise_std, 1)
        return (f_val + noise)[0]

    def __call__(self, x):
        pass

    def __str__(self):
        return "{} [d = {}] ".format(self.name, self.dim)


class Branin(BenchmarkFunction):
    r"""Branin-Hoo function defined as:

    \[
        f(x) = a(x_1 - bx_0^2 + cx_0 - r)^2 + s(1 - t)\cos(x_0) + s
    \]
    with the following values for the parameters
        \\[
            a=1, \\quad b=\\frac{5.1}{4\\pi^2}, \\quad c= \\frac{5}{\pi}, \\quad r=6, \\quad s=10, \\quad t= \\frac{1}{8\\pi}
        \\]
    """

    def __init__(self, noise_params : Tuple = (0.0, None)):
        global_minimizers = [(-np.pi, 12.275), (np.pi, 2.275), (9.42478, 2.475)]
        global_minimum = (global_minimizers, 0.397887)
        search_space = np.array([[-5.0, 10.0], [0.0, 15.0]])
        super().__init__("Branin", search_space, global_minimum, noise_params=noise_params)

    def __call__(self, x : np.ndarray):
        b = 5.1 / (4 * np.square(np.pi))
        c = 5.0 / (np.pi)
        s = 10.0
        r = 6.0
        t = 1.0 / (8 * np.pi)
        f_val = np.square(x[1] - b* np.square(x[0]) + c*x[0] - r) + s * (1 - t) *(np.cos(x[0])) + s
        return self.add_noise(f_val)

class Booth(BenchmarkFunction):
    r"""Booth function defined as:

    \[
        f(x) = (x_0 + 2x_1 - 7)^2 + (2x_0 + x_1 - 5)^2
    \]
    """

    def __init__(self, noise_params : Tuple = (0.0, None)):
        global_minimizers = [(1.0, 3.0)]
        global_minimum = (global_minimizers, 0.0)
        search_space = np.array([[-10.0, 10.0], [-10.0, 10.0]])
        super().__init__("Booth", search_space, global_minimum, noise_params=noise_params)

    def __call__(self, x : np.ndarray):
        f_val = np.square(x[0] + 2*x[1] - 7) + np.square(2 * x[0] + x[1] - 5) 
        return self.add_noise(f_val)

class SixHumpCamel(BenchmarkFunction):
    r"""Six Hump Camel function defined as:

    \[
        f(x) = (4 -2.1x_0^2 +\\frac{x_0^4}{3})x_0^2 + x_0x_1 + (-4 + 4x_1^2)x_1^2
    \]
    """

    def __init__(self, noise_params : Tuple = (0.0, None)):
        global_minimizers = [(0.0898, -0.7126), (-0.0898, 0.7126)]
        global_minimum = (global_minimizers, 0.0)
        search_space = np.array([[-3.0, 3.0], [-2.0, 2.0]])
        super().__init__("SixHumpCamel", search_space, global_minimum, noise_params=noise_params)

    def __call__(self, x : np.ndarray):
        f_val = (4 - 2.1*np.square(x[0]) + (x[0]**4)/3) * np.square(x[0]) + x[0]*x[1] + (-4 + 4*np.square(x[1])) * np.square(x[1]) 
        return self.add_noise(f_val)

class Rosenbrock(BenchmarkFunction):
    r"""Rosenbrock function in \(d\) dimensions is defined as:

    \[
        f(x) = \sum\limits_{i = 1}^{d - 1} 100 * (x_{i + 1} - x_i^2)^2 + (x_i - 1)^2
    \]
    """

    def __init__(self, d:int = 2, noise_params : Tuple = (0.0, None)):
        self.d = d
        global_minimizers = [[0 for _ in range(d)]]
        global_minimum = (global_minimizers, 0.0)
        search_space = np.array([[-5.0, 10.0] for _ in range(d)]).reshape(d, 2)
        super().__init__("Rosenbrock_{}".format(d), search_space, global_minimum, noise_params=noise_params)

    def __call__(self, x : np.ndarray):
        sum_out = 0.0        
        for i in range(self.d - 1):
            sum_out += 100 * np.square((x[i + 1] - x[i]**2)) + np.square(x[i] - 1)
        return self.add_noise(sum_out)


class Hartmann3(BenchmarkFunction):

    def __init__(self, noise_params : Tuple = (0.0, None)):
        global_minimizers = [[0.114614, 0.555649, 0.852547]]
        global_minimum = (global_minimizers, -3.86278)
        search_space = np.array([[0.0, 1.0] for _ in range(3)]).reshape(-1,2)
        self.alpha = np.array([1.0, 1.2, 3.0, 3.2])
        self.A = [[3.0, 10.0, 30.0], [0.1, 10.0, 35.0], [3.0, 10.0, 30.0], [0.1, 10.0, 35.0]]
        self.P = (10**(-4)) * np.array([[3689, 1170, 2673],\
            [4699, 4387, 7470],\
            [1091, 8732, 5547],\
            [381, 5743, 8828]]).reshape(4,3)
        super().__init__("Hartmann_3", search_space, global_minimum, noise_params=noise_params)

    def __call__(self, x):
        f = 0
        for i in range(4):
            inner = 0
            for j in range(3):
                inner += self.A[i][j]*np.square(x[j] - self.P[i][j])
            f += self.alpha[i] * np.exp(-inner)
        return self.add_noise(-f)

class Ackley(BenchmarkFunction):

    def __init__(self, d = 8, noise_params : Tuple = (0.0, None)):
        self.d = d
        minimizer = [[0. for _ in range(self.d)]]
        global_minimum = (minimizer, 0)
        search_space = np.array([[-10.0, 52.768] for _ in range(self.d)]).reshape(d, 2)
        super().__init__("Ackley_{}".format(d), search_space, global_minimum, noise_params=noise_params)

    def __call__(self, x):
        a, b, c = 20, 0.2, 2*np.pi
        s1 = -b*np.sqrt(  ( 1/self.d) * np.sum([x[i]**2 for i in range(self.d)]))
        s2 = (1/self.d) * np.sum([np.cos(c * x[i]) for i in range(self.d)])
        f = -a*np.exp(s1) - np.exp(s2) + a + np.exp(1)
        return self.add_noise(f) # x.shape[0]) 

class Shekel(BenchmarkFunction): # Shekel function with m = 10

    def __init__(self, noise_params : Tuple = (0.0, None)):
        minimizer = [[4.0, 4.0, 4.0, 4.0]]
        global_minimum = (minimizer, -10.5364)
        search_space = np.array([[0.0, 10.0] for _ in range(4)]).reshape(4, 2)
        super().__init__("Shekel", search_space, global_minimum, noise_params=noise_params)

    def __call__(self, xx):
        m = 10
        b = 0.1 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5])
        C = [[4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0], 
            [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6], 
            [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0], 
            [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6]]

        outer = 0
        for ii in range(m):
                bi = b[ii]
                inner = 0
                for jj in range(4):
                    xj = xx[jj]
                    Cji = C[jj][ii]
                    inner = inner + (xj-Cji)**2
                outer = outer + 1/(inner+bi)
        y = -outer
        return self.add_noise(y)
