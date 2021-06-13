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
