import numpy as np
from benchmark_functions import BenchmarkFunction, Branin
import unittest as ut


class GenericBFTestCase(ut.TestCase):
    """
    Test case for every BenchmarkFunction.
    """

    def __init__(self, method_name, fun_class = BenchmarkFunction):
        super().__init__(method_name)
        self.fun_class = fun_class
        self.uninf_gmin = (None, None)

    def test_bidimensional_bf(self):
        """
        Test: dimension of the search space is equal to the number of lower/upper bounds specified.
        """
        search_space = np.array([[0.1, 1.0], [2.0, 5.0]])
        benchmark_function = self.fun_class("Bidimensional benchmark function", search_space, self.uninf_gmin)
        self.assertEqual(benchmark_function.dim, 2)
    @ut.expectedFailure
    def test_single_element_in_bounds_search_space(self):
        """
        Test: lists of lower/upper bound cannot contain singletons
        """
        search_space = np.array([[0.1], [2.0]])
        self.fun_class("Invalid upper lower bound in search space", search_space, self.uninf_gmin)

    @ut.expectedFailure
    def test_empty_search_space_dimension(self):
        """
        Test: search space cannot be empty
        """
        self.fun_class("Empty search space", np.array([]), self.uninf_gmin)

    @ut.expectedFailure
    def test_negative_noise_std(self):
        """
        Test: noise std cannot be negative
        """
        self.fun_class("Negative noise std benchmark function",\
            np.array([[0.1, 2.0]]),\
            self.uninf_gmin,\
            (-5.0, np.random.RandomState(42)))



class BraninBFTestCase(GenericBFTestCase):

    def __init__(self, method_name):
        super().__init__(method_name, fun_class=Branin)
        self.no_noise = (0.0, None)

    def test_global_minima(self):
        br = Branin(self.no_noise)
        for gmin in br.global_min[0]:
            self.assertEqual(round(br(gmin),6), br.global_min[1])
