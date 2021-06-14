import numpy as np


class OptimizerOptions:
    """Options of the optimizer.

    Parameters
    ----------
    expand_fun : ExpansionProcedure
        function which describe how a partition is split.
    v_1 : float
        constant \\( v_1 \\geq 1\\). 
    rho : float
        constant \\(\\rho \\in (0,1)\\).
    gfun: Function
        a non-decreasing non-negative function \\(g\\) such that given
        \\[d_k (x,x^\\prime) = \\sqrt{k(x, x) + k(x^\\prime, x^\\prime) - 2k(x,x^\\prime)}\\]
        we have \\(\\forall x,x^\\prime \\in X, \\quad d_k(x, x^\\prime) \\leq g(d(x,x^\\prime))\\) with \\(d\\) any metric on \\(X\\) 
        and \\(\\exists \\delta > 0\\) s.t. \\(\\forall r \\leq \\delta\\)
        \\[ C_k r^\\alpha \\leq g(r) \\leq C^\\prime_k r^\\alpha\\]
        for some \\(C_k, C_k^\\prime > 0\\) and \\(\\alpha \\in (0,1)\\)
    lam : float
        noise standard deviation. It is assumed that noise \\(\\epsilon \\sim \\mathcal{N}(0, \\lambda^2) \\)
    noise_var : float
        noise variance. It is set to be lam^2
    delta : float
        constant \\(\\in (0,1)\\). The upper confidence bound used is valid with probability \\(1 - \\delta\\)
    fnorm : float
        norm of the function \\(f\\) in RKHS. If it is unknown it is set to be 1.
    qbar : float
        oversampling parameter. It must be > 0.
    """

    def __init__(self, expand_fun,\
         gfun,\
         v_1 : float = 1.0,\
         rho : float = 0.5,\
         lam:float = 1e-5,\
         noise_var:float = 1.,\
         delta:float=0.5,\
         fnorm:float=1.,\
         qbar:int = 1,\
         seed:int=42,\
         early_stopping=None,\
         verbose : bool = False):
        self.expand_fun = expand_fun
        self.lam = lam
        self.v_1 = v_1
        self.rho = rho
        self.verbose = verbose
        self.gfun = gfun
        self.fnorm = fnorm
        self.qbar = qbar
        self.noise_var = noise_var
        self.delta = delta
        self.random_state = np.random.RandomState(seed)
        self.early_stopping = early_stopping

    def __str__(self):
        return """
        Parameters:\n
        \t -) lambda  = {}\n
        \t -) noise variance = {}\n
        \t -) |f| = {}\n
        \t -) qbar = {}\n
        \t -) delta = {}
        """.format(self.lam, self.noise_var, self.fnorm, self.qbar, self.delta)
        