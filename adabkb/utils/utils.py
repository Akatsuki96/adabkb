import numpy as np
from sklearn.gaussian_process.kernels import Kernel

def flatten_list(lst):
    result = []
    for x in lst:
        if isinstance(x, list):
            for elem in x:
                result.append(elem)
        else:
            result.append(x)
    return result


def diagonal_dot(X, dot):
    n = X.shape[0]
    res = np.zeros(n)
    for i in range(n):
        res[i] = dot(X[i, :])

    return res

def stable_invert_root(U, S):
    n = U.shape[0]
    assert U.shape == (n, n)
    assert S.shape == (n,)

    thresh = S.max() * max(S.shape) * np.finfo(S.dtype).eps
    stable_eig = np.logical_not(np.isclose(S, 0., atol=thresh))
    m = sum(stable_eig)

    U_thin = U[:, stable_eig]
    S_thin = S[stable_eig]

    assert U_thin.shape == (n, m)
    assert S_thin.shape == (m,)

    S_thin_inv_root = (1 / np.sqrt(S_thin)).reshape(-1, 1)

    assert np.all(np.isfinite(S_thin_inv_root))

    return U_thin, S_thin_inv_root

def tonp(x):
    return np.asarray(x)


def to_minlen(lst, num_exp = 1):
    ris = []
    min_len = np.min([len(l) for l in lst])
    for l in lst:
        if len(l) > min_len:
            ris.append(l[:min_len])
        else:
            ris.append(l)
    return np.array(ris)#.reshape(num_exp, min_len)