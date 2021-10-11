import h5py
import numpy as np
import pandas as pd


import torch
import scipy.io as scio


class Dataset:

    def __init__(self, name, path):
        self.name = name
        self.path = path


    def __preprocess__(self, X, Y):
        pass

    def __split__(self, training_perc = 0.8):
        pass



class HTRU2(Dataset):

    def __init__(self, path):
        super().__init__("HTRU2", path)
        self.d = 8

    def preprocess_x(self, Xtr, Xts):
        if isinstance(Xtr, np.ndarray):
            mXtr = Xtr.mean(axis=0, keepdims=True, dtype=np.float32).astype(Xtr.dtype)
            sXtr = Xtr.std(axis=0, keepdims=True, dtype=np.float32, ddof=1).astype(Xtr.dtype)
        else:
            mXtr = Xtr.mean(dim=0, keepdims=True)
            sXtr = Xtr.std(dim=0, keepdims=True)

        Xtr -= mXtr
        Xtr /= sXtr
        Xts -= mXtr
        Xts /= sXtr

        return Xtr, Xts

    def preprocess_y(self, Ytr, Yts):
        Ytr = Ytr * 2 - 1
        Yts = Yts * 2 - 1
        return Ytr.reshape((-1, 1)), Yts.reshape((-1, 1))
    
    def __split__(self):
        X = pd.read_csv(self.path,names=["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "T"], dtype=np.float64)
        X = X.sample(frac=1.)        
        Y = X["T"].values
        X = X.drop(labels=["T"], axis=1).values
        training_part = int(X.shape[0] * 0.7)
        Xtr, Xts = np.array(X[:training_part], dtype=np.float64).reshape(-1, 8), np.array(X[training_part:],  dtype=np.float64).reshape(-1,8)
        ytr, yts = np.array(Y[:training_part],  dtype=np.float64).reshape(-1,1), np.array(Y[training_part:],  dtype=np.float64).reshape(-1,1)
        Ntr = int(X.shape[0] * 0.8)
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        idx_tr = idx[:Ntr]
        idx_ts = idx[Ntr:]

        Xtr, Xts = self.preprocess_x(X[idx_tr], X[idx_ts])
        Ytr, Yts = self.preprocess_y(Y[idx_tr], Y[idx_ts])
        return torch.from_numpy(Xtr), torch.from_numpy(Ytr), torch.from_numpy(Xts), torch.from_numpy(Yts)



class Magic(Dataset):

    def __init__(self, path):
        super().__init__("Magic", path)#"./data/magic04.data")
        self.d = 10

    def preprocess_x(self, Xtr, Xts):
        if isinstance(Xtr, np.ndarray):
            mXtr = Xtr.mean(axis=0, keepdims=True, dtype=np.float32).astype(Xtr.dtype)
            sXtr = Xtr.std(axis=0, keepdims=True, dtype=np.float32, ddof=1).astype(Xtr.dtype)
        else:
            mXtr = Xtr.mean(dim=0, keepdims=True)
            sXtr = Xtr.std(dim=0, keepdims=True)

        Xtr -= mXtr
        Xtr /= sXtr
        Xts -= mXtr
        Xts /= sXtr

        return Xtr, Xts

    def preprocess_y(self, Ytr, Yts):
        Ytr = Ytr * 2 - 1
        Yts = Yts * 2 - 1
        return Ytr.reshape((-1, 1)), Yts.reshape((-1, 1))
    
    def __split__(self):
        X = pd.read_csv(self.path,names=["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "T"])
        #X = X.sample(frac=1.)        
        Y = X["T"].values
        X = X.drop(labels=["T"], axis=1).values
        Y[Y=="g"] = -1.0
        Y[Y=="h"] = 1.0
        print("X shape: ",X.shape)
        Ntr = int(X.shape[0] * 0.8)
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        idx_tr = idx[:Ntr]
        idx_ts = idx[Ntr:]

        Xtr, Xts = self.preprocess_x(X[idx_tr], X[idx_ts])
        Ytr, Yts = Y[idx_tr], Y[idx_ts]#self.preprocess_y(Y[idx_tr], Y[idx_ts])

        Xtr, Xts = np.array(Xtr, dtype=np.float64).reshape(-1, 10), np.array(Xts,  dtype=np.float64).reshape(-1,10)
        ytr, yts = np.array(Ytr,  dtype=np.float64).reshape(-1,1), np.array(Yts,  dtype=np.float64).reshape(-1,1)
        return torch.from_numpy(Xtr), torch.from_numpy(ytr), torch.from_numpy(Xts), torch.from_numpy(yts)

class CASP(Dataset):
    
    def __init__(self, path):
        super().__init__("CASP", path)# "./data/CASP.csv")
        self.d = 9

    def preprocess_x(self, Xtr, Xts):
        Xm = Xtr.mean(axis=0, keepdims=True, dtype=np.float32).astype(Xtr.dtype)
        Xstd = Xtr.std(axis=0, keepdims=True, dtype=np.float32, ddof=1).astype(Xtr.dtype)
        Xtr -= Xm
        Xtr /= Xstd
        Xts -= Xm
        Xts /= Xstd
        return Xtr, Xts

    def __split__(self):

        X = pd.read_csv(self.path)
        X = X.sample(frac = 1)
        Y = X['RMSD'].values
        X = X.drop(labels=["RMSD"], axis=1).values
        training_part = int(X.shape[0] * 0.7)
        Xtr, Xts = np.array(X[:training_part], dtype=np.float32).reshape(-1, 9), np.array(X[training_part:],  dtype=np.float32).reshape(-1,9)
        ytr, yts = np.array(Y[:training_part],  dtype=np.float32).reshape(-1,1), np.array(Y[training_part:],  dtype=np.float32).reshape(-1,1)

        print("[--] X shape: {} (training: {}, test: {})".format(X.shape, Xtr.shape, Xts.shape))
        Xtr, Xts = self.preprocess_x(Xtr, Xts)

        return torch.from_numpy(Xtr), torch.from_numpy(ytr), torch.from_numpy(Xts), torch.from_numpy(yts)


__all__ = ('CASP', 'Magic', 'HTRU2')