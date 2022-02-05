import numpy as np
from sklearn import metrics


class Metric:
    def __init__(self, name):
        self.name = name
        
    def __call__(self):
        raise NotImplemented()
        
class OneMinusAUC(Metric):
    def __init__(self):
        super().__init__("1 - AUC")
        
    def __call__(self, y_true, y_pred):
        if not isinstance(y_true, np.ndarray):
            y_true: np.ndarray = y_true.numpy()
        if not isinstance(y_pred, np.ndarray):
            y_pred: np.ndarray = y_pred.numpy()

        fpr, tpr, thresholds = metrics.roc_curve(
            y_true.reshape((-1, 1)), y_pred.reshape((-1, 1)), pos_label=1)
        return 1 - metrics.auc(fpr, tpr)
        
class CERR(Metric):
    def __init__(self):
        super().__init__("CERR")
        
    def __call__(self, y_true, y_pred):
        y_true = y_true.cpu()
        y_pred = y_pred.cpu()
        if not isinstance(y_true, np.ndarray):
            y_true: np.ndarray = y_true.numpy()
        if not isinstance(y_pred, np.ndarray):
            y_pred: np.ndarray = y_pred.numpy()
            


        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1) * 2 - 1
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1) * 2 - 1

        return np.mean(np.sign(y_pred.ravel()) != np.sign(y_true.ravel()))

class MSE(Metric):
    def __init__(self):
        super().__init__("MSE")
        
    def __call__(self, y_true, y_pred):
        if not isinstance(y_true, np.ndarray):
            y_true: np.ndarray = y_true.numpy()
        if not isinstance(y_pred, np.ndarray):
            y_pred: np.ndarray = y_pred.numpy()
        return np.mean((y_true - y_pred)**2)   


__all__ = ('OneMinusAUC', 'CERR', 'MSE')