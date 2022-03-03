import numpy as np
from adabkb.utils import diagonal_dot

from adabkb.options import OptimizerOptions


<<<<<<< HEAD
class AdaptiveOptimizer:
    
    def __init__(self, options):
=======


class AbsOptimizer:
    def __init__(self, options: OptimizerOptions = None):
>>>>>>> a82973072a828de84640becd1014a956bae86e78
        self.options = options
        self.node2idx = {}
        self.num_nodes = 0
        
    def get_node_idx(self, node):
        return self.node2idx[hash(node)]

    def register_nodes(self, nodes):
        new_regs = []
        node_idx = []
        for node in nodes:
            if hash(node) not in self.node2idx:
                self.node2idx[hash(node)] = self.num_nodes
                new_regs.append(node.x)
                node_idx.append(self.num_nodes)
                self.num_nodes += 1
            else:
                node_idx.append(self.node2idx[hash(node)])
     #           print("[--] node: {} Already reg!!!!!!!!".format(node.x))
        return np.asarray(new_regs), np.asarray(node_idx)
        
        
    def step(self):
        pass
    
    def update(self, idx, ys):
        pass