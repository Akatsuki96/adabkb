import numpy as np

class ExpansionProcedure:
    """Function which takes a node::PartitionNode and returns a set of node obtained by partitioning the cell associated to the node.
    """
    
    def __call__(self, node):
        pass



class GreedyExpansion(ExpansionProcedure):

    """Expansion procedure which consists in splitting cells along their longest side.

    Parameters
    ----------
    seed : int
        integer used to initialize a random number generator
    random_dim : bool
        if True the side to split is randomly choosen otherwise the longest side is choosen
    random_split : bool
        if True partition size are randomly choosen otherwise the split is uniform
    """

    def __init__(self, seed = 4242, random_dim = False, random_split = False):
        self.random_state = np.random.RandomState(seed)
        self.random_dim = random_dim
        self.random_split = random_split

    def _get_side_to_split(self, node):
        side_lengths = np.abs(node.partition[:, 1] - node.partition[:, 0])
        max_len = np.max(side_lengths)
        if self.random_dim:
            return self.random_state.choice(range(0, node.partition.shape[0]), 1)
        return np.argmax(side_lengths)

    def _standard_split(self, id, node):
        max_len = float(np.abs(node.partition[:, 1][id] - node.partition[:, 0][id])/node.N) 
        children = []
        for i in range(node.N):
            new_partition = node.partition.copy()
            lb = new_partition[:, 0]
            ub = new_partition[:, 1]
            lb[id] = float(node.partition[:, 0][id] + max_len * float(i))
            ub[id] = float(node.partition[:, 0][id] + max_len * float(i + 1))
            children.append((new_partition, node.index*node.N + i ))
        return children

    def _random_split(self, id, node):
        m, M = node.partition[:, 0][id], node.partition[:, 1][id]
        pivots = (M - m) * self.random_state.random(node.N) + m
        pivots.sort()
        lenghts = [piv - m for piv in pivots]
        children = []
        last_lb = m
        for i in range(node.N):
            new_partition = node.partition.copy()
            lb, ub = new_partition[:, 0], new_partition[:, 1] 
            lb[id] = last_lb
            ub[id] = pivots[i]
            children.append((new_partition, node.index*node.N + i)) 
            last_lb = ub[id]
        return children

    def __call__(self, node):
        id = self._get_side_to_split(node)
        if self.random_split:
            return self._random_split(id, node)
        return self._standard_split(id, node)
    

