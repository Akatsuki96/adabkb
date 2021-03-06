import numpy as np
from .expansion_procedure import ExpansionProcedure

class PartitionTreeNode:

    """Implementation of a partition tree. It is used to explore the search space.
    Parameters
    ----------
    partition : numpy array
        an \\(d \\times 2\\) array in which each entry [l, u] represent a lower bound (l) and an upper bound (u).
        It represent the cell associated to the node.
    N : int
        number of children obtained expanding the node.
    father : PartitionTreeNode
        father node. If the node is a root this can will be None.
    level : int
        level in the tree. It is initialized with 0.
    index : int
        index in a level of the tree. It is initialized with 0.
    expansion_procedure : ExpansionProcedure
        function called when the node is expanded.  
    """

    def __init__(self, partition : np.ndarray, N: int, father, level : int = 0, index : int = 0, expansion_procedure : ExpansionProcedure = ExpansionProcedure()):
        self.partition = partition
        self.N = N
        self.index = index
        self.father = father
        self.level = level
        self.children = []
        self.expand_fun = expansion_procedure
        self.x = self.partition.mean(axis=1)


    def adapt(self, x):
        adapted = False
        for i in range(self.partition.shape[0]):
            if self.partition[i][0] > x[i]:
                x[i] = self.partition[i][0]
                adapted = True
            elif self.partition[i][1] < x[i]:
                x[i] = self.partition[i][1]
                adapted = True
        return x, adapted

    def expand_node(self):
        """function which creates and add children to the tree.
        Returns
        -------
        children : List<PartitionTreeNode>
            list containing children.
        """
        children = self.expand_fun(self)
        for (partition,i) in children:
            self.children.append(PartitionTreeNode(partition, self.N, self, self.level + 1, i, self.expand_fun))
        return self.children


    def __repr__(self):
        return "({}, N = {}, level: {})".format(self.partition, self.N, self.level)

    def __eq__(self, other_node):
        if isinstance(other_node, PartitionTreeNode):
            return np.all(self.partition == other_node.partition)
        return False
    
    def __hash__(self):
        return hash(tuple(self.x))