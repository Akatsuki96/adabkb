import numpy as np
from adabkb.optimizer import AdaptiveOptimizer
from adabkb.surrogate import BKB

from adabkb.partition_tree import PartitionTreeNode


class AdaBKB(AdaptiveOptimizer):
        
    def __init__(self, search_space, options):
        super().__init__(options)
        self.model = BKB(**options.model_options)
        root = PartitionTreeNode(
            partition = search_space, 
            N = self.options.N, 
            father = None, 
            expansion_procedure = self.options.expand_fun
        )
        self.num_eval = 0
        self.leaf_set = np.array([root]) #search_space
        self.node_idx = np.zeros(1, dtype=int)
        self.best_lcb = (None, -np.inf, -1) # (best_x, best_lcb)

        self.I = np.zeros(1)
        self.Vh = np.array([self.compute_Vh(i) for i in range(self.h_max+1)], dtype=float)
        self.register_nodes([root])
        self.model.initialize(root.x)
   
    @property
    def g(self):
        return self.model.kernel.confidence_function
   
    @property
    def ucb(self):
        return self.model.ucbs
    
    @property
    def lcb(self):
        return self.model.lcbs
   
    @property
    def means(self):
        return self.model.means
    
    @property
    def stds(self):
        return np.sqrt(self.model.variances)
    
    @property
    def beta(self):
        return self.model.beta
   
    @property
    def h_max(self):
        return self.options.h_max
   
    def compute_Vh(self, level):
        return self.g(self.options.v_1 * pow(self.options.rho, level))

    def __can_be_expanded(self, node_idx, level):
  #      print(self.beta * self.stds[node_idx], self.Vh[level])
        return self.beta * self.stds[node_idx] <= self.Vh[level] and level < self.h_max

    def __select_candidate(self):
        return np.argmax(self.I)

    def __expand_leaf(self, node, leaf_idx, node_idx):
        children = node.expand_node()
        zeros = np.zeros(len(children))
        self.leaf_set = np.concatenate((np.delete(self.leaf_set, leaf_idx), children))
        self.I = np.concatenate((np.delete(self.I, leaf_idx), zeros))
        new_nodes, nodes_idx = self.register_nodes(children)
        self.model.extend_arm_set(new_nodes)
        self.node_idx = np.concatenate((np.delete(self.node_idx, leaf_idx), nodes_idx))
        new_leaf_idx = list(range(self.leaf_set.shape[0] - len(children) , self.leaf_set.shape[0]))
        self.__update_I(new_leaf_idx)
    #    self.__prune_leafset(np.asarray(new_leaf_idx))
   #     print("-----------------------------------------------------")

    
    def __prune_leafset(self, leaf_idx):
                
        node_idx = self.node_idx[leaf_idx]
        levels = np.asarray([node.level for node in self.leaf_set[leaf_idx]])

        subopt_partitions = (self.I[leaf_idx] < self.best_lcb[1]) & (self.node_idx[leaf_idx] != self.best_lcb[2])
        idx_to_erase = leaf_idx[subopt_partitions]
#        idx_to_erase = idx_to_erase[self.node_idx[subopt_partitions] != self.best_lcb[2]]
        
        
        print("[PRUNING] IDX: {}".format(leaf_idx))   
        print("[--] node_idx: {}".format(self.node_idx[leaf_idx]))
        print("[--] subopt: {}".format(subopt_partitions))
        print("[--] best lcb: {}".format(self.best_lcb))
     #   print("[--] SUBOPT: {}".format(subopt_partitions))
        print("[--] idx to erase: {}".format(idx_to_erase)) 
        
        self.leaf_set = np.delete(self.leaf_set, idx_to_erase)
        self.I = np.delete(self.I, idx_to_erase)
        self.node_idx = np.delete(self.node_idx, idx_to_erase)
         
    def step(self):
        while True:
            leaf_idx = self.__select_candidate()
            node = self.leaf_set[leaf_idx]
            node_idx = self.get_node_idx(node)
            print("[!!] SELECTED:  node: {}\tleaf: {}".format(node_idx, leaf_idx))
            if self.__can_be_expanded(node_idx, node.level) and (node.level > 0 or self.num_eval > 0):
                print("-------------------- EXPAND ----------------")
                self.__expand_leaf(node, leaf_idx, node_idx)
            else:
                # Evaluation step
                print("-------------------- EVAL ----------------")
                return node, leaf_idx

    def __update_I(self, leaf_indices):
        father_idx = [] 
        levels = []
        for node in self.leaf_set[leaf_indices]:
            father_idx.append(self.get_node_idx(node.father))
            levels.append(node.level)


        father_idx, levels = np.asarray(father_idx), np.asarray(levels)
      
        self.I[leaf_indices] = np.min([self.model.ucbs[self.node_idx[leaf_indices]], self.model.ucbs[father_idx] + self.Vh[levels - 1]], axis=0) + self.Vh[levels]


    def __update_best_lcb(self, xs, node_indices, leaf_indices):
        for i in range(node_indices.shape[0]):
            if self.best_lcb[0] is None or node_indices[i] == self.best_lcb[2] or self.best_lcb[1] < self.lcb[node_indices[i]]:
                self.best_lcb = (xs[i], self.lcb[node_indices[i]], node_indices[i])   
    
    def update(self, leaf_idx, xs, ys):
        xs = np.asarray(xs)
        idx = np.asarray([self.node_idx[i] for i in leaf_idx])
        leaf_indices = np.asarray(leaf_idx)
        self.num_eval+=1
        if idx == 0 and self.leaf_set[0].level==0:
            self.model.full_update(idx, np.asarray([ys]))
            self.I[0] = self.model.ucbs[0] + self.Vh[0] 
        else:
            self.model.update_emb(idx, np.asarray([ys]))
            self.model.update_mean_variances(self.node_idx)
            self.__update_I(list(range(len(self.leaf_set))))
        self.__update_best_lcb(xs, idx, leaf_indices)
        if self.leaf_set[0].level > 0:
            self.__prune_leafset(np.array(list(range(self.leaf_set.shape[0]))))
