# Ada-BKB: Adaptive Budgeted Kernelized Bandit
Implementation of **Ada-BKB: Scalable Gaussian Process Optimization on Continuous Domain by Adaptive Discretization** published in AISTATS 2022. 

A preprint of the paper can be found at the following link: https://arxiv.org/abs/2106.08598

## Install
To install adabkb library, clone the repository and use pip
```
pip install .
```

## Citation
If you use this library, please cite it as below.
~~~
@InProceedings{pmlr-v151-rando22a,
  title = 	 { Ada-BKB: Scalable Gaussian Process Optimization on Continuous Domains by Adaptive Discretization },
  author =       {Rando, Marco and Carratino, Luigi and Villa, Silvia and Rosasco, Lorenzo},
  booktitle = 	 {Proceedings of The 25th International Conference on Artificial Intelligence and Statistics},
  pages = 	 {7320--7348},
  year = 	 {2022},
  editor = 	 {Camps-Valls, Gustau and Ruiz, Francisco J. R. and Valera, Isabel},
  volume = 	 {151},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {28--30 Mar},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v151/rando22a/rando22a.pdf},
  url = 	 {https://proceedings.mlr.press/v151/rando22a.html},
  abstract = 	 { Gaussian process optimization is a successful class of algorithms(e.g. GP-UCB) to optimize a black-box function through sequential evaluations. However, for functions with continuous domains, Gaussian process optimization has to rely on either a fixed discretization of the space, or the solution of a non-convex ptimization subproblem at each evaluation. The first approach can negatively affect performance, while the second approach requires a heavy computational burden. A third option, only recently theoretically studied, is to adaptively discretize the function domain. Even though this approach avoids the extra non-convex optimization costs, the overall computational complexity is still prohibitive. An algorithm such as GP-UCB has a runtime of $O(T^4)$, where $T$ is the number of iterations. In this paper, we introduce Ada-BKB (Adaptive Budgeted Kernelized Bandit), a no-regret Gaussian process optimization algorithm for functions on continuous domains, that provably runs in $O(T^2 d_\text{eff}^2)$, where $d_\text{eff}$ is the effective dimension of the explored space, and which is typically much smaller than $T$. We corroborate our theoretical findings with experiments on synthetic non-convex functions and on the real-world problem of hyper-parameter optimization, confirming the good practical performances of the proposed approach. }
}
~~~