# Ada-BKB: Adaptive Budgeted Kernelized Bandit
Implementation of Ada-BKB.

## Install
First, you need to install dependences and you can do it with
`pip install -r requirements.txt`
Then
`pip install .`

To repeat the hyperparameter tuning experiment you have to install Falkon[https://github.com/FalkonML/falkon] and download the datasets:

- HTRU: https://archive.ics.uci.edu/ml/datasets/HTRU2
- CASP: https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure
- Magic04: https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope

## Citation
If you use this library, please cite it as below.
~~~
@misc{rando2021adabkb,
      title={Ada-BKB: Scalable Gaussian Process Optimization on Continuous Domain by Adaptive Discretization}, 
      author={Marco Rando and Luigi Carratino and Silvia Villa and Lorenzo Rosasco},
      year={2021},
      eprint={2106.08598},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
~~~