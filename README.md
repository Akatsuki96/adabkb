# Ada-BKB: Adaptive Budgeted Kernelized Bandit
Implementation of **Ada-BKB: Scalable Gaussian Process Optimization on Continuous Domain by Adaptive Discretization** published in AISTATS 2022. A preprint of the paper can be found at the following link: https://arxiv.org/abs/2106.08598

## Install
First, you need to install dependences and you can do it with
```
pip install -r requirements.txt
```
Then
```
pip install .
```
To repeat paper experiments, other dependeces are needed. You can find the instruction to run paper experiments in *papers_experiment* folder

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