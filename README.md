# Ada-BKB: Adaptive Budgeted Kernelized Bandit
Implementation of **Ada-BKB: Scalable Gaussian Process Optimization on Continuous Domain by Adaptive Discretization** published in AISTATS 2022. 

A preprint of the paper can be found at the following link: https://arxiv.org/abs/2106.08598

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
@inproceedings{rando2022adabkb,
      title={Ada-BKB: Scalable Gaussian Process Optimization on Continuous Domain by Adaptive Discretization}, 
      author={Marco Rando and Luigi Carratino and Silvia Villa and Lorenzo Rosasco},
      year = {2022},
      booktitle = {(to appear in) Proceedings of The 25th International Conference on Artificial Intelligence and Statistics},
}
~~~