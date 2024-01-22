HLMG
===============================
Source code and data for "Hierarchical graph representation learning with multi-granularity features for anti-cancer drug response prediction"

![Framework of HLMG](https://github.com/weiba/HLMG/HLMG.png)  
# Requirements
- python==3.10.8
- pytorch==1.13.0
- numpy==1.23.4+mkl
- scipy==1.9.3
- pandas==1.5.1
- scikit-learn=1.1.3
- pubchempy==1.0.4
- hickle==5.0.2
- pubchempy==1.0.4
- rdkit==2023.3.3
# Usage
- python main.py {Classification task}
- python main_ic50.py {Regression task}
- python main_new.py  {New drugs or cell lines task, You need to define whether to clear rows or columns by setting t_dim=0 or 1}

All *main*.py files can complete a single experiment. Because of the randomness of dividing test data and training data, we recorded the true value of the test data during the algorithm performance. Therefore, the output of the main file includes the true and predicted values of the test data that have been cross-validated many times. In the subsequent statistical analysis, we analyze the output of the main file. The myutils.py file contains all the tools needed for the performance and analysis of the entire experiment, such as the calculation of AUC, AUPRC,  ACC, F1 score, and MCC. All functions are developed using PyTorch and support CUDA.

# Contact
If you have any question regard our code or data, please do not hesitate to open a issue or directly contact me (weipeng1980@gmail.com).
