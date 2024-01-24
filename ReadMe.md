HLMG
===============================
Source code and data for "Hierarchical graph representation learning with multi-granularity features for anti-cancer drug response prediction"

![Framework of HLMG](https://github.com/weiba/HLMG/blob/master/HLMG.png)  
# Requirements
All implementations of HLMG are based on PyTorch. HLMG requires the following dependencies:
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
# Data
- Data defines the data used by the model
	- GDSC/Data/
		- cell_drug.csv records the log IC50 association matrix of cell line-drug.
		- cell_drug_binary.csv records the binary cell line-drug association matrix.
		- gene_feature.csv records cell line gene expression features.
		- drug_feature.csv records the fingerprint features of drugs.
		- null_mask.csv records the null values in the cell line-drug association matrix.
		- threshold.csv records the drug sensitivity threshold.
        - smiles_inchi.csv records drug smiles.
        - cell_exp_2369.csv records the gene expression features used to construct the pathway substructure.
        - 34pathway_score990.pkl records pathway substructure PPI score.
        - 2369_PPI_990.csv records PPI network.
        - 5-fold_CV.csv records the 5-fold cross-validation set for the classification task.
        - testset.csv records the independent test set for the classification task.
        - 5-fold_CV_ic50.csv records the 5-fold cross-validation set for the regression task.
        - testset_ic50.csv records the independent test set for the regression task.
	- CCLE/Data/
		- cell_drug.csv records the log IC50 association matrix of cell line-drug.
		- cell_drug_binary.csv records the binary cell line-drug association matrix.
		- gene_feature.csv records cell line gene expression features.
		- drug_feature.csv records the fingerprint features of drugs.
        - ccle_drug_smiles.csv records drug smiles.
        - cell_exp_2369.csv records the gene expression features used to construct the pathway substructure.
        - 34pathway_score990.pkl records pathway substructure PPI score.
        - CCLE_2369_PPI_990.csv records PPI network.
        - 5-fold_CV.csv records the 5-fold cross-validation set for the classification task.
        - testset.csv records the independent test set for the classification task.
        - 5-fold_CV_ic50.csv records the 5-fold cross-validation set for the regression task.
        - testset_ic50.csv records the independent test set for the regression task.
- brics.py defines the drug substructure feature process.
- load_data.py defines the data loading of the model.
- model.py defines the complete HLMG model.
- myutils.py defines the tool functions needed to run the entire algorithm as well as the evaluation metrics.
- sampler.py defines the sampling method of the model.
## Preprocessing your own data
Explanations on how you can process your own data and prepare it for HLMG running.

> In our study, we followed the data preprocessing steps described in MOLI[1] , and the data preprocessing code was derived from [MOLI](https://github.com/hosseinshn/MOLI). The cell line pathway data followed the steps described in DRPreter[2], with data and code from [DRPreter](https://github.com/babaling/DRPreter).
> 
> [1]Sharifi-Noghabi, H., et al. MOLI: multi-omics late integration with deep neural networks for drug response prediction. Bioinformatics 2019;35(14):i501-i509.[2]Shin, J., et al. DRPreter: Interpretable Anticancer Drug Response Prediction Using Knowledge-Guided Graph Neural Networks and Transformer. International Journal of Molecular Sciences 2022;23(22):13919.

> You can download the processing data via the link above, or use the data provided in the GDSC/Data/ and CCLE/Data folders
# Usage
Once you have configured the environment, you can simply run HLMG by running command:
>To build and evaluate our model, we randomly select one-tenth of the positive samples and the same number of negative samples as the independent test set, and the rest of the data as the cross-validation set. We split the dataset in the ***sampler.py*** file and in the ***main.py*** file.

For classification tasks, you can use the following command:
```
python main.py
```
For regression tasks, you can use the following command:
```
python main_ic50.py
```
For the new drug or new cell line task, you just need to set up t_dim=0 or 1 in main_new.py to run the following command:
```
python main_new.py
```
Once the program has finished running, you are able to see the evaluation results of the independent test set directly on the screen! At the same time the program saves the test set's predictions and true labels in . /result_data/, and saves the test set and cross-validation set in ./Data/

All of the above commands complete a single experiment. The output of the main file consists of the true and predicted values of the test set as well as the divided test set and cross-validation set. In the subsequent statistical analysis, we will analyze the output of the master file. myutils.py file contains the performance of the whole experiment and all the tools needed for the analysis, such as the computation of AUC, AUPRC, PCC, SCC and RMSE. All functions are developed using PyTorch with CUDA support.

If you want to evaluate the test set results later, you can use the following command:
```
python evaluate.py
```
# Contact
If you have any question regard our code or data, please do not hesitate to open a issue or directly contact me (weipeng1980@gmail.com).
