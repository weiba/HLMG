import numpy as np
import torch

from myutils import *
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

def load_data(args):
    if args.data == 'gdsc':
        return _load_gdsc(args)
    elif args.data == 'ccle':
        return _load_ccle(args)
    else:
        raise NotImplementedError

def _load_gdsc(args):

    data_dir = "./HLMG_final/GDSC/Data/"

    # 加载细胞系-药物IC50反应矩阵
    res_ic50 = pd.read_csv(data_dir + "cell_drug.csv", index_col=0, header=0)
    res_ic50 = np.array(res_ic50, dtype=np.float32)

    # 加载细胞系-药物二值反应矩阵
    res = pd.read_csv(data_dir + "cell_drug_binary.csv", index_col=0, header=0)
    res = np.array(res, dtype=np.float32)
    pos_num = sp.coo_matrix(res).data.shape[0]

    # 加载药物-指纹特征矩阵
    drug_feature = pd.read_csv(data_dir + "drug_feature.csv", index_col=0, header=0)
    drug_feature = np.array(drug_feature, dtype=np.float32)

    # 加载细胞系-基因特征矩阵
    exprs= pd.read_csv(data_dir + "gene_feature.csv", index_col=0, header=0)
    exprs = np.array(exprs, dtype=np.float32)   
    
    # 加载null_mask
    null_mask = pd.read_csv(data_dir + "null_mask.csv", index_col=0, header=0)
    null_mask = np.array(null_mask, dtype=np.float32)

    return res_ic50, res, drug_feature, exprs, null_mask, pos_num, args

def _load_ccle(args):
    data_dir = "./HLMG/CCLE/Data/"

    # 加载细胞系-药物IC50反应矩阵
    res_ic50 = pd.read_csv(data_dir + "cell_drug.csv", index_col=0, header=0)
    res_ic50 = np.array(res_ic50, dtype=np.float32)


    # 加载细胞系-药物矩阵
    res = pd.read_csv(data_dir + "cell_drug_binary.csv", index_col=0, header=0)
    res = np.array(res, dtype=np.float32)
    pos_num = sp.coo_matrix(res).data.shape[0]

    # 加载药物-指纹特征矩阵
    drug_feature = pd.read_csv(data_dir + "drug_feature.csv", index_col=0, header=0)
    drug_feature = np.array(drug_feature, dtype=np.float32)

    # 加载细胞系-基因特征矩阵
    exprs = pd.read_csv(data_dir + "gene_feature.csv", index_col=0, header=0)
    exprs = np.array(exprs, dtype=np.float32)

    # 加载null_mask
    null_mask = np.zeros(res.shape, dtype=np.float32)
    return res_ic50 ,res, drug_feature, exprs, null_mask, pos_num, args


