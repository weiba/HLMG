# coding: utf-8
import sys
sys.path.append("./.")
import argparse
from load_data import load_data
from myutils import *
from HLMG.GDSC.HLMG_New import HLMG_new
import copy
import pickle
import pandas as pd
import numpy as np
import pandas as pd
from load_data import load_data
from sklearn.model_selection import KFold
from sampler import RandomSampler
from myutils import *
import logging
from brics import *


parser = argparse.ArgumentParser(description="Run HLMG")
parser.add_argument('-device', type=str, default="cuda:0", help='cuda:number or cpu')
parser.add_argument('-data', type=str, default='gdsc', help='Dataset{gdsc or ccle}')
parser.add_argument('--lr', type=float,default=0.005,
                    help="the learning rate")
parser.add_argument('--wd', type=float,default=1e-5,
                    help="the weight decay for l2 normalizaton")
parser.add_argument('--gamma', type=float,default=8.7,
                    help="the scale for sigmod")
parser.add_argument('--epochs', type=float,default=1000,
                    help="the epochs for model")
parser.add_argument('--tol', type=float,default=50,
                    help="early stop count for model")

args = parser.parse_args()


logging.basicConfig(
    format='%(message)s',
    filename='mylog.log',
    level=logging.INFO
)

#===================== data prepare ================
gene_path = './HLMG/GDSC/Data'
with open(gene_path + '/34pathway_score990.pkl', 'rb') as file:
    kegg = pickle.load(file)
PPI = pd.read_csv(gene_path+'/2369_PPI_990.csv', index_col=0)
cell_exp = pd.read_csv(gene_path+'/cell_exp_2369.csv', index_col=0)

G, X, models_params, pathway_contain = creat_G_and_X_and_models_prarams(kegg, PPI, cell_exp)

res_ic50, cell_drug, drug_finger, exprs, null_mask1, pos_num, args = load_data(args)


#========================= load brics  ================

da1 = pd.read_csv('./HLMG/GDSC/Data/smile_inchi.csv',index_col=0)
smis = da1['smiles'].values
drug_subfeat = []; drug_fragments = {}; SMARTS = []
masks = []
for smi in smis:
    sub_smi, sm  = BRICS_GetMolFrags(smi)
    sub_features = [np.array(get_Morgan(item)) for item in sub_smi]
    mask1 = np.ones(len(sub_features)).reshape(1,-1)
    mask2 = np.zeros(20-len(sub_features)).reshape(1,-1)
    mask_12 = np.concatenate((mask1,mask2),axis=1)
    masks.append(mask_12)
    sub_features = np.concatenate((sub_features,np.zeros((20-len(sub_features),512))),axis=0)
    drug_subfeat.append(np.array(sub_features))
    SMARTS.append(sm)
    drug_fragments[str(smi)] = sub_smi
drug_e = torch.tensor(drug_subfeat,device='cuda:0')
drug_mask = torch.tensor(masks)
drug_mask = drug_mask.squeeze(dim=1)


ori_null_mask = copy.deepcopy(null_mask1)
cell_sim = calculate_gene_exponent_similarity7(x=torch.from_numpy(exprs).to(dtype=torch.float32, device='cuda:0'), mu=3)
drug_sim = jaccard_coef7(tensor=torch.from_numpy(drug_finger).to(dtype=torch.float32, device='cuda:0'))




#====================================dataset=========================================
def update_Adjacency_matrix (A, test_samples):
    m = test_samples.shape[0]
    A_tep = A.copy()
    for i in range(m):
        if test_samples[i,2] ==1:
            A_tep [int(test_samples[i,0]), int(test_samples[i,1])] = 0
    return A_tep
kf = KFold(n_splits=5, shuffle=True,random_state=2023)
kfold = KFold(n_splits=5, shuffle=True,random_state=2023)
x, y = cell_drug.shape

t_dim = 0
final_auc_d = []
final_aupr_d = []


for train_index, test_index in kf.split(np.arange(x)): # t_dim==0 : x  ;  t_dim==1 : y.
    null_mask = copy.deepcopy(null_mask1)
    train_p = copy.deepcopy(cell_drug)
    train_n = copy.deepcopy(cell_drug+null_mask - 1)
    if t_dim == 0:
    # row
        train_p[test_index, :] = 0
        train_n[test_index, :] = 0
    else:
    # col
        args.lr = 0.05
        args.tol = 20
        train_p[:, test_index] = 0
        train_n[:, test_index] = 0
    train_p = sp.coo_matrix(train_p)
    train_n = sp.coo_matrix(train_n)
    train_pos = list(zip(train_p.row, train_p.col, train_p.data))
    train_neg = list(zip(train_n.row, train_n.col, train_n.data + 1))
    train_neg.extend(train_pos)
    train_samples = np.array(train_neg)
    test_p = copy.deepcopy(cell_drug)
    test_n = copy.deepcopy(cell_drug+null_mask - 1)
    if t_dim == 0:
    # row
        test_p[train_index, :] = 0
        test_n[train_index, :] = 0
    else:
    # col
        test_p[:, train_index] = 0
        test_n[:, train_index] = 0
        

    test_p = sp.coo_matrix(test_p)
    test_n = sp.coo_matrix(test_n)
    test_pos = list(zip(test_p.row, test_p.col, test_p.data))
    test_neg = list(zip(test_n.row, test_n.col, test_n.data + 1))
    test_neg.extend(test_pos)
    test_samples = np.array(test_neg)
    new_A = update_Adjacency_matrix(cell_drug, test_samples)
    if t_dim == 0:
    # row
        null_mask[test_index, :] = 1
    else:
    # col
        null_mask[:, test_index] = 1
#===================== negtive sampler ====================

    neg_adj_mat = copy.deepcopy(cell_drug+ ori_null_mask)
    neg_adj_mat = np.abs(neg_adj_mat - np.array(1))
    if t_dim == 0:
    # row
        neg_adj_mat[train_index, :] = 0
    else:
    # col
        neg_adj_mat[:, train_index] = 0

    neg_adj_mat = sp.coo_matrix(neg_adj_mat)
    pos_adj_mat = copy.deepcopy(cell_drug)
    if t_dim == 0:
    # row
        pos_adj_mat[train_index, :] = 0
    else:
    # col
        pos_adj_mat[:, train_index] = 0

    all_row = neg_adj_mat.row
    all_col = neg_adj_mat.col
    all_data = neg_adj_mat.data
    index = np.arange(all_data.shape[0])
    # 采样负测试集
    test_n = int(pos_adj_mat.sum())
    test_neg_index = np.random.choice(index, test_n, replace=False)
    test_row = all_row[test_neg_index]
    test_col = all_col[test_neg_index]
    test_data = all_data[test_neg_index]
    test = sp.coo_matrix((test_data, (test_row, test_col)), shape=cell_drug.shape)
    train = sp.coo_matrix(pos_adj_mat)
    banlence_mask = mask(train, test, dtype=bool)      # independent test set

#==========================================================================

    for train_ind, test_ind in kfold.split(np.arange(sum(sum(new_A)))):

        true_data_s = pd.DataFrame()
        predict_data_s = pd.DataFrame()
        sam = RandomSampler(new_A, train_ind, test_ind, null_mask) #  cross-validation set

        auc,aupr,true_data, predict_data = HLMG_new(cell_exprs=exprs,drug_finger=drug_finger,
                                                            sam=sam,gip=new_A,test_da=cell_drug,test_ma=banlence_mask,
                                                            args=args,models_params=models_params,X=X,G=G,
                                                            drug_sub=drug_e,drug_mask=drug_mask,
                cell_sim=cell_sim,drug_sim=drug_sim,device=args.device,)

        true_data_s = pd.concat([true_data_s,translate_result(true_data)])
        predict_data_s = pd.concat([predict_data_s,translate_result(predict_data)])

        final_auc_d.append(auc)
        final_aupr_d.append(aupr)
        print('creent auc :',np.mean(final_auc_d))


if t_dim == 0:
    print('new_cell_auc:',np.mean(final_auc_d),np.var(final_auc_d))
    print('new_cell_aupr:',np.mean(final_aupr_d),np.var(final_aupr_d))
else:
    print('new_drug_auc:',np.mean(final_auc_d),np.var(final_auc_d))
    print('new_drug_aupr:',np.mean(final_aupr_d),np.var(final_aupr_d))

