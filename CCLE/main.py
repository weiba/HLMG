import argparse
import pickle
import numpy as np
import pandas as pd
from load_data import load_data
from models import hlmg, Optimizer_mul
from sklearn.model_selection import KFold
from sampler import RandomSampler_yz
from myutils import *
import logging
from brics import *
import pandas as pd
import warnings
#warnings.filterwarnings('ignore')


#================== 声明 ==========================

parser = argparse.ArgumentParser(description="Run HLMG")
parser.add_argument('-device', type=str, default="cuda:0", help='cuda:number or cpu')
parser.add_argument('-data', type=str, default='ccle', help='Dataset{gdsc or ccle}')
parser.add_argument('--lr', type=float,default=0.001,
                    help="the learning rate")
parser.add_argument('--wd', type=float,default=1e-5,
                    help="the weight decay for l2 normalizaton")
parser.add_argument('--gamma', type=float,default=8.7,
                    help="the scale for sigmod")
parser.add_argument('--epochs', type=float,default=1000,
                    help="the epochs for model")
args = parser.parse_args()

logging.basicConfig(
    format='%(message)s',
    filename='mylog.log',
    level=logging.INFO
)

#===================== data prepare ================
gene_path = './HLMG/CCLE/Data'
with open(gene_path + '/34pathway_score990.pkl', 'rb') as file:
    kegg = pickle.load(file)
PPI = pd.read_csv(gene_path+'/CCLE_2369_PPI_990.csv', index_col=0)
cell_exp = pd.read_csv(gene_path+'/cell_exp_2369.csv', index_col=0)

G, X, models_params, pathway_contain = creat_G_and_X_and_models_prarams(kegg, PPI, cell_exp)

res_ic50, res_binary, drug_finger, exprs, null_mask, pos_num, args = load_data(args)
cell_sim_f = calculate_gene_exponent_similarity7(x=torch.from_numpy(exprs).to(dtype=torch.float32, device='cuda:0'), mu=3)
drug_sim_f = jaccard_coef7(tensor=torch.from_numpy(drug_finger).to(dtype=torch.float32, device='cuda:0'))

#========================= load brics  ================

da1 = pd.read_csv('./HLMG/CCLE/Data/ccle_drug_smiles.csv',index_col=0)
smis = da1['smile'].values
drug_subfeat = []; drug_fragments = {}; SMARTS = []
masks = []
for smi in smis:
    sub_smi, sm  = BRICS_GetMolFrags(smi)
    sub_features = [np.array(get_Morgan(item)) for item in sub_smi]
    mask1 = np.ones(len(sub_features)).reshape(1,-1)
    mask2 = np.zeros(20-len(sub_features)).reshape(1,-1)
    mask = np.concatenate((mask1,mask2),axis=1)
    masks.append(mask)
    sub_features = np.concatenate((sub_features,np.zeros((20-len(sub_features),512))),axis=0)
    drug_subfeat.append(np.array(sub_features))
    SMARTS.append(sm)
    drug_fragments[str(smi)] = sub_smi
drug_e = torch.tensor(drug_subfeat,device='cuda:0')
drug_mask = torch.tensor(masks)
drug_mask = drug_mask.squeeze(dim=1)


k = 5
n_kfolds = 5
aucs = []
auprs = []
true_datas = pd.DataFrame()
predict_datas = pd.DataFrame()


#==================================dataset split==============================
np.random.seed(2023) 
a = np.arange(pos_num)
b = np.random.choice(a, size=169, replace=False)
ind = np.zeros(1696, dtype=bool)
ind[b] = True
a = a[~ind]
pos_adj_mat = null_mask + res_binary
neg_adj_mat = sp.coo_matrix(np.abs(pos_adj_mat - np.array(1)))
all_row = neg_adj_mat.row
all_col = neg_adj_mat.col
all_data = neg_adj_mat.data
index = np.arange(all_data.shape[0])
c = np.random.choice(index, 169, replace=False)
kfold = KFold(n_splits=k, shuffle=True, random_state=2023)

for n_kfold in range(n_kfolds):
    for train_index, val_index in kfold.split(a):
        train_i = np.zeros(len(a), dtype=bool)
        train_i[train_index] = True
        val_i = np.zeros(len(a), dtype=bool)
        val_i[val_index] = True
        sampler = RandomSampler_yz(res_ic50, res_binary, a[train_i], a[val_i], null_mask,b,c)

#================================== end ====================================
        gip_input = sampler.train_data.to_dense().numpy()
        cell_sim_gip = torch.from_numpy(gaussian_kernel_matrix(gip_input)).to(dtype=torch.float32, device='cuda:0')
        drug_sim_gip = torch.from_numpy(gaussian_kernel_matrix(gip_input.T)).to(dtype=torch.float32, device='cuda:0')

        cell_sim = 0.5*cell_sim_gip+0.5*cell_sim_f
        drug_sim = 0.5*drug_sim_gip+0.5*drug_sim_f

        model = hlmg(adj_mat=sampler.train_data,  cell_exprs=exprs, drug_finger=drug_finger,
                        gamma=args.gamma, models_params=models_params,X=X,
                    G=G,drug_sub=drug_e,drug_mask=drug_mask,
                    drug_sim = drug_sim, cell_sim = cell_sim,device=args.device).to(args.device)

        opt = Optimizer_mul(model, sampler.train_ic50, sampler.val_ic50, sampler.train_data, sampler.val_data, sampler.val_mask, sampler.train_mask,
                            sampler.test_data,sampler.test_mask,sampler.test_ic50,
                roc_auc,ap_score, lr=args.lr, wd=args.wd, epochs=args.epochs, device=args.device).to(args.device)
        
        auc,aupr, true_data, best_predict = opt()
        true_datas = true_datas.append(translate_result(true_data))
        predict_datas = predict_datas.append(translate_result(best_predict))
        auc, aupr, _, _, _ = evaluate_all(true_data, best_predict)
        aucs.append(auc)
        print('current auc and time :',np.mean(aucs),len(aucs))
        auprs.append(aupr)

print('auc : ',np.mean(aucs))
print('aupr : ',np.mean(auprs))
logging.info(f"final_auc:{np.mean(aucs):.4f},var:{np.var(aucs)},final_aupr:{np.mean(auprs):.4f},var:{np.var(auprs)}")
cv_set, test_set = save_dataset(sampler.test_mask,null_mask,res_binary)
cv_set.to_csv("./HLMG/CCLE/Data/5-fold_CV.csv",index=False)
test_set.to_csv("./HLMG/CCLE/Data/testset.csv",index=False)

pd.DataFrame(true_datas).to_csv("./HLMG/CCLE/result_data/true_data.csv")
pd.DataFrame(predict_datas).to_csv("./HLMG/CCLE/result_data/predict_data.csv")
