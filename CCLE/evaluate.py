import sys
sys.path.append("./.")
from myutils import evaluate_all
import pandas as pd
import numpy as np
import torch

data_dir = "./result_data/"
auc_all = []
ap_all = []
acc_all = []
f1_all = []
mcc_all = []
for i in np.arange(25):
    predict_data = pd.read_csv(data_dir + "predict_data.csv", index_col=0, header=0, nrows=1, skiprows=i)
    predict_data = predict_data.dropna(axis=1,how='any')
    predict_data = np.array(predict_data)[0]
    predict_data = torch.tensor(predict_data)
    true_data = pd.read_csv(data_dir + "true_data.csv", index_col=0, header=0, nrows=1, skiprows=i)
    true_data = true_data.dropna(axis=1,how='any')
    true_data = np.array(true_data)[0]
    true_data = torch.tensor(true_data)
    auc, ap, acc, f1, mcc = evaluate_all(true_data, predict_data) #This is for the classification task. If you want to evaluate the regression task, replace it with the evaluate_regression function in the myutils.py
    auc_all.append(auc)
    ap_all.append(ap)
    acc_all.append(acc)
    f1_all.append(f1)
    mcc_all.append(mcc)
file_write_obj = open(data_dir +"result_all", 'a+')
file_write_obj.writelines("|final auc: {}|final ap: {}|final acc: {}|final f1: {}|final mcc: {}"
                          .format(str(np.mean(auc_all)), str(np.mean(ap_all)), str(np.mean(acc_all)), str(np.mean(f1_all)), str(np.mean(mcc_all))))
file_write_obj.write('\n')
file_write_obj.writelines("|final auc_var: {}|final ap_var: {}|final acc_var: {}|final f1_var: {}|final mcc_var: {}"
                          .format(str(np.var(auc_all)), str(np.var(ap_all)), str(np.var(acc_all)), str(np.var(f1_all)), str(np.var(mcc_all))))
file_write_obj.write('\n')
file_write_obj.close()