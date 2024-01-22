from models import hlmg_new, Optimizer_new
from myutils import *



def HLMG_new( cell_exprs, drug_finger, sam,gip,test_da,test_ma,
                args,models_params,X,G,drug_sub,drug_mask,
                        cell_sim,drug_sim,device):
        gip_input = gip
        cell_sim_gip = torch.from_numpy(gaussian_kernel_matrix(gip_input)).to(dtype=torch.float32, device='cuda:0')
        drug_sim_gip = torch.from_numpy(gaussian_kernel_matrix(gip_input.T)).to(dtype=torch.float32, device='cuda:0')
        cell_sim = 0.5*cell_sim_gip+0.5*cell_sim
        drug_sim = 0.5*drug_sim_gip+0.5*drug_sim

        model = hlmg_new(adj_mat=sam.train_data,  cell_exprs=cell_exprs, drug_finger=drug_finger,
                gamma=args.gamma, models_params=models_params,X=X,G=G,
                drug_sub=drug_sub,drug_mask=drug_mask,
                        drug_sim=drug_sim,cell_sim=cell_sim,device=args.device).to(args.device)

        opt = Optimizer_new(model,  sam.train_data, sam.test_data, sam.test_mask,sam.train_mask,
                        test_da, test_ma,roc_auc,ap_score,
                        lr=args.lr, wd=args.wd, epochs=args.epochs,tol=args.tol, device=args.device).to(args.device)

        best_auc,best_aupr, true_data,predict_data= opt()

        return best_auc,best_aupr,true_data, predict_data
