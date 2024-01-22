import pandas as pd
import torch
import torch.nn.functional as fun
import torch.nn as nn
from abc import ABC
import torch.optim as optim
from myutils import *

class ConstructAdjMatrix(nn.Module, ABC):
    def __init__(self, original_adj_mat, device="cpu"):
        super(ConstructAdjMatrix, self).__init__()
        self.adj = original_adj_mat.to(device)
        self.device = device

    def forward(self):
        d_x = torch.diag(torch.pow(torch.sum(self.adj, dim=1)+1, -0.5))
        d_y = torch.diag(torch.pow(torch.sum(self.adj, dim=0)+1, -0.5))

        agg_cell_lp = torch.mm(torch.mm(d_x, self.adj), d_y)
        agg_drug_lp = torch.mm(torch.mm(d_y, self.adj.T), d_x)

        d_c = torch.pow(torch.sum(self.adj, dim=1)+1, -1)
        self_cell_lp = torch.diag(torch.add(d_c, 1))
        d_d = torch.pow(torch.sum(self.adj, dim=0)+1, -1)
        self_drug_lp = torch.diag(torch.add(d_d, 1))
        return agg_cell_lp, agg_drug_lp, self_cell_lp, self_drug_lp


class LoadFeature(nn.Module, ABC):
    def __init__(self, cell_exprs, drug_finger, device="cpu"):
        super(LoadFeature, self).__init__()
        cell_exprs = torch.from_numpy(cell_exprs).to(device)
        self.cell_feat = torch_z_normalized(cell_exprs,dim=1).to(device)
        self.drug_feat = torch.from_numpy(drug_finger).to(device)

    def forward(self):
        cell_feat = self.cell_feat
        drug_feat = self.drug_feat
        return cell_feat, drug_feat
    
class Pathway_GCN(nn.Module):
    def __init__(self, models_params,cell_one_exp, lp_g):
        super(Pathway_GCN, self).__init__()
        
        # Create a list of model sequences
        model_seqs = [nn.Sequential(
            nn.Linear(params["input_size"], params["hidden_size"]),
            nn.ReLU(),
            nn.Linear(params["hidden_size"], params["output_size"])
        ) for params in models_params]

        # Create a module list from the model sequences
        self.models = nn.ModuleList(model_seqs).to("cuda:0")
        self.model_num = len(cell_one_exp)
        self.exp = cell_one_exp
        self.lpls = lp_g
        
    def forward(self):
        
        output_cal_l = []
        for i in range(self.model_num):
            c_exp = torch.mm(self.lpls[i],self.exp[i])
            out = self.models[i](c_exp.T)
            output_cal_l.append(out)
            
        return output_cal_l
    
class att_fus_pathway(nn.Module):
    def __init__(self,input_dim,att_dim,sum):
        super(att_fus_pathway, self).__init__()

        self.super = nn.Parameter(torch.randn(att_dim, 1))
        self.fc = nn.Linear(input_dim, att_dim, bias=True)
        self.ac = nn.ReLU()
        self.soft = nn.Softmax(dim=1)
        self.sum = sum
        self.out = nn.Linear(input_dim, 1020, bias=True)
        self.bn = nn.BatchNorm1d(1020)
    def forward(self, x):
        uit = self.ac(self.fc(x)) 
        alp = torch.matmul(uit, self.super)
        ait = self.soft(alp)
        weit = x * ait
        if self.sum == True:
            out = torch.sum(weit, axis=1)
        else:
            out = weit.reshape(x.shape[0],-1)
        return out, ait
    
class att_fus_sub(nn.Module):
    def __init__(self,input_dim,att_dim,sum):
        super(att_fus_sub, self).__init__()
        
        self.super = nn.Parameter(torch.randn(att_dim, 1))
        self.fc = nn.Linear(input_dim, att_dim, bias=True)
        self.ac = nn.ReLU()
        self.soft = nn.Softmax(dim=1)
        self.sum = sum
        self.out = nn.Linear(input_dim, 1020, bias=True)
        self.bn = nn.BatchNorm1d(1020)
    def forward(self, x,mask):
        ex_e_mask = (1.0 - mask) * -10000.0
        uit = self.ac(self.fc(x))
        alp= torch.matmul(uit, self.super)
        alpp = alp + ex_e_mask
        ait = self.soft(alpp)
        weit = x * ait
        if self.sum == True:
            out = torch.sum(weit, axis=1)
        else:
            out = weit.reshape(x.shape[0],-1)
        return out, ait

class SimLayer(nn.Module, ABC):
    def __init__(self, sim: torch.Tensor, knn: int, in_dim: int, embed_dim: int, act=fun.relu, bias=False):
        super(SimLayer, self).__init__()
        self.sim = sim
        self.knn = knn
        self.diffuse_lm = nn.Linear(in_dim, embed_dim, bias=bias)
        self.act = act
        self.laplace = self.__calculate_laplace()

    @staticmethod
    def k_near_graph(sim: torch.Tensor, k: int):
        """
        Calculate the k near graph as feature space adjacency.
        :param sim: similarity matrix, torch.Tensor
        :param k: k, int
        :return: weighted adjacency matrix
        """
        threshold = torch.min(torch.topk(sim, k=k, dim=1).values, dim=1).values.view([-1, 1])
        sim = torch.where(sim.ge(threshold), sim, torch.zeros_like(sim))
        return sim

    @staticmethod
    def diffuse_laplace(adj: torch.Tensor):
        d_x = torch.diag(torch.pow(torch.sum(adj, dim=1), -0.5))
        d_y = torch.diag(torch.pow(torch.sum(adj, dim=0), -0.5))
        adj = torch.mm(torch.mm(d_x, adj), d_y)
        return adj

    def __calculate_laplace(self):
        adj = SimLayer.k_near_graph(sim=self.sim, k=self.knn)
        diffuse_x_laplace = SimLayer.diffuse_laplace(adj=adj)
        return diffuse_x_laplace

    def forward(self, x: torch.Tensor):
        out = self.act(self.diffuse_lm(torch.mm(self.laplace, x)))
        return out
    


class GDecoder(nn.Module, ABC):
    def __init__(self, gamma):
        super(GDecoder, self).__init__()
        self.gamma = gamma

    def forward(self, cell_emb, drug_emb):
        Corr = torch_corr_x_y(cell_emb, drug_emb)
        output = scale_sigmoid(Corr, alpha=self.gamma)
        return output
    

class GDecoder_regression(nn.Module, ABC):
    def __init__(self,gamma):
        super(GDecoder_regression, self).__init__()
        self.gamma = gamma

    def forward(self, cell_emb, drug_emb):
        output = torch.mul(torch_corr_x_y(cell_emb, drug_emb),self.gamma)
        return output

class Hierarchical_Agg(nn.Module):
    def __init__(self, adj_mat,adj_mat_T,self_c,self_d, cell_sim,drug_sim,sim_balance):
        super(Hierarchical_Agg, self).__init__()
        
        self.adj_mat = adj_mat
        self.adj_mat_T = adj_mat_T
        self.lp_c = self_c
        self.lp_d = self_d

        self.cell_sim = cell_sim
        self.drug_sim = drug_sim
        
        self.in_dim = 2040
        self.out_dim = 2040
        self.sim_c = SimLayer(sim = self.cell_sim, knn=7, in_dim=self.in_dim , embed_dim=self.out_dim )
        self.sim_d = SimLayer(sim = self.drug_sim, knn=7, in_dim=self.in_dim , embed_dim=self.out_dim )
        
        self.agg_cs = nn.Linear(self.in_dim ,self.out_dim ,bias=True)   
        self.agg_c1 = nn.Linear(self.in_dim ,self.out_dim ,bias=True)    
        self.agg_c2 = nn.Linear(self.in_dim ,self.out_dim ,bias=True)
        
        self.agg_ds = nn.Linear(self.in_dim ,self.out_dim ,bias=True)
        self.agg_d1 = nn.Linear(self.in_dim ,self.out_dim ,bias=True)
        self.agg_d2 = nn.Linear(self.in_dim ,self.out_dim ,bias=True)

    def forward(self,exp,fig):
        
        sf_d = self.sim_d(fig)
        sf_c = self.sim_c(exp)
        
        cells_1 = self.agg_cs(torch.mm(self.lp_c, exp))
        cells_r = torch.mul(cells_1, exp)
        cells = cells_1 + cells_r
        cell1_1 = self.agg_c1(torch.mm(self.adj_mat, fig))
        cell1_r = torch.mul(cell1_1, exp)
        cell1 = cell1_1 + cell1_r
        cell2_1 = self.agg_c2(torch.mm(self.adj_mat, sf_d))
        cell2_r = torch.mul(cell2_1, exp)
        cell2 = cell2_1 + cell2_r
        
        drugs_1 = self.agg_ds(torch.mm(self.lp_d, fig))
        drugs_r = torch.mul(drugs_1, fig) 
        drugs = drugs_1 + drugs_r
        drug1_1 = self.agg_d1(torch.mm(self.adj_mat_T, exp))
        drug1_r = torch.mul(drug1_1, fig) 
        drug1 = drug1_1 + drug1_r
        drug2_1 = self.agg_d2(torch.mm(self.adj_mat_T, sf_c))
        drug2_r = torch.mul(drug2_1, fig) 
        drug2 = drug2_1 + drug2_r
        
        cellagg = fun.relu(cells + cell1 + cell2)
        drugagg = fun.relu(drugs + drug1 + drug2)

        return cellagg, drugagg

class Hierarchical_Agg_new(nn.Module):
    def __init__(self, adj_mat,adj_mat_T,self_c,self_d, cell_sim,drug_sim,sim_balance):
        super(Hierarchical_Agg_new, self).__init__()
        
        self.adj_mat = adj_mat
        self.adj_mat_T = adj_mat_T
        self.lp_c = self_c
        self.lp_d = self_d

        self.cell_sim = cell_sim
        self.drug_sim = drug_sim
        
        self.in_dim = 2040
        self.out_dim = 2040
        self.sim_c = SimLayer(sim = self.cell_sim, knn=7, in_dim=self.in_dim , embed_dim=self.out_dim )
        self.sim_d = SimLayer(sim = self.drug_sim, knn=7, in_dim=self.in_dim , embed_dim=self.out_dim )
        
        self.agg_cs = nn.Linear(self.in_dim ,self.out_dim ,bias=True)   
        self.agg_c1 = nn.Linear(self.in_dim ,self.out_dim ,bias=True)    
        self.agg_c2 = nn.Linear(self.in_dim ,self.out_dim ,bias=True)
        
        self.agg_ds = nn.Linear(self.in_dim ,self.out_dim ,bias=True)
        self.agg_d1 = nn.Linear(self.in_dim ,self.out_dim ,bias=True)
        self.agg_d2 = nn.Linear(self.in_dim ,self.out_dim ,bias=True)

    def forward(self,exp,fig):
        
        sf_d = self.sim_d(fig)
        sf_c = self.sim_c(exp)
        
        cells_1 = 1*self.agg_cs(torch.mm(self.lp_c, exp))
        cells_r = torch.mul(cells_1, exp)
        cells = cells_1 + cells_r
        cell1_1 = 0.01*self.agg_c1(torch.mm(self.adj_mat, fig))
        cell1_r = torch.mul(cell1_1, exp)
        cell1 = cell1_1 + cell1_r
        cell2_1 = 0.001*self.agg_c2(torch.mm(self.adj_mat, sf_d))
        cell2_r = torch.mul(cell2_1, exp)
        cell2 = cell2_1 + cell2_r
        
        drugs_1 = 1*self.agg_ds(torch.mm(self.lp_d, fig))
        drugs_r = torch.mul(drugs_1, fig) 
        drugs = drugs_1 + drugs_r
        drug1_1 = 0.01*self.agg_d1(torch.mm(self.adj_mat_T, exp))
        drug1_r = torch.mul(drug1_1, fig) 
        drug1 = drug1_1 + drug1_r
        drug2_1 = 0.001*self.agg_d2(torch.mm(self.adj_mat_T, sf_c))
        drug2_r = torch.mul(drug2_1, fig) 
        drug2 = drug2_1 + drug2_r
        
        cellagg = fun.relu(cells + cell1 + cell2)
        drugagg = fun.relu(drugs + drug1 + drug2)

        return cellagg, drugagg

class hlmg(nn.Module, ABC):
    def __init__(self, adj_mat, cell_exprs, drug_finger,  gamma,models_params, X, G,drug_sub,drug_mask,
                 drug_sim, cell_sim,
                 device="cpu"):
        super(hlmg, self).__init__()
        
        construct_adj_matrix = ConstructAdjMatrix(adj_mat, device=device)  
        loadfeat = LoadFeature(cell_exprs, drug_finger, device=device)
        agg_cell_lp, agg_drug_lp, self_cell_lp, self_drug_lp = construct_adj_matrix()
        cell_feat,drug_feat = loadfeat()
        self.cexp = cell_feat
        self.figprint = drug_feat
        self.drug_sub = drug_sub.float().to("cuda:0")
        self.drug_mask = drug_mask.reshape(drug_finger.shape[0],20,1).float().to("cuda:0")
        
        self.ldd = nn.Linear(self.figprint.shape[1],2040-512,bias=True).to(device)
        self.lcc = nn.Linear(cell_exprs.shape[1],1020,bias=True).to(device)
        self.pgcn = Pathway_GCN(models_params, X, G).to("cuda:0")
        self.d_att = att_fus_sub(512,256,sum=True)
        self.c_att = att_fus_pathway(30,128,sum=False)


        self.agg_h = Hierarchical_Agg(adj_mat=agg_cell_lp,adj_mat_T=agg_drug_lp,self_c=self_cell_lp,self_d=self_drug_lp,
                                        cell_sim=cell_sim,drug_sim=drug_sim,sim_balance=1).to(device)
        self.decoder = GDecoder(gamma=gamma)

        self.bnd1 = nn.BatchNorm1d(2040-512)
        self.bnd2 = nn.BatchNorm1d(512)
        self.bnc1 = nn.BatchNorm1d(1020)
        self.bnc2 = nn.BatchNorm1d(1020)
    
    def forward(self):

        sub_emb1,_ = self.d_att(self.drug_sub,self.drug_mask)
        sub_emb = self.bnd2(sub_emb1)
        sub_emb_1 = self.bnd1(self.ldd(self.figprint))
        drug_x = torch.cat([sub_emb_1,sub_emb],dim=1) 

        cell_x = self.pgcn()  
        cell_x = torch.stack(cell_x,dim=0)
        cell_x = cell_x.permute(1,0,2)
        tran_out1,_ = self.c_att(cell_x)  
        tran_out  = self.bnc2(tran_out1)
        sub_cell = self.bnc1(self.lcc(self.cexp))
        cell_emb_3_t = torch.cat([sub_cell,tran_out],dim=1)
           
        cell_emb_3_t_out, drug_x_out = self.agg_h(cell_emb_3_t,drug_x)
        output = self.decoder(cell_emb_3_t_out, drug_x_out)   

        return output
    
class hlmg_ic50(nn.Module, ABC):
    def __init__(self, adj_mat, cell_exprs, drug_finger,  gamma,models_params, X, G,drug_sub,drug_mask,
                 drug_sim, cell_sim,
                 device="cpu"):
        super(hlmg_ic50, self).__init__()
        
        construct_adj_matrix = ConstructAdjMatrix(adj_mat, device=device)  
        loadfeat = LoadFeature(cell_exprs, drug_finger, device=device)
        agg_cell_lp, agg_drug_lp, self_cell_lp, self_drug_lp = construct_adj_matrix()
        cell_feat,drug_feat = loadfeat()
        self.cexp = cell_feat
        self.figprint = drug_feat
        self.drug_sub = drug_sub.float().to("cuda:0")
        self.drug_mask = drug_mask.reshape(drug_finger.shape[0],20,1).float().to("cuda:0")
        
        self.ldd = nn.Linear(self.figprint.shape[1],2040-512,bias=True).to(device)
        self.lcc = nn.Linear(cell_exprs.shape[1],1020,bias=True).to(device)
        self.pgcn = Pathway_GCN(models_params, X, G).to("cuda:0")
        self.d_att = att_fus_sub(512,256,sum=True)
        self.c_att = att_fus_pathway(30,128,sum=False)


        self.agg_h = Hierarchical_Agg(adj_mat=agg_cell_lp,adj_mat_T=agg_drug_lp,self_c=self_cell_lp,self_d=self_drug_lp,
                                        cell_sim=cell_sim,drug_sim=drug_sim,sim_balance=1).to(device)
        self.decoder = GDecoder_regression(gamma=gamma)

        self.bnd1 = nn.BatchNorm1d(2040-512)
        self.bnd2 = nn.BatchNorm1d(512)
        self.bnc1 = nn.BatchNorm1d(1020)
        self.bnc2 = nn.BatchNorm1d(1020)
    
    def forward(self):

        sub_emb1,_ = self.d_att(self.drug_sub,self.drug_mask)
        sub_emb = self.bnd2(sub_emb1)
        sub_emb_1 = self.bnd1(self.ldd(self.figprint))
        drug_x = torch.cat([sub_emb_1,sub_emb],dim=1) 

        cell_x = self.pgcn()  
        cell_x = torch.stack(cell_x,dim=0)
        cell_x = cell_x.permute(1,0,2)
        tran_out1,_ = self.c_att(cell_x)  
        tran_out  = self.bnc2(tran_out1)
        sub_cell = self.bnc1(self.lcc(self.cexp))
        cell_emb_3_t = torch.cat([sub_cell,tran_out],dim=1)
           
        cell_emb_3_t_out, drug_x_out = self.agg_h(cell_emb_3_t,drug_x)
        output = self.decoder(cell_emb_3_t_out, drug_x_out)   

        return output

class Optimizer_mul(nn.Module):
    def __init__(self, model,  train_ic50, test_ic50, train_data, test_data, test_mask, train_mask, 
                 independ_data,ind_mask,ind_ic50,evaluate_fun,evluate_fun2,lr=0.01, wd=1e-05, epochs=200, test_freq=20, device="cpu"):
        super(Optimizer_mul, self).__init__()
        self.model = model.to(device)

        self.train_ic50= train_ic50.to(device)
        self.test_ic50 = test_ic50.to(device)
        self.train_data = train_data.to(device)
        self.test_data = test_data.to(device)
        self.train_mask = train_mask.to(device)
        self.test_mask = test_mask.to(device)
        
        self.ind_ic50 = ind_ic50.to(device)
        self.ind_data = independ_data.to(device)
        self.ind_mask = ind_mask.to(device)
        
        self.evaluate_fun = evaluate_fun
        self.evaluate_fun2 = evluate_fun2
        self.lr = lr
        self.wd = wd

        self.epochs = epochs
        self.test_freq = test_freq
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

    def forward(self):
        best_auc = 0
        tole = 0          
        
        true_data = torch.masked_select(self.test_data, self.test_mask)
        ind_true_data = torch.masked_select(self.ind_data, self.ind_mask)

        for epoch in torch.arange(self.epochs+1):
            predict_data= self.model()
            loss = cross_entropy_loss(self.train_data, predict_data, self.train_mask)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            ind_data_masked = torch.masked_select(predict_data, self.ind_mask)
            predict_data_masked = torch.masked_select(predict_data, self.test_mask)
            auc = self.evaluate_fun(true_data, predict_data_masked)        

            if auc > best_auc:
                best_auc = auc
                best_predict = torch.masked_select(predict_data, self.ind_mask)
                ind_auc = self.evaluate_fun(ind_true_data, ind_data_masked)
                ind_aupr = self.evaluate_fun2(ind_true_data, ind_data_masked)
                tole = 0
            else:
                tole +=1
                
            if epoch % self.test_freq == 0:
                print("epoch:%4d" % epoch.item(), "loss:%.6f" % loss.item(), "auc:%.4f" % auc,"ind_auc:%.4f" % ind_auc)
        
        print("Fit finished.")

        return ind_auc,ind_aupr, ind_true_data, best_predict

class hlmg_new(nn.Module, ABC):
    def __init__(self, adj_mat, cell_exprs, drug_finger,  gamma,models_params, X, G,drug_sub,drug_mask,
                 drug_sim, cell_sim,
                 device="cpu"):
        super(hlmg_new, self).__init__()
        
        construct_adj_matrix = ConstructAdjMatrix(adj_mat, device=device)  
        loadfeat = LoadFeature(cell_exprs, drug_finger, device=device)
        agg_cell_lp, agg_drug_lp, self_cell_lp, self_drug_lp = construct_adj_matrix()
        cell_feat,drug_feat = loadfeat()
        self.cexp = cell_feat
        self.figprint = drug_feat
        self.drug_sub = drug_sub.float().to("cuda:0")
        self.drug_mask = drug_mask.reshape(drug_finger.shape[0],20,1).float().to("cuda:0")
        
        self.ldd = nn.Linear(self.figprint.shape[1],2040-512,bias=True).to(device)
        self.lcc = nn.Linear(cell_exprs.shape[1],1020,bias=True).to(device)
        self.pgcn = Pathway_GCN(models_params, X, G).to("cuda:0")
        self.d_att = att_fus_sub(512,256,sum=True)
        self.c_att = att_fus_pathway(30,128,sum=False)


        self.agg_h = Hierarchical_Agg_new(adj_mat=agg_cell_lp,adj_mat_T=agg_drug_lp,self_c=self_cell_lp,self_d=self_drug_lp,
                                        cell_sim=cell_sim,drug_sim=drug_sim,sim_balance=1).to(device)
        self.decoder = GDecoder(gamma=gamma)

        self.bnd1 = nn.BatchNorm1d(2040-512)
        self.bnd2 = nn.BatchNorm1d(512)
        self.bnc1 = nn.BatchNorm1d(1020)
        self.bnc2 = nn.BatchNorm1d(1020)
    
    def forward(self):

        sub_emb1,_ = self.d_att(self.drug_sub,self.drug_mask)
        sub_emb = self.bnd2(sub_emb1)
        sub_emb_1 = self.bnd1(self.ldd(self.figprint))
        drug_x = torch.cat([sub_emb_1,sub_emb],dim=1) 

        cell_x = self.pgcn()  
        cell_x = torch.stack(cell_x,dim=0)
        cell_x = cell_x.permute(1,0,2)
        tran_out1,_ = self.c_att(cell_x)  
        tran_out  = self.bnc2(tran_out1)
        sub_cell = self.bnc1(self.lcc(self.cexp))
        cell_emb_3_t = torch.cat([sub_cell,tran_out],dim=1)
           
        cell_emb_3_t_out, drug_x_out = self.agg_h(cell_emb_3_t,drug_x)
        output = self.decoder(cell_emb_3_t_out, drug_x_out)   

        return output 
    
class Optimizer_new(nn.Module):
    def __init__(self, model,  train_data, test_data, test_mask, train_mask,ind_data,ind_mask,
                 evaluate_fun, evaluate_fun2,
                 lr=0.001, wd=1e-05, epochs=200, tol=50,test_freq=20, device="cpu"):#
        super(Optimizer_new, self).__init__()
        self.model = model.to(device)

        self.train_data = train_data.to(device)
        self.test_data = test_data.to(device)
        self.train_mask = train_mask.to(device)
        self.test_mask = test_mask.to(device)
        
        self.evaluate_fun = evaluate_fun
        self.evaluate_fun2 = evaluate_fun2



        self.lr = lr
        self.early_stop_count = tol
        self.wd = wd

        self.epochs = epochs
        self.test_freq = test_freq
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

        self.ind_data = torch.from_numpy(ind_data).float().to(device)
        self.ind_mask = ind_mask.to(device)


    def forward(self):
        best_auc = 0
        best_aupr = 0
        tol = 0
        tol_auc = 0
        true_data = torch.masked_select(self.test_data, self.test_mask)
        train_da = torch.masked_select(self.ind_data, self.ind_mask)
        
        
        for epoch in torch.arange(self.epochs + 1):

            predict_data = self.model()
            loss_binary = cross_entropy_loss(self.train_data, predict_data, self.train_mask)
            loss =  loss_binary
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            predict_ind_masked = torch.masked_select(predict_data, self.ind_mask)
            predict_data_masked = torch.masked_select(predict_data, self.test_mask)
            train_auc = self.evaluate_fun(train_da, predict_ind_masked)
            train_aupr = self.evaluate_fun2(train_da, predict_ind_masked)
            auc = self.evaluate_fun(true_data, predict_data_masked)

            if tol != self.early_stop_count:
                if auc > tol_auc:
                    tol = 0
                    tol_auc = auc
                    best_auc = train_auc

                    best_predict = predict_ind_masked
                    best_aupr = train_aupr
                else:
                    tol += 1
            else:
                break
            if epoch % self.test_freq == 0:
                print("epoch:%4d" % epoch.item(), "loss:%.4f" % loss.item(), "auc:%.4f" % best_auc,"aupr:%.4f" % best_aupr,"yanzheng:%.4f" % auc)

        print("Fit finished.")
        return best_auc, best_aupr, train_da, best_predict#true_data
    



    
class Optimizer_mul_ic50(nn.Module):
    def __init__(self, model,  train_ic50, test_ic50, train_data, test_data, test_mask, train_mask, 
                 independ_data,ind_mask,ind_ic50,
                 evaluate_fun,sc_score,rmse_score,
                 lr=0.01, wd=1e-05, epochs=200, test_freq=20, device="cpu"):
        super(Optimizer_mul_ic50, self).__init__()
        self.model = model.to(device)

        self.train_ic50= train_ic50.to(device)
        
        self.test_ic50 = test_ic50.to(device)
        self.train_data = train_data.to(device)
        
        self.test_data = test_data.to(device)
        self.train_mask = train_mask.to(device)
        self.test_mask = test_mask.to(device)
        
        self.ind_ic50 = ind_ic50.to(device)
        self.ind_data = independ_data.to(device)
        self.ind_mask = ind_mask.to(device)

        self.evaluate_fun2 = evaluate_fun
        self.evaluate_scc = sc_score
        self.evaluate_rmse  = rmse_score
        self.lr = lr
        self.wd = wd
        
        self.epochs = epochs
        self.test_freq = test_freq
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

    def forward(self):

        best_pcc = 0
        ind_pcc =0
        ind_scc =0
        ind_rmse =0
        tole = 0
        pcc = 0
        
        true_data = torch.masked_select(self.test_ic50, self.test_mask)
        ind_true_data = torch.masked_select(self.ind_ic50, self.ind_mask)

        for epoch in torch.arange(self.epochs+1):
            predict_data= self.model()
            loss = mse_loss(self.train_ic50, predict_data, self.train_mask)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
                
            ind_data_masked = torch.masked_select(predict_data, self.ind_mask)
            predict_data_masked = torch.masked_select(predict_data, self.test_mask)
            pcc = self.evaluate_fun2(true_data, predict_data_masked)        

            if pcc > best_pcc:
                best_pcc = pcc
                best_predict = torch.masked_select(predict_data, self.ind_mask)
                ind_pcc = self.evaluate_fun2(ind_true_data, ind_data_masked)
                ind_scc = self.evaluate_scc(ind_true_data, ind_data_masked)
                ind_rmse = self.evaluate_rmse(ind_true_data, ind_data_masked)
                tole = 0
            else:
                tole +=1


            if epoch % self.test_freq == 0:
                print("epoch:%4d" % epoch.item(), "loss:%.6f" % loss.item(), "pcc:%.4f" % pcc,"ind_pcc:%.4f" % ind_pcc)
        
        print("Fit finished.")
        return ind_pcc,ind_scc,ind_rmse, ind_true_data, best_predict