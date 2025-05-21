import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from econml.dml import CausalForestDML
from xbart import XBART

# Implementations from Fawkes et al. 2022 in https://github.com/MrHuff/doubly_robust
# Average treatment effect baselines. 

def calculate_pval_symmetric(bootstrapped_list, test_statistic):
    pval_right = 1 - 1 / (bootstrapped_list.shape[0] + 1) * (1 + (bootstrapped_list <= test_statistic).sum())
    pval_left = 1 - pval_right
    pval = 2 * min([pval_left.item(), pval_right.item()])
    return pval

class CausalForest_baseline_test():
    def __init__(self,X_tr,T_tr,Y_tr,bootstrap=250):
        self.n,self.d=X_tr.shape
        self.est=CausalForestDML(discrete_treatment=True,drate=True)
        self.est.fit(X=X_tr,T=T_tr,Y=Y_tr,)
        self.ref_stat = self.est.ate(X_tr)[0]
        self.bootstrap=bootstrap
        self.X_tr = X_tr
        self.T_tr = T_tr
        self.Y_tr = Y_tr

    def permutation_test(self):
        inf_object = self.est.ate__inference()
        return inf_object.pvalue()[0][0], self.ref_stat
    
    

class BART_baseline_test():
    def __init__(self,X_tr,T_tr,Y_tr,bootstrap=250):
        x_tr_0, y_tr_0, x_tr_1, y_tr_1 = self.sep_dat(X_tr,Y_tr,T_tr)
        self.ref_stat = self.fit_BART(x_tr_0, y_tr_0, x_tr_1, y_tr_1 )
        self.bootstrap=bootstrap
        self.X_tr = X_tr
        self.T_tr = T_tr
        self.Y_tr = Y_tr

    def fit_BART(self,x_tr_0,y_tr_0,x_tr_1, y_tr_1):
        est_0 = XBART(num_trees=100, num_sweeps=10, burnin=1)
        est_1 = XBART(num_trees=100, num_sweeps=10, burnin=1)
        est_0.fit(x_tr_0, y_tr_0.squeeze())
        est_1.fit(x_tr_1, y_tr_1.squeeze())
        return self.calc_effect(x_tr_0,x_tr_1,est_0,est_1)


    def calc_effect(self,x0,x1,est_0,est_1):
        ref_mat_0_0 = est_0.predict(x0)
        ref_mat_0_1 = est_0.predict(x1)
        ref_mat_1_0 = est_1.predict(x0)
        ref_mat_1_1 = est_1.predict(x1)
        pred_0=np.concatenate([ref_mat_0_0,ref_mat_0_1],axis=0)
        pred_1=np.concatenate([ref_mat_1_0,ref_mat_1_1],axis=0)
        #concat vectors then take mean
        return (pred_1-pred_0).mean()

    def sep_dat(self,X,Y,T):
        mask_0 = (T==0).squeeze()
        x_tr_0,y_tr_0= X[mask_0,:],Y[mask_0]
        x_tr_1,y_tr_1= X[~mask_0,:],Y[~mask_0]
        return x_tr_0,y_tr_0,x_tr_1,y_tr_1

    def sep_val(self,X,T):
        mask_0 = (T==0).squeeze()
        x_tr_0 = X[mask_0,:]
        x_tr_1= X[~mask_0,:]
        return x_tr_0,x_tr_1
    #
    # def boostrap_data(self):
    #     ind_0=np.random.randint(0,self.x_test_0.shape[0],self.boostrap_size)
    #     ind_1=np.random.randint(0,self.x_test_1.shape[0],self.boostrap_size)
    #     return self.x_test_0[ind_0,:],self.x_test_1[ind_1,:]

    def permutation_test(self):
        effect_list=[]
        for i in range(self.bootstrap):
            Y = np.random.permutation(self.Y_tr)

            x_tr_0, y_tr_0, x_tr_1, y_tr_1 = self.sep_dat(self.X_tr, Y, self.T_tr)
            stat = self.fit_BART(x_tr_0, y_tr_0, x_tr_1, y_tr_1)
            effect_list.append(stat)

        effect_list=np.array(effect_list)
        pval=calculate_pval_symmetric(effect_list,self.ref_stat)

        return pval,self.ref_stat
    
    
def doubly_robust(df, X, T, Y):
    ps = LogisticRegression(C=1e6, max_iter=1000).fit(df[X], df[T]).predict_proba(df[X])[:, 1]
    mu0 = LinearRegression().fit(df.query(f"{T}==0")[X], df.query(f"{T}==0")[Y]).predict(df[X])
    mu1 = LinearRegression().fit(df.query(f"{T}==1")[X], df.query(f"{T}==1")[Y]).predict(df[X])
    return (
        np.mean(df[T]*(df[Y] - mu1)/ps + mu1) -
        np.mean((1-df[T])*(df[Y] - mu0)/(1-ps) + mu0)
    )

class vanilla_dr_baseline_test():
    def __init__(self,X,T,Y,n_bootstraps):
        self.n,self.d=X.shape
        columns=[f'x_{i}' for i in range(self.d)] +['D']+['Y']
        # self.cov_string =''
        # for i in range(self.d):
        #     self.cov_string+=f' + x_{i}'
        self.X,self.T,self.Y = X,T,Y
        self.columns = columns
        self.dfs = pd.DataFrame(np.concatenate([X,T,Y],axis=1),columns=columns)
        self.n_bootstraps = n_bootstraps
        self.x_col = [f'x_{i}' for i in range(self.d)]
        self.ref_stat = doubly_robust(self.dfs,X=self.x_col,T='D',Y='Y')
    #TODO, when permuting just go back to what we did
    def permutation_test(self):
        rd_results = []
        for i in range(self.n_bootstraps):
            Y = np.random.permutation(self.Y)
            s = pd.DataFrame(np.concatenate([self.X,self.T,Y],axis=1),columns=self.columns)
            stat = doubly_robust(s,self.x_col,'D','Y')
            rd_results.append(stat)
        rd_results = np.array(rd_results)
        pval=calculate_pval_symmetric(rd_results,self.ref_stat)
        return pval,self.ref_stat

