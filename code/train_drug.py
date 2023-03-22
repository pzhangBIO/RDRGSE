import numpy as np
import random
import torch
import torch.nn.functional as F
from cogsl import Cogsl
import warnings
from copy import deepcopy
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import scipy.sparse as sp

import pandas as pd 
import numpy as np
from utils import sparse_mx_to_torch_sparse_tensor,normalize_adj
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import math
class Link_Prediction(nn.Module):
    def __init__(self, n_representation, hidden_dims=[128, 32], dropout=0.3):
        super(Link_Prediction, self).__init__()
        self.n_representation = n_representation
        self.linear1 = nn.Linear(2*self.n_representation, hidden_dims[0])
        self.linear2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.linear3 = nn.Linear(hidden_dims[1], 2)
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax()
        self.init_weights()

    def forward(self, x):
       # x = torch.cat((x1,x2),1) # N * (2 node_dim)
        
        x = F.relu(self.linear1(x)) # N * hidden1_dim
        x = self.dropout(x)
        x = F.relu(self.linear2(x)) # N * hidden2_dim
        x = self.dropout(x)
        x = self.linear3(x) # N * 2
        x = self.softmax(x) # N * ( probility of each event )
        return x

    def init_weights(self):
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)
        init.xavier_uniform_(self.linear3.weight)

class DataSet():
    def __init__(self, dataset, x, y, view1, view2, view1_indice, view2_indice, idx_train, idx_val, idx_test):
        self.dataset = dataset
        self.x = x
        self.y = y
        self.view1 = view1
        self.view2 = view2
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.num_node = x.size(0)
        self.num_feature = x.size(1)
        self.num_class = int(torch.max(y)) + 1
        self.v1_indices = view1_indice
        self.v2_indices = view2_indice

    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.view1 = self.view1.to(device)
        self.view2 = self.view2.to(device)
        self.v1_indices = self.v1_indices.to(device)
        self.v2_indices = self.v2_indices.to(device)
        return self

    def normalize(self, adj):
        if self.dataset in ["wikics", "ms"]:
            adj_ = (adj + adj.t())
            normalized_adj = adj_
        else:
            adj_ = (adj + adj.t())
            normalized_adj = self._normalize(adj_ + torch.eye(adj_.shape[0]).to(adj.device).to_sparse())
        return normalized_adj

    def _normalize(self, mx):
        mx = mx.to_dense()
        rowsum = mx.sum(1) + 1e-6  # avoid NaN
        r_inv = rowsum.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx.to_sparse()

def convert_num():
    pi=pd.read_csv('../dataset/nc_drug/ncRNA.csv',header=None)
    dis=pd.read_csv('../dataset/nc_drug/drug.csv',header=None)
    data=pd.read_csv('../dataset/nc_drug/nc_drug.csv',header=None)
    all_node=pd.concat([pi,dis],axis=0,ignore_index=True)
    all_node[1]=[i for i in range(len(all_node))]
    d=dict(zip(all_node[0],all_node[1]))
    data=data.replace(d)
    positive=list(zip(data[0],data[1]))
    return positive,d,pi[0].tolist()
def generate_negative(pi, dis, pos_samples):
    """ generate negative sample by random change one of a pairs"""
    pairs = []
    for i in pi:
        for j in dis:
            if (i,j) not in pos_samples:
                pairs.append((i,j))
    print("len(ne_pairs)",len(pairs)) 
    return list(set(pairs))
def merge_samples(p_samples, n_samples):
    pos_sample=[]
    neg_sample=[]
    for w in p_samples:
        pos_sample.append(w)

    for i in n_samples:
        neg_sample.append(i)
    import random
    neg_sample=random.sample(neg_sample,len(pos_sample))
    samples=pos_sample+neg_sample
    pos_lab=[int(1) for i in range(len(pos_sample))]
    neg_lab=[int(0) for i in range(len(pos_sample))]
    label=pos_lab+neg_lab
    print(len(pos_sample),len(neg_sample))

    return samples,label

def load_data(data_path):
    feature = sp.load_npz(data_path+"/feat.npz")
    feature = feature.todense()
    
    #feature = preprocess_features(feature)
    feature = torch.FloatTensor(np.array(feature))


    positive,d,ncrna=convert_num()
    pi=[]
    dis=[]
    for i,j in d.items():
        if i in ncrna:
            pi.append(j)
        else:
            dis.append(j)
    neg=generate_negative(pi, dis, positive)
    sample,label=merge_samples(positive,neg)
    sample=pd.DataFrame(sample)
    sample['label']=label
    from sklearn.utils import shuffle
    sample=shuffle(sample)

    import math
    all_=set(sample.index.tolist())
    label=sample['label'].tolist()
    label = np.array(label)
    label = torch.LongTensor(label)

    ori_view1 = sp.load_npz(data_path+"/v1_knn"+".npz")       
    ori_view2 = sp.load_npz(data_path+"/v2_diff"+".npz")
    ori_view1_indice = torch.load(data_path+"/v1_2"+".pt")
    ori_view2_indice = torch.load(data_path+"/v2_40"+".pt")
    
    ori_view1 = sparse_mx_to_torch_sparse_tensor(normalize_adj(ori_view1)) 
    ori_view2 = sparse_mx_to_torch_sparse_tensor(normalize_adj(ori_view2))
    ####sample很重要
    return all_,sample,feature,label,ori_view1,ori_view2,ori_view1_indice,ori_view2_indice
def gen_auc_mima(logits, label):
        preds = torch.argmax(logits, dim=1)
        test_f1_macro = f1_score(label.cpu(), preds.cpu(), average='macro')
        test_f1_micro = f1_score(label.cpu(), preds.cpu(), average='micro')
        
        best_proba = F.softmax(logits, dim=1)
        if logits.shape[1] != 2:
            auc = roc_auc_score(y_true=label.detach().cpu().numpy(),
                                                    y_score=best_proba.detach().cpu().numpy(),
                                                    multi_class='ovr'
                                                    )
        else:
            auc = roc_auc_score(y_true=label.detach().cpu().numpy(),
                                                    y_score=best_proba[:,1].detach().cpu().numpy()
                                                    )
        return test_f1_macro, test_f1_micro, auc

from  collections import Iterable 
def flatten(items,ignore_types=(str,bytes)):
    for x in items:
        if isinstance(x,Iterable) and not isinstance(x,ignore_types):
            yield from flatten(x)
        else:
            yield x


from sklearn.metrics import precision_recall_curve,roc_curve,auc,balanced_accuracy_score

#acc,prec,re,f1=prediction(predlabel,labels)
def prediction(predlabel,labels):
    predlabel_s=[]
    labels_s=[]
    for x in flatten(predlabel):
        predlabel_s.append(x)
    for x in flatten(labels):
        labels_s.append(x)

    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
    acc=accuracy_score(labels_s, predlabel_s)
    precision=precision_score(labels_s, predlabel_s)
    recall=recall_score(labels_s, predlabel_s)
    f1=f1_score(labels_s, predlabel_s)
    return acc,precision,recall,f1
def loss_acc( output, y):

  
    criterion = torch.nn.BCELoss(reduction='mean')
    loss = criterion(output, y)
    
    #loss = F.nll_loss(output, y)
    predlabel=output.data.numpy()>0.5
    predlabel=predlabel.astype(np.int32)
    labels=y.int().numpy()
    acc,prec,re,f1 = prediction(predlabel, labels)
    return loss, acc

def train_mi(x, views):
    vv1, vv2, v1v2 = main_model.get_mi_loss(x, views)
    return mi_coe * v1v2 + (vv1 + vv2) * (1 - mi_coe) / 2

def train_cls(data,a):
    new_v1, new_v2 = main_model.get_view(data)
    emb1,emb2 = main_model.get_cls_loss(new_v1, new_v2, data.x)
    
    emb1=pd.DataFrame(emb1.detach().numpy())
    emb2=pd.DataFrame(emb2.detach().numpy())
    emb1.insert(0,'m',[i for i in range(len(emb1))])
    emb2.insert(0,'m',[i for i in range(len(emb2))])
    a.columns=['pi','dis','label']

    e1=pd.merge(a,emb1,left_on='pi',right_on='m',how='left')
    e1_=pd.merge(e1,emb1,left_on='dis',right_on='m',how='left')

    e1_.drop(['pi','dis','label','m_x','m_y'],axis=1,inplace=True)

    e2=pd.merge(a,emb2,left_on='pi',right_on='m',how='left')
    e2_=pd.merge(e2,emb2,left_on='dis',right_on='m',how='left')
    e2_.drop(['pi','dis','label','m_x','m_y'],axis=1,inplace=True)

    prob_v1=mlp(torch.tensor(e1_.values))
    prob_v2=mlp(torch.tensor(e2_.values))
    # print(prob_v1.shape)

    curr_v = main_model.get_fusion(torch.tensor(e1_.values).to_sparse(), prob_v1, torch.tensor(e1_.values).to_sparse(), prob_v2)
    #emb = main_model.get_v_cls_loss(curr_v, data.x)
    prob=mlp(curr_v.to_dense())
    
    views = [curr_v, e1_, e2_]
    
    logits_v1=prob_v1[:,1]
    logits_v2=prob_v2[:,1]
    logits_v=prob[:,1]
    label=torch.LongTensor(a[['label']].values)
    label=label.float()
    label=torch.squeeze(label)
    # print('logits_v1',logits_v1[data.idx_train])
    # print('label',label[data.idx_train])
    loss_v1,_ = loss_acc(logits_v1[data.idx_train], label[data.idx_train])
    loss_v2,_ = loss_acc(logits_v2[data.idx_train], label[data.idx_train])
    loss_v,_= loss_acc(logits_v[data.idx_train], label[data.idx_train])
    return cls_coe * loss_v + (loss_v1 + loss_v2) * (1 - cls_coe) / 2, views,label,logits_v
def evaluate(prob,label):

    def auroc(prob,label):
        y_true=label.data.numpy().flatten()
        y_scores=prob.data.numpy().flatten()
        fpr,tpr,thresholds=roc_curve(y_true,y_scores)
        auroc_score=auc(fpr,tpr)
        return auroc_score,fpr,tpr

    def auprc(prob,label):
        y_true=label.data.numpy().flatten()
        y_scores=prob.data.numpy().flatten()
        precision,recall,thresholds=precision_recall_curve(y_true,y_scores)
        auprc_score=auc(recall,precision)
        return auprc_score,precision,recall

    
    auroc_score,fpr,tpr=auroc(prob,label)
    auprc_score,precision,recall=auprc(prob,label)

    predlabel=prob.data.numpy()>0.5
    predlabel=predlabel.astype(np.int32)
    labels=label.int().numpy()
    acc,prec,re,f1 = prediction(predlabel, labels)
    return auroc_score,auprc_score, acc,prec,re,f1 ,fpr,tpr,precision,recall

def test(data,a):
    new_v1, new_v2 = main_model.get_view(data)
    emb1,emb2 = main_model.get_cls_loss(new_v1, new_v2, data.x)
    
    emb1=pd.DataFrame(emb1.detach().numpy())
    emb2=pd.DataFrame(emb2.detach().numpy())
    emb1.insert(0,'m',[i for i in range(len(emb1))])
    emb2.insert(0,'m',[i for i in range(len(emb2))])
    a.columns=['pi','dis','label']
    e1=pd.merge(a,emb1,left_on='pi',right_on='m',how='left')
    e1_=pd.merge(e1,emb1,left_on='dis',right_on='m',how='left')
    
    e1_.drop(['pi','dis','label','m_x','m_y'],axis=1,inplace=True)

    e2=pd.merge(a,emb2,left_on='pi',right_on='m',how='left')
    e2_=pd.merge(e2,emb2,left_on='dis',right_on='m',how='left')
    e2_.drop(['pi','dis','label','m_x','m_y'],axis=1,inplace=True)
    
    prob_v1=mlp(torch.tensor(e1_.values))
    prob_v2=mlp(torch.tensor(e2_.values))
  

    curr_v = main_model.get_fusion(torch.tensor(e1_.values).to_sparse(), prob_v1, torch.tensor(e1_.values).to_sparse(), prob_v2)
    #emb = main_model.get_v_cls_loss(curr_v, data.x)
    prob=mlp(curr_v.to_dense())
    
    views = [curr_v, e1_, e2_]
    
    logits_v1=prob_v1[:,1]
    logits_v2=prob_v2[:,1]
    logits_v=prob[:,1]
    label=torch.LongTensor(a[['label']].values)
    label=label.float()
    label=torch.squeeze(label)
    auroc_score,auprc_score, acc,prec,re,f1,fpr,tpr,precision,recall  = evaluate(logits_v[data.idx_test], label[data.idx_test])
    return auroc_score,auprc_score, acc,prec,re,f1 ,fpr,tpr,precision,recall



warnings.filterwarnings('ignore')
torch.autograd.set_detect_anomaly(True)
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
cls_hid_1=256
gen_hid=32
mi_hid_1=128
com_lambda_v1=0.5
com_lambda_v2=0.5
lam=0.5
alpha=0.5
cls_dropout=0.5
ve_dropout=0.5
tau=0.2
pyg=False
big=False
batch=0
dataset='pi'
ve_lr=0.001
ve_weight_decay=0.0
cls_lr=0.01
cls_weight_decay=5e-4
mi_lr=0.01
mi_weight_decay=0.0
main_epoch=500
temp_r=1e-3
inner_ne_epoch=1
inner_cls_epoch=1
inner_mi_epoch=5
cls_coe=0.3
mi_coe=0.3



recall_s=[]
precision_s=[]
fprs=[]
tprs=[]
test_aucs=[]
test_auprs=[]
accs=[]
precs=[]
res=[]
f1s=[]


data_path = "../dataset/mi_drug_LRGCPND"
all_,a,feature,label,ori_view1,ori_view2,ori_view1_indice,ori_view2_indice=load_data(data_path)

from sklearn.model_selection import StratifiedKFold,KFold
kf = KFold(n_splits=5,random_state=1234,shuffle=True)

#kf = StratifiedKFold(n_splits=3,random_state=0,shuffle=True)
for train_index, test_index in kf.split(a):

    print('------------------------------------------------------------')
    val_ind=random.sample(set(train_index),math.ceil(len(train_index)*0.2))
    train_ind=list(set(train_index)-set(val_ind))
    test_ind=test_index
    idx_train=np.array(train_ind)
    idx_val=np.array(val_ind)
    idx_test=np.array(test_ind)

    data=DataSet(dataset='pi', x=feature, y=label, view1=ori_view1, view2=ori_view2,
                   view1_indice=ori_view1_indice, view2_indice=ori_view2_indice,
                   idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)

    main_model = Cogsl(data.num_feature, cls_hid_1, 128,
                                        gen_hid, mi_hid_1, com_lambda_v1, com_lambda_v2,
                            lam, alpha, cls_dropout,ve_dropout,tau,pyg,big, batch, dataset)

    mlp=Link_Prediction(128, hidden_dims=[128, 2], dropout=0.3)

    opti_ve = torch.optim.Adam(main_model.ve.parameters(), lr=ve_lr, weight_decay=ve_weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opti_ve, 0.99)

    opti_cls = torch.optim.Adam(main_model.cls.parameters(), lr=cls_lr, weight_decay=cls_weight_decay)
    opti_mi = torch.optim.Adam(main_model.mi.parameters(), lr=mi_lr, weight_decay=mi_weight_decay)
    opti_mlp = torch.optim.Adam(mlp.parameters(), lr=0.001)
    best_acc_val = 0
    best_loss_val = 1e9
    best_test = 0
    best_v = None
    best_v_cls_weight = None


    for epoch in range(main_epoch):
        curr = np.log(1 + temp_r * epoch)
        curr = min(max(0.05, curr), 0.1)
        
        #for inner_ne in range(inner_ne_epoch):
        main_model.train()
        mlp.train()
        opti_ve.zero_grad()
        opti_cls.zero_grad()
        opti_mi.zero_grad()
        opti_mlp.zero_grad()
        # train_cls()
        cls_loss, views,label,prob = train_cls(data,a)
        cls_loss.backward()
        scheduler.step()
        opti_ve.step()
        opti_cls.step()
        opti_mi.step()
        opti_mlp.step()
        scheduler.step()

    
        main_model.eval()
        mlp.eval()
        
        cls_loss, views,label,prob = train_cls(data,a)
        # logits_v_val = self.main_model.get_v_cls_loss(views[0], self.data.x)
  
        loss_val, acc_val = loss_acc(prob[data.idx_val], label[data.idx_val])
        if acc_val >= best_acc_val and best_loss_val > loss_val:
            best_acc_val = max(acc_val, best_acc_val)
            best_loss_val = loss_val
            best_v_cls_weight = deepcopy(main_model.state_dict())
            best_mlp = deepcopy(mlp.state_dict())
            best_v = views[0]
        print("EPOCH ",epoch, "\tCUR_LOSS_VAL ", loss_val, "\tCUR_ACC_Val ", acc_val, "\tBEST_ACC_VAL ", best_acc_val)
    with torch.no_grad():
        main_model.load_state_dict(best_v_cls_weight)
        mlp.load_state_dict(best_mlp)
        main_model.eval()
        mlp.eval()
        auroc_score,auprc_score, acc,prec,re,f1,fpr,tpr,precision,recall =test(data,a)                
    
        print("auc:{},aupr:{},re:{},acc:{},prec:{},f1:{}".format(auroc_score,auprc_score, acc,prec,re,f1))
        test_aucs.append(auroc_score)
        test_auprs.append(auprc_score)
        recall_s.append(recall)
        precision_s.append(precision)
        accs.append(acc)
        precs.append(prec)
        res.append(re)
        f1s.append(f1)
        fprs.append(fpr)
        tprs.append(tpr)

pd.DataFrame(test_aucs).to_csv("../result/test_auc_scores.csv",index=0)
pd.DataFrame(test_auprs).to_csv("../result/test_auprs.csv",index=0)
pd.DataFrame(accs).to_csv("../result/accs.csv", index=0)
pd.DataFrame(precs).to_csv("../result/precs.csv", index=0)
pd.DataFrame(res).to_csv("../result/res.csv", index=0)
pd.DataFrame(f1s).to_csv("../result/f1s.csv", index=0)

pd.DataFrame(precision_s).to_csv("../result/precision_s.csv",index=0)
pd.DataFrame(recall_s).to_csv("../result/recall_s.csv",index=0)
pd.DataFrame(fprs).to_csv("../result/fprs.csv",index=0)  
pd.DataFrame(tprs).to_csv("../result/tprs.csv",index=0)  

    
