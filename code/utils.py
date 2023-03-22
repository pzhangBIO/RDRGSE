import torch
import numpy as np
import scipy.sparse as sp
from torch_sparse import SparseTensor


def accuracy(output, label):
    """ Return accuracy of output compared to label.
    Parameters
    ----------
    output:
        output from model (torch.Tensor)
    label:
        node label (torch.Tensor)
    """
    #print(output.max(1)[1])
    preds = output.max(1)[1].type_as(label)
    correct = preds.eq(label).double()
    correct = correct.sum()
    return correct / len(label)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def sparse_mx_to_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    rows = torch.from_numpy(sparse_mx.row).long()
    cols = torch.from_numpy(sparse_mx.col).long()
    values = torch.from_numpy(sparse_mx.data)
    return SparseTensor(row=rows, col=cols, value=values, sparse_sizes=torch.tensor(sparse_mx.shape))

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def convert_num():
    pi=pd.read_csv('ncRNA.csv',header=None)
    dis=pd.read_csv('drug.csv',header=None)
    data=pd.read_csv('nc_drug.csv',header=None)
    all_node=pd.concat([pi,dis],axis=0,ignore_index=True)
    all_node[1]=[i for i in range(len(all_node))]
    d=dict(zip(all_node[0],all_node[1]))
    data=data.replace(d)
    positive=list(zip(data[0],data[1]))
    return positive,d
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
    print("Loading {} dataset...".format(dataset))
   
    feature = sp.load_npz(data_path+"/feat.npz")
  
    feature = feature.todense()
    

    feature = torch.FloatTensor(np.array(feature))

    positive,d=convert_num()
    pi=[]
    dis=[]
    for i,j in d.items():
        if i.startswith('pi'):
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

    train_ind=random.sample(all_,math.ceil(len(sample)*0.6))
    val_ind=random.sample(all_,math.ceil(len(sample)*0.2))
    test_ind=list(all_-set(train_ind))
    idx_train=np.array(train_ind)
    idx_val=np.array(val_ind)
    idx_test=np.array(test_ind)
    label = np.array(label)
    label = torch.LongTensor(label)

    ori_view1 = sp.load_npz(data_path+"/v1_knn"+".npz")       
    ori_view2 = sp.load_npz(data_path+"/v2_diff"+".npz")
    ori_view1_indice = torch.load(data_path+"/v1_2"+".pt")
    ori_view2_indice = torch.load(data_path+"/v2_40"+".pt")
    
    ori_view1 = sparse_mx_to_torch_sparse_tensor(normalize_adj(ori_view1)) 
    ori_view2 = sparse_mx_to_torch_sparse_tensor(normalize_adj(ori_view2))
    ####
    return sample,DataSet(dataset=args.dataset, x=feature, y=label, view1=ori_view1, view2=ori_view2,
                   view1_indice=ori_view1_indice, view2_indice=ori_view2_indice,
                   idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)



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
