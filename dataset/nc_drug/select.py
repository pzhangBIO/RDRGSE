
import pandas as pd
import numpy as np
import torch
import scipy.sparse as sp 
import networkx as nx
import random
import copy
from scipy.spatial.distance import pdist,squareform
from torch import nn
from torch.nn import init
import math

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import metrics


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
def load_index(allnode):
    ind2ind={i:j for i,j in zip(allnode['id'],range(len(allnode)))}
    return  ind2ind


def generate_edge():
    #generate 0/1 matrix by bernoulli
    single_pi=pd.read_csv('../data/circRNA.csv',names=['id'])
    single_dis=pd.read_csv('../data/dis.csv',names=['id'])
    pi_sim=pd.read_csv('../data/touying_circ.csv', header=None).values
    dis_sim=pd.read_csv('../data/touying_dis.csv', header=None).values

    
    dis_sim = torch.from_numpy(dis_sim)
    berno = torch.bernoulli(dis_sim)
    df_dis=pd.DataFrame(berno.numpy().astype(int))

    allnode=pd.concat([single_pi,single_dis])
    ind2ind=load_index(allnode)
    re_dis=single_dis['id'].replace(ind2ind).tolist()
    df_dis.index=re_dis
    df_dis.columns=re_dis
    sp_dis=sp.coo_matrix(df_dis).nonzero()
    edge=list(zip(sp_dis[0],sp_dis[1]))
    edge_dis=pd.DataFrame(edge).replace({i:j for i,j in enumerate(df_dis)})

    pi_sim=torch.from_numpy(pi_sim)
    berno_pi=torch.bernoulli(pi_sim)
    df_pi=pd.DataFrame(berno_pi.numpy().astype(int))
    re_pi=single_pi['id'].replace(ind2ind).tolist()
    df_pi.index=re_pi
    df_pi.columns=re_pi
    sp_pi=sp.coo_matrix(df_pi).nonzero()
    edge_pi=list(zip(sp_pi[0],sp_pi[1]))
    edge_pi=pd.DataFrame(edge_pi).replace({i:j for i,j in enumerate(df_pi)})
    return edge_pi,edge_dis,ind2ind,single_pi,single_dis

def generate_ed(pi_dis_train,adj_train):

    #print('pi_dis_train',pi_dis_train)


    edge_pi,edge_dis,ind2ind,single_pi,single_dis=generate_edge()
    #print(ind2ind)
    pi_dis_edge=pd.concat([edge_pi,edge_dis])
    
    adj_edge=pi_dis_train.replace(ind2ind)
    adj_edge.drop('index',axis=1,inplace=True)
    adj_edge.columns=[0,1]
    adj_raw=adj_edge.copy()
    #print(adj_edge)
    alledge=pd.concat([edge_pi,edge_dis,adj_edge])
    # print('alledge',alledge)

    ## construct graph,add edge
    nxedge=list(zip(alledge[0],alledge[1]))
    #print(nxedge)
    G=nx.Graph()
    G.add_edges_from(nxedge)

    for i in ind2ind.values():
        if int(i) not in set(alledge[0]) :
            G.add_node(i)
 
    degree_cert=nx.degree_centrality(G)

    ##construct all edge matrix
    s=adj_train.toarray().shape[0] 
    data=pd.DataFrame(np.zeros((s,s)))
    ind=[]
    for i in data.index.tolist():
        for j in data.index.tolist():
            ind.append((i,j))

    all_edge=pd.DataFrame(ind)
    all_edge.columns=['node1','node2']
    ##construct degree_cert matrix
    #print(degree_cert)
    d=pd.DataFrame.from_dict(degree_cert,orient='index')
    d=d.reset_index().rename({'index':'node'})
    d.columns=['node1','degree_cer']
  
    ### combine  all edge matrix with degree_cert matrix
    d1=pd.merge(all_edge,d,on='node1',how='left')
   
    d2=pd.merge(d1,d,left_on='node2',right_on='node1',how='left')
 
    fina_edge=d2.drop('node1_y',axis=1)
    fina_edge['edge_degree']=(fina_edge['degree_cer_x']+fina_edge['degree_cer_y'])/2
    ##这里的final_edge_matrix 为重构出来的矩阵四块都有值。
    final_edge_matrix=pd.DataFrame(np.array(fina_edge['edge_degree'].tolist()).reshape(s,-1))

    ##different generate 0/1 matrix

    con_gra=pd.DataFrame(torch.bernoulli(torch.from_numpy(final_edge_matrix.values)).numpy().astype(int))

    con_gra_=con_gra.values

    ###聚多少个类
    if 1< math.ceil(con_gra.sum().sum()/len(con_gra_)) <=10:
        n=1
    elif 10< math.ceil(con_gra.sum().sum()/len(con_gra_)) <=100:
        n=100
    elif 100< math.ceil(con_gra.sum().sum()/len(con_gra_)) <=1000:
        n=100
    
    try:
        n_cluster=len(set([round(i) for i in (con_gra.sum()/n).tolist()]))
    except UnboundLocalError:
        n_cluster=len(set([round(i) for i in (con_gra.sum()/100).tolist()]))

    print('聚类K:',n_cluster)
    ###开始聚类
    Kmeans=KMeans(n_clusters=n_cluster)
    y=Kmeans.fit_predict(final_edge_matrix)
    final_edge_matrix.insert(0,'label',y)
    center=pd.DataFrame(Kmeans.cluster_centers_)
    center.insert(0,'label',range(0,n_cluster))
    #datafram 存放所有的小子图，计算每一类与中心的距离
    datafram=[]
    for i in set(y):
        matrix=final_edge_matrix.loc[final_edge_matrix['label']==i,:]
        center_value=center.loc[center['label']==i,:]
        matrix.drop('label',axis=1,inplace=True)
        center_value.drop('label',axis=1,inplace=True)
        center_v=center_value.iloc[0,:].tolist()
        val=[]
        for index,row in matrix.iterrows():
            val.append(np.linalg.norm(np.array(row) - np.array(center_v)))
        matrix.insert(0,'dist',val)
        matrix.insert(0,'label',i)
        datafram.append(matrix)

    all_subgraph=pd.concat(datafram)
    ####选择距离小于均值的
    select_data=[]
    for j in set(y):
        sig_mat=all_subgraph.loc[all_subgraph['label']==j,:]
        sig_mat_sort=sig_mat.sort_values(by='dist')
        sig_mat_sort=sig_mat_sort.loc[sig_mat_sort['dist']<np.mean(sig_mat_sort['dist']),:]
        select_data.append(sig_mat_sort)
    data_select=pd.concat(select_data)
    ##取交集
    common_edge=set()

    for m in select_data:
        for inds,n in pi_dis_edge.iterrows():
            if n[0] in m.index.tolist() or n[0] in m.index.tolist():
                common_edge.add(inds)    
    pi_dis_edge=pi_dis_edge.loc[list(common_edge)]

    #print('s',s)

    pi_num=[int(i) for i in range(0,len(single_pi))]
    dis_num=[int(i) for i in range(single_pi.shape[0],single_pi.shape[0]+single_dis.shape[0])]
    all_num=[]
    all_num.extend(pi_num)
    all_num.extend(dis_num)
    all_node=[]
    all_node.extend(single_pi['id'].tolist())
    all_node.extend(single_dis['id'].tolist())
    node_dict=dict(zip(pd.DataFrame(all_num)[0],pd.DataFrame(all_node)[0]))


    adj_edge=adj_edge.replace(node_dict)
    adj_edge=adj_edge.replace(ind2ind)
    final_list=list(zip(adj_edge[0],adj_edge[1]))###准备生成一个adj
    final_list2=list(zip(pi_dis_edge[0],pi_dis_edge[1]))


    adj=pd.DataFrame(np.zeros((len(set(adj_raw[0])),len(set(adj_raw[1])))))
    a=list(set(adj_raw[1]))
    a.sort(key = adj_raw[1].tolist().index)
    adj.columns=a
    b=list(set(adj_raw[0]))
    b.sort(key = adj_raw[0].tolist().index)   
    adj.index=b
    #print('adj',adj)

    for i in final_list:
        if i[0] in b and i[1] in a:
            adj.loc[i]=1.0
    ##填补adj(1478,24)

    pi_sim=pd.DataFrame(np.zeros((len(set(adj_raw[0])),len(set(adj_raw[0])))))
    pi_a=list(set(adj_raw[0]))
    pi_a.sort(key = adj_raw[0].tolist().index)
    pi_sim.columns=pi_a
    pi_sim.index=pi_a
    for j in final_list2:
        if j[0] in pi_a and j[1] in pi_a:
            pi_sim.loc[j]=1.0    

    dis_sim=pd.DataFrame(np.zeros((len(set(adj_raw[1])),len(set(adj_raw[1])))))
    dis_a=list(set(adj_raw[1]))
    dis_a.sort(key = adj_raw[1].tolist().index)

    dis_sim.columns=dis_a
    dis_sim.index=dis_a
    #print('dis_sim',dis_sim)  
    for j in final_list2:
        if j[0] in dis_a and j[1] in dis_a:
            dis_sim.loc[j]=1.0   


    def construct_adj(adj,pi,dis):
        pi_mat=pi.values
        dis_mat=dis.values
        
        mat1=np.hstack((pi_mat,adj))
        mat2=np.hstack((adj.T,dis_mat))
        mat=np.vstack((mat1,mat2))
        return mat

    A=construct_adj(adj.values,pi_sim,dis_sim)

   
    recon_adj_train = sp.csr_matrix(A)
   
    print('view1 edge',recon_adj_train.A.sum())
    
    adj=recon_adj_train
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj) 
    return adj

