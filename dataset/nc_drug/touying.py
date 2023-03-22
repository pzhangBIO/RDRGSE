'''
根据位数筛选，alldegree/max(degree)-min(degree)
随后将挑选得到的数据，根据度值计算概率值，相加
'''
import pandas as pd
import numpy as np
import math
import torch

import torch.nn as nn
def sss1(data0):
    s = np.zeros([len(data0),len(data0)])
    s += np.array([data0])
    s += np.array([data0]).T
    s[s<2]=0
    s[s==2]=1
    return s


data=pd.read_csv('AssociationMatrix.csv',header=None).T

dis_degree=data.sum(axis=0)
all_matrix=[]
for i in range(data.shape[1]):
    a=sss1(data.iloc[:,i])
    all_matrix.append(a)

degree=pd.DataFrame({'dis':dis_degree})
degree.insert(0,'num',[i for i in range(0,data.shape[1])])
de=degree.sort_values(by='dis',ascending=False)
final=np.zeros((data.shape[0],data.shape[0]))
medi=pd.Series(list(set(de['dis']))).median()
##根据位数筛选，alldegree/max(degree)-min(degree)
n=math.ceil(data.sum().sum()/de['dis'].max()-de['dis'].min())

w=math.ceil(medi)
deshape=[]
while w>0:
    
    de_shape=de.loc[de['dis']==w].shape[0]
    for j in range(de_shape):
        d=de.loc[de['dis']==w].loc[:,'num'].tolist()[j]
        if len(deshape)<n:
            deshape.append(d)
        else:
            break
    w=w-1

print(len(deshape)==n)


a=torch.from_numpy(de.loc[deshape][['dis']].values).float()
a=a/float(a.mean())

m = nn.Softmax(dim=0) 
b=np.squeeze(np.array(m(a))).tolist()
c=de.loc[deshape]
c.insert(2,'prob',b)

for i in deshape: 
    p=c.loc[c['num']==i,'prob']
    p=float(p)
    final+=p*all_matrix[i]


final=torch.from_numpy(final)

pd.DataFrame(torch.bernoulli(final).data.numpy()).to_csv('touying_drug.csv',index=0,header=0)