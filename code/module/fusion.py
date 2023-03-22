import torch
import torch.nn as nn
import torch.nn.functional as F
class fuse(nn.Module):


    def __init__(self, channels,inter_channels):
        super(fuse, self).__init__()
        #inter_channels = channels
       
        self.local_att = nn.Sequential(
           
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(inter_channels),
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
        )
 
        self.sigmoid = nn.Sigmoid()
        self.linear=nn.Linear(inter_channels,256)
        self.linear2=nn.Linear(64,16)
    def forward(self, residual,residual2):
        # print('residual',residual)
        # print('residual2',residual2)
        residual = residual.to_dense()
        residual2 = residual2.to_dense()

        residual=torch.unsqueeze(residual,dim=0)
        residual2=torch.unsqueeze(residual2,dim=0)

        residual=residual.permute(0, 2, 1)
        residual2=residual2.permute(0, 2, 1)
        xa = residual+residual2
 
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        

        xl=torch.squeeze(xl)
        xl=self.linear(xl)
        xg=torch.squeeze(xg)
        xg=self.linear(xg)        
        xlg = xl + xg
        #xlg=torch.squeeze(xlg)
        #xlg=self.linear(xlg)


        wei = self.sigmoid(xlg)

       
        residual=residual.permute(0, 2, 1)
        residual2=residual2.permute(0, 2, 1)
        residual=torch.squeeze(residual)
        residual2=torch.squeeze(residual2)

        xo =   residual * wei + residual2 * (1 - wei)
        #xo = self.linear2(xo)
        # print(xo)
        return xo.to_sparse()

class Fusion(nn.Module):
    def __init__(self, lam, alpha, name):
        super(Fusion, self).__init__()
        self.lam = lam
        self.alpha = alpha
        self.name = name

    def get_weight(self, prob):
        out, _ = prob.topk(2, dim=1, largest=True, sorted=True)
        fir = out[:, 0]
        sec = out[:, 1]
        w = torch.exp(self.alpha*(self.lam*torch.log(fir+1e-8) + (1-self.lam)*torch.log(fir-sec+1e-8)))
        return w

    def forward(self, v1, prob_v1, v2, prob_v2):
        print('v1',v1)
        print('v2',v2)
        print('v1',v1.shape)
        print('v2',v2.shape)        
        w_v1 = self.get_weight(prob_v1)
        w_v2 = self.get_weight(prob_v2)
        beta_v1 = w_v1 / (w_v1 + w_v2)
        beta_v2 = w_v2 / (w_v1 + w_v2)
        if self.name not in ["citeseer", "digits", "polblogs"]:
            beta_v1 = beta_v1.diag().to_sparse()
            beta_v2 = beta_v2.diag().to_sparse()
            v = torch.sparse.mm(beta_v1, v1) + torch.sparse.mm(beta_v2, v2)
            return v
        else :
            beta_v1 = beta_v1.unsqueeze(1)
            beta_v2 = beta_v2.unsqueeze(1)
           # v = beta_v1 * v1.to_dense() + beta_v2 * v2.to_dense()
            v = beta_v1 * v1.to_dense() + beta_v2 * v2.to_dense()
            return v.to_sparse()
