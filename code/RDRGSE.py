import torch
import torch.nn as nn
import torch.nn.functional as F
from module.view_estimator import View_Estimator
from module.cls import Classification
from module.mi_nce import MI_NCE
from module.fusion import Fusion,fuse


class RDRGSE(nn.Module):
    def __init__(self, num_feature, cls_hid_1, num_class, gen_hid, mi_hid_1,
                 com_lambda_v1, com_lambda_v2, lam, alpha, cls_dropout, ve_dropout, tau, pyg, big, batch, name):
        super(RDRGSE, self).__init__()
        self.cls = Classification(num_feature, cls_hid_1, num_class, cls_dropout, pyg)
        self.ve = View_Estimator(num_feature, gen_hid, com_lambda_v1, com_lambda_v2, ve_dropout, pyg)
        self.mi = MI_NCE(num_feature, mi_hid_1, tau, pyg, big, batch)
        self.fusion = Fusion(lam, alpha, name)
        
    def get_view(self, data):
        new_v1, new_v2 = self.ve(data)
        return new_v1, new_v2

    def get_mi_loss(self, feat, views):
        mi_loss = self.mi(views, feat)
        return mi_loss

    def get_cls_loss(self, v1, v2, feat):
        emb1 = self.cls(feat, v1, "v1")
        emb2 = self.cls(feat, v2, "v2")
        return emb1,emb2

    def get_v_cls_loss(self, v, feat):
        emb=self.cls(feat, v, "v")
        return emb

    def get_fusion(self, v1, prob_v1, v2, prob_v2):
        feature_fuse = fuse(v1.shape[1],v1.shape[0])

        v = feature_fuse(v1, v2)
        return v
