import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math
from utils import dgc_precompute
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import MessagePassing, APPNP
import numpy as np
from BernNet import BernNet

class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat, nclass, group=1):
        super(SGC, self).__init__()
        self.group = group
        self.temp = nn.Parameter(torch.FloatTensor(1.0*np.ones(group)))
        self.nfeat = nfeat // self.group
        # self.fuse = nn.Linear(self.nfeat * self.group, self.nfeat)
        self.W = nn.Linear(self.nfeat, nclass)


    def forward(self, x):
        if self.group > 1:
            # self.temp[0].data=torch.clamp(self.temp[0].data,1)
            feat_num = x.size(1)//self.group
            feature = self.temp[0] * x[:,:feat_num] + self.temp[1] * x[:,feat_num:]
            # temp = F.relu(self.temp).reshape(self.group, 1)
            # feature = x.reshape(x.size(0), self.group, self.nfeat).permute(0,2,1)
            # feature = (feature @ temp).reshape(x.size(0),self.nfeat)
        else:
            feature = torch.clamp(self.temp,0) * x

        return self.W(feature)

class MLP2(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat, nhid, nclass, group=1, dp=0.2):
        super(MLP2, self).__init__()
        self.group = group
        self.temp = nn.Parameter(torch.FloatTensor(1.0*np.ones(group)))
        self.nfeat = nfeat // self.group
        # self.fuse = nn.Linear(self.nfeat * self.group, self.nfeat)
        self.W1 = nn.Linear(self.nfeat, nhid)
        self.W2 = nn.Linear(nhid, nclass)
        self.dp = dp
        self.act = nn.PReLU()
        self.num_class = nclass


    def forward(self, x):
        temp = F.relu(self.temp).reshape(self.group, 1)
        feature = x.reshape(x.size(0), self.group, self.nfeat).permute(0,2,1)
        feature = (feature @ temp).reshape(x.size(0),self.nfeat)
        x = self.act(self.W1(feature))
        x = nn.Dropout(p=self.dp)(x)
        return self.W2(x)


class MLP(Module):
    """
    A Simple two layers MLP to make SGC a bit better.
    """
    def __init__(self, nfeat, nhid, nclass, dp=0.2):
        super(MLP, self).__init__()
        self.W1 = nn.Linear(nfeat, nhid)
        self.W2 = nn.Linear(nhid, nclass)
        self.dp = dp
        self.act = nn.PReLU()
        self.num_class = nclass

    def forward(self, x):
        x = self.act(self.W1(x))
        x = nn.Dropout(p=self.dp)(x)
        return self.W2(x)



class DGCT(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat, nclass, T=2.0, K=100):
        super(DGCT, self).__init__()

        self.W = nn.Linear(nfeat, nclass)
        self.T = nn.Parameter(torch.tensor(T))
        self.K = K

    def forward(self, x, adj):
        x = dgc_precompute(x, adj, self.K, self.T)
        return self.W(x)


class GraphConvolution(Module):
    """
    A Graph Convolution Layer (GCN)
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Linear(in_features, out_features, bias=bias)
        self.init()

    def init(self):
        stdv = 1. / math.sqrt(self.W.weight.size(1))
        self.W.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = self.W(input)
        output = torch.spmm(adj, support)
        return output

class GCN(nn.Module):
    """
    A Two-layer GCN.
    """
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, use_relu=True):
        x = self.gc1(x, adj)
        if use_relu:
            x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x




def get_model(model_opt, nfeat, nclass, nhid=0, dropout=0, cuda=True, T=2.0, K=10,dprate=0.0,group=1):
    if model_opt == "GCN":
        model = GCN(nfeat=nfeat,
                    nhid=nhid,
                    nclass=nclass,
                    dropout=dropout)
    elif model_opt == "SGC":
        model = SGC(nfeat=nfeat,
                    nclass=nclass,
                    group=group)
    elif model_opt == "MLP":
        model = MLP(nfeat=nfeat,
                    nhid=nhid,
                    nclass=nclass,
                    dp = dropout)
    elif model_opt == "MLP2":
        model = MLP2(nfeat=nfeat,
                    nhid=nhid,
                    nclass=nclass,
                    dp = dropout,
                    group = group)
    elif model_opt == "DGCT":
        model = DGCT(nfeat=nfeat,
                    nclass=nclass,
                    T=T,
                    K=K)
    elif model_opt == "BernNet":
        model = BernNet(nfeat=nfeat,
                        n_hidden=nhid,
                        nclass=nclass,
                        dropout=dropout,
                        dprate=dprate,
                        K=K)
    else:
        raise NotImplementedError('model:{} is not implemented!'.format(model_opt))

    if cuda: model.cuda()
    return model
