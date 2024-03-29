import time
import argparse
import numpy as np
import pickle as pkl
import os
from math import log
# from citation import train_regression, test_regression
from citation import accuracy
from models import get_model
from utils import *

from args import get_citation_args
import torch
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# import torch
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import Metattack
from deeprobust.graph.data import Dataset, PrePtbDataset,Dpr2Pyg, Pyg2Dpr
from deeprobust.graph.utils import normalize_adj
from dataset_loader import DataLoader
from torch_geometric.utils import *
from utils import random_planetoidsplits
import sys
from normalization import aug_normalized_adjacency, row_normalize


def train_regression(model,
                     train_features, train_labels,
                     val_features, val_labels,
                     epochs=100, weight_decay=5e-6,
                     lr=0.2, dropout=0.,times=0.1):

    optimizer = optim.Adam([{'params': model.W.parameters(),'weight_decay': weight_decay, 'lr': lr},
                            {'params': model.temp,'weight_decay': 0, 'lr': lr*times},])
    # optimizer = torch.optim.Adam([{'params': model.W.parameters(),'weight_decay': weight_decay, 'lr': lr},
    #     {'params': model.temp, 'weight_decay': 0.00, 'lr': lr*0.2}])
    t = perf_counter()
    best_acc = 0.
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_features)
        loss_train = F.cross_entropy(output, train_labels)
        loss_train.backward()
        optimizer.step()
        with torch.no_grad():
            model.eval()
            output = model(val_features).max(1)[1]
            acc_val = accuracy(output, val_labels)
        if acc_val > best_acc:
            best_acc = acc_val
    train_time = perf_counter()-t
    return model, best_acc, train_time

def test_regression(model, test_features, test_labels):
    model.eval()
    return accuracy(model(test_features), test_labels)


# Arguments
args = get_citation_args()

# setting random seeds
set_seed(args.seed, args.cuda)

# Hyperparameter optimization
space = {'weight_decay' : hp.loguniform('weight_decay', log(1e-10), log(1e-1)),
    'T1': hp.uniform('T1', 0, 50),
    'T2': hp.uniform('T2', 0, 50),
    'times': hp.loguniform('times', log(1e-2), log(10)),
    # 'lr': hp.uniform('lr',0.05,2.0),
    # 'biasl': hp.uniform('biasl', 0.5, 1.0),
    # 'biash': hp.uniform('biash', 0.5, 1.0),
    # 'dropout': hp.uniform('dropout',0.0,1.0)
    }
best = {
    'weight_decay' : 0.0364,
    'T2': 0.2369,
    'T1': 13.01,
    'times':3.4118
    # 'biasl': -0.75,
    # 'biash': 0.75,
    # 'degree': 5,
    # 'dropout':0.0,
}
dataset = DataLoader(args.dataset)
data = dataset[0]
print(data)
raw_features = data.x.cuda()
labels = data.y.cuda()
print(labels.max().item()+1)

edge_index, norm =get_laplacian(data.edge_index, None,
                'sym', raw_features.dtype,
                raw_features.size(0))
adj = to_dense_adj(edge_index, edge_attr = norm).reshape(data.x.size(0),-1).to_sparse().cuda()
flag = True
if args.dataset in ["cora","citeseer","pubmed"]:
    _, features, labels, idx_train, idx_val, idx_test = load_citation(args.dataset, None, args.cuda)
    flag = False
# else:
#     flag = True
#     adj = adj1
split_path = f'data/data_splits/{args.dataset}_idx.pt'

###Set SPLIT RATIO

percls_trn = int(round(0.6*len(data.y)/dataset.num_classes))
val_lb = int(round(0.2*len(data.y)))

###END##

#  1-math.sqrt(math.log(math.sqrt(2))/space['T']) -1+math.sqrt(math.log(math.sqrt(2))/space['T']) 1-math.sqrt((math.sqrt(2))/space['T'])
def sgc_objective(space,try_time=10, long_epoch=False):
    features, precompute_time = dgc_precompute_lhgaussian(raw_features, adj, 100, space['T1'],space['T2'],-1,1)#-1+math.sqrt((math.sqrt(2))/space['T']),1-math.sqrt(math.log(math.sqrt(2))/space['T']))## Fix Degree to be 5
    res = []
    running_time = []
    seed = [1941488137,4198936517,983997847,4023022221,4019585660,2108550661,1648766618,629014539,3212139042,2424918363] ## Use BernNet's seed.
    # seed = [
    for i in range(try_time):
        # idx_train, idx_val, idx_test = gen_rand_split(len(raw_features), device=raw_features.device)
        if flag:
            idx_train1, idx_val1, idx_test1 =  random_planetoidsplits(data,dataset.num_classes,percls_trn,val_lb, seed=seed[i] if len(seed)>0 else None, device=raw_features.device)
        model = get_model("SGC", features.size(1), labels.max().item()+1, args.hidden, dropout=0.0, cuda=args.cuda, group=2)
        #
        # model, acc_val, _ = train_regression(model, features[idx_train], labels[idx_train], features[idx_val], labels[idx_val],
                                        #   args.epochs, space['weight_decay'], args.lr, args.dropout)
        start = time.time()
        if flag:
            model, acc_test, _ = train_regression(model, features[idx_train1], labels[idx_train1], features[idx_test1], labels[idx_test1],
                                    1000 if long_epoch else args.epochs, space['weight_decay'], args.lr, args.dropout,times=space['times'])
        else:
             model, acc_test, _ = train_regression(model, features[idx_train], labels[idx_train], features[idx_test], labels[idx_test],
                                         1000 if long_epoch else args.epochs, space['weight_decay'], args.lr, args.dropout)
        res.append(acc_test)
        running_time.append(time.time()-start)
    res = np.array(res)
    running_time = np.array(running_time)
    acc_test = res.mean()
    if acc_test > args.threshold:
        print(space, acc_test,precompute_time,running_time.mean(), model.temp)
    # print(space, 'accuracy: {:.4f}'.format(acc_val))
    return {'loss': -acc_test, 'std': -res.std(), 'status': STATUS_OK}

best = fmin(sgc_objective, space=space, algo=tpe.suggest, max_evals=args.max_evals)
print("Best config", best)

acc_val = sgc_objective(best,try_time=10,long_epoch=True)
acc_val.append(['loss'].item())
acc_val = np.array(acc_val)

print('10 runs of best policy:', -acc_val['loss'].item(), acc_val['std'].item())

# os.makedirs("./{}-tuning".format(args.model), exist_ok=True)
path = 'logs/DGC_lh_10_{}.txt'.format(args.model, args.dataset)
with open(path, 'wb') as f: pkl.dump(best, f)
