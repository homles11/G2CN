import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from scipy.special import comb
from functools import partial
import time
import itertools
import sys
import pickle as pkl
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
import networkx as nx
import scipy.sparse as ssp
from normalization import aug_normalized_adjacency, row_normalize, normalized_adjacency
from time import perf_counter

import networkx as nx
from sklearn.model_selection import train_test_split
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add
import matplotlib.pyplot as plt

import torch.sparse as ts
import math

def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

def coo_block_diag(arrs):
    bad_args = [k for k in range(len(arrs)) if not (isinstance(arrs[k], torch.Tensor) and arrs[k].ndim == 2)]
    if bad_args:
        raise ValueError("arguments in the following positions must be 2-dimension tensor: %s" % bad_args)

    shapes = torch.tensor([a.shape for a in arrs])

    i = []
    v = []
    r, c = 0, 0
    for k, (rr, cc) in enumerate(shapes):
        i += [arrs[k]._indices() + torch.tensor([[r],[c]]).to(arrs[0].device)]
        v += [arrs[k]._values()]
        r += rr
        c += cc
    if arrs[0].is_cuda:
        out = torch.cuda.sparse.FloatTensor(torch.cat(i, dim=1).to(arrs[0].device), torch.cat(v), torch.sum(shapes, dim=0).tolist())
    else:
        out = torch.sparse.FloatTensor(torch.cat(i, dim=1), torch.cat(v), torch.sum(shapes, dim=0).tolist())
    return out




def sp_eye(n, device=torch.device('cuda')):
    eye = sp.eye(n).tocoo()
    eye = sparse_mx_to_torch_sparse_tensor(eye).float().to(device)
    return eye

def gen_rand_split(n_samples, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, device=torch.device('cpu')):
    rand_idx = torch.randperm(n_samples, device=device)
    n_train = int(train_ratio * n_samples)
    n_val = int(val_ratio * n_samples)
    n_test = n_samples - n_train - n_val

    idx_train = rand_idx[:n_train]
    idx_val = rand_idx[n_train:n_train+n_val]
    idx_test = rand_idx[-n_test:]
    
    return idx_train, idx_val, idx_test

def random_planetoidsplits(data, num_classes, percls_trn=20, val_lb=500, seed=None, device='cpu'):#12134):
    index=[i for i in range(0,data.y.shape[0])]
    train_idx=[]
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(data.y.cpu() == c)[0]
        if len(class_idx)<percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn,replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx=rnd_state.choice(rest_index,val_lb,replace=False)
    test_idx=[i for i in rest_index if i not in val_idx]

    return torch.LongTensor(train_idx).to(device), torch.LongTensor(val_idx).to(device), torch.LongTensor(test_idx).to(device)


def plot_tsne(data, labels, n_classes, tsne_dir='figs', file_name='SGC'):

    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import os
    """ Input: 
            - model weights to fit into t-SNE
            - labels (no one hot encode)
            - num_classes
    """
    n_components = 2

    tsne = TSNE(n_components=n_components, init='pca', perplexity=40, random_state=0)
    tsne_res = tsne.fit_transform(data)

    v = pd.DataFrame(data,columns=[str(i) for i in range(data.shape[1])])
    v['y'] = labels
    v['label'] = v['y'].apply(lambda i: str(i))
    v["t1"] = tsne_res[:,0]
    v["t2"] = tsne_res[:,1]

    sns.scatterplot(
        x="t1", y="t2",
        hue="y",
        palette=sns.color_palette(["#52D1DC", "#8D0004", "#845218","#563EAA", "#E44658", "#63C100", "#FF7800"]),
        legend=False,
        data=v,
    )
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('') 
    plt.ylabel('')
    if not os.path.exists(tsne_dir):
        os.makedirs(tsne_dir)
    plt.savefig(os.path.join(tsne_dir, file_name+'_t-SNE.png'))

def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=0.031,
                  num_steps=20,
                  step_size=0.003):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    device = X.device

    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    return X_pgd

def transform_to_noisy_labels(orig_labels, noise_ratio, n_classes):
    # n_samples = len(orig_labels)
    device = orig_labels.device
    noisy_mask = torch.rand(*orig_labels.shape, device=device) < noise_ratio
    rand_mat = torch.rand(len(orig_labels), n_classes, device=device)
    rand_mat.scatter_(1, orig_labels.unsqueeze(1), np.inf)
    noisy_labels = rand_mat.argmin(1)
    # import pdb; pdb.set_trace()
    return orig_labels.masked_scatter(noisy_mask, noisy_labels)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_citation(dataset_str="cora", normalization="AugNormAdj", cuda=True, aug=1.0):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    # normalize
    if normalization == "AugNormAdj":
        adj_normalizer = partial(aug_normalized_adjacency, aug=aug)
    else:
        adj_normalizer = partial(normalized_adjacency, aug=aug)
    adj = adj_normalizer(adj, aug=aug)
    features = row_normalize(features)

    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return adj, features, labels, idx_train, idx_val, idx_test

def sgc_precompute(features, adj, degree):
    t = perf_counter()
    for i in range(degree):
        features = torch.spmm(adj, features)
    precompute_time = perf_counter()-t
    return features, precompute_time

def dgc_precompute(features, adj, degree, T, scheme='euler'):
    if degree == 0 or T == 0.:
        return features, 0.
    compute = dgc_precompute_euler if scheme == 'euler' else dgc_precompute_rk
    return compute(features, adj, degree, T)

def dgc_precompute_dense(features, adj, degree, T, scheme='euler'):
    t = perf_counter()
    h = T * 1.0 / degree
    eye = torch.eye(adj.shape[0], device=adj.device)
    # eye = sparse_mx_to_torch_sparse_tensor(eye).float().to(adj.device)
    op = (1-h) * eye + h * adj
    for i in range(degree):
        features = torch.mm(op, features)
    precompute_time = perf_counter()-t
    return features, precompute_time

def dgc_precompute_gaussian(features, adj, degree, T, bias, scheme='euler'):
    if degree == 0 or T == 0.:
        return features, 0.
    t = perf_counter()
    h = T * 1.0 / degree
    eye = sp.eye(adj.shape[0]).tocoo()
    eye = sparse_mx_to_torch_sparse_tensor(eye).float().to(adj.device)
    Lap = ((1-bias) * eye - adj).to_dense().to_sparse_csr()
    # Lap_sqr = torch.sparse.mm(Lap, Lap)
    # op = eye - h * Lap_sqr
    for i in range(degree):
        features = features - h * torch.spmm(Lap, torch.spmm(Lap, features))
        # features = features - h* torch.spmm(op, features)
    precompute_time = perf_counter()-t
    return features, precompute_time

def precompute_bernstein10(features, adj):
    t = time.time()
    eye = sp.eye(adj.shape[0]).tocoo()
    eye = sparse_mx_to_torch_sparse_tensor(eye).float().to(adj.device)
    adj2 = (2*eye - adj).to_sparse_csr()
    adj = adj.to_sparse_csr()
    feature_list = []
    for j in range(11):
        feature_list.append(features)
        for i in range(j):
            feature_list[j] = torch.spmm(adj, feature_list[j])
        for i in range(10-j):
            feature_list[j] = torch.spmm(adj2, feature_list[j])
        feature_list[j] = 1/(2**10) * comb(10,j) * feature_list[j]
    features = torch.cat(feature_list,dim=1)
    return features, time.time() - t

def precompute_bernstein2(features, adj):
    t = time.time()
    eye = sp.eye(adj.shape[0]).tocoo()
    eye = sparse_mx_to_torch_sparse_tensor(eye).float().to(adj.device)
    adj2 = (2*eye - adj).to_sparse_csr()
    adj = adj.to_sparse_csr()
    feature_list = []
    for j in range(11):
        if j == 0:
            feature_list.append(features)
            for i in range(j):
                feature_list[j] = torch.spmm(adj, feature_list[j])
            for i in range(10-j):
                feature_list[j] = torch.spmm(adj2, feature_list[j])
            feature_list[j] = 1/(2**2) * comb(2,j) * feature_list[j]
        elif j==10:
            feature_list.append(features)
            for i in range(j):
                feature_list[1] = torch.spmm(adj, feature_list[1])
            for i in range(10-j):
                feature_list[1] = torch.spmm(adj2, feature_list[1])
            feature_list[1] = 1/(2**2) * comb(2,2) * feature_list[1]
    features = torch.cat(feature_list,dim=1)
    return features, time.time() - t

def dgc_precompute_lhgaussian(features, adj, degree, T1, T2, biasl=-0.75, biash=0.75, scheme='euler'):
    if degree == 0 or T1 == 0. or T2==0:
        return features, 0.
    t = perf_counter()
    # h1 = T1 * 1.0 / degree
    # h2 = T2 * 1.0 / degree
    h1 = np.around(np.sqrt(T1 * 1.0 / degree),2)
    h2 = np.around(np.sqrt(T2 * 1.0 / degree),2)
    num_nodes = features.size(0)
    eye = sp.eye(adj.shape[0]).tocoo()
    eye = sparse_mx_to_torch_sparse_tensor(eye).float().to(adj.device)
    Lapl = ((1- biasl) * eye - adj) * h1
    Laph = ((1- biash) * eye - adj) * h2
    Lap_expand = coo_block_diag([Lapl, Laph])
    Lap_expand = Lap_expand.to_sparse_csr()
    features_expand = torch.vstack([features,features])
    for i in range(degree):
        features_expand = features_expand - torch.spmm(Lap_expand, torch.spmm(Lap_expand, features_expand))
    features_out = torch.hstack([features_expand[0:num_nodes],features_expand[num_nodes:2*num_nodes]])
    # features_out = features_expand.reshape(2,features.size(0),-1).permute(1,0,2).reshape(features.size(0),-1)
    precompute_time = perf_counter()-t
    return features_out, precompute_time

def dgc_precompute_lhgaussian_taylor(features, adj, degree, T1, T2, biasl=-0.75, biash=0.75, scheme='euler'):
    if degree == 0 or T1 == 0. or T2 == 0:
        return features, 0.
    t = perf_counter()
    # h = T * 1.0 / degree
    num_nodes = features.size(0)
    eye = sp.eye(adj.shape[0]).tocoo()
    eye = sparse_mx_to_torch_sparse_tensor(eye).float().to(adj.device)
    Lapl = ((1- biasl) * eye - adj)*np.around(np.sqrt(T1),2)
    Laph = ((1- biash) * eye - adj)*np.around(np.sqrt(T2),2)
    Lap_expand = coo_block_diag([Lapl, Laph]).to_sparse_csr()
    features_expand = torch.vstack([features,features])

    current_features_extend = features_expand
    factor = 1
    for i in range(degree):
        current_features_extend = torch.spmm(Lap_expand, torch.spmm(Lap_expand, current_features_extend))*(-1)/(i+1)
        features_expand = features_expand + current_features_extend
    features_out = torch.hstack([features_expand[0:num_nodes],features_expand[num_nodes:2*num_nodes]])
    precompute_time = perf_counter()-t
    return features_out, precompute_time

def dgc_precompute_lhgaussian2(features, adj, degree, T, biasl=-0.75, biash=0.75, scheme='euler'):
    if degree == 0 or T == 0.:
        return features, 0.
    t = perf_counter()
    h = T * 1.0 / degree
    eye = sp.eye(adj.shape[0]).tocoo()
    sparse_tensor = sparse_mx_to_torch_sparse_tensor(eye)
    eye = sparse_tensor.to(features.device)
    Lap = []
    Lapl = ((1- biasl) * eye - adj)
    Laph = ((1- biash) * eye - adj)
    Lap.append(Lapl)
    Lap.append(Laph)
    Lap_extend = torch.stack(Lap)
    # Lap_expand = torch.block_diag(Lapl, Laph).to_sparse_csr()
    features_extend = torch.stack([features for i in range(2)])
    for i in range(degree):
        features_extend = features_extend - h * torch.bmm(Lap_extend, torch.bmm(Lap_extend, features_extend))

    features_out = features_extend.permute(1,0,2).reshape(features.size(0),-1)
    precompute_time = perf_counter()-t
    return features_out, precompute_time

def dgc_precompute_lhgaussian_iter(features, adj, degree, T, biasl=-0.75, biash=0.75, scheme='euler'):
    # print(degree)
    if degree == 0 or T == 0.:
        return features, 0.
    t = time.time()
    h = T * 1.0 / degree
    eye = sp.eye(adj.shape[0]).tocoo()
    sparse_tensor = sparse_mx_to_torch_sparse_tensor(eye)
    # sparse_tensor = torch.sparse_csr_tensor(torch.tensor(eye.indptr, dtype=torch.int64),
    #                           torch.tensor(eye.indices, dtype=torch.int64),
    #                           torch.tensor(eye.data), dtype=torch.double, device=adj.device)
    eye = sparse_tensor.to(features.device)
    adj = adj.cpu()
    Lapl = ((1- biasl) * eye - adj).to_sparse_csr()
    Laph = ((1- biash) * eye - adj).to_sparse_csr()
    # Lap_expand = torch.block_diag(Lapl, Laph).to_sparse_csr()
    features1 = features
    features2 = features
    t1 = time.time()
    for i in range(degree):
        features1 = features1 - h * torch.spmm(Lapl, torch.spmm(Lapl, features1))
        features2 = features2 - h * torch.spmm(Laph, torch.spmm(Laph, features2))
        if i % 100 == 0:
            print(time.time()-t1)
            t1 = time.time()
    features_out = torch.cat([features1, features2], dim=1)
    precompute_time = time.time()-t
    return features_out, precompute_time

def dgc_precompute_4gaussian(features, adj, degree, T, biasl, biash):
    if degree == 0 or T == 0.:
        return features, 0.
    t = perf_counter()
    h_low = T * 1.0 / degree
    eye = sp.eye(adj.shape[0]).tocoo()
    eye = sparse_mx_to_torch_sparse_tensor(eye).float().to(adj.device)
    attr = []
    attr.append(((1 - biasl) * eye - adj).to_dense())
    attr.append(((1 - biasl - (biash-biasl)*0.33) * eye - adj).to_dense())
    attr.append(((1 - biasl - (biash-biasl)*0.66) * eye - adj).to_dense())
    attr.append(((1 - biash) * eye - adj).to_dense())
    # Lap_expand = torch.sparse.blkdiag(Lap1, Lap2, Lap3, Lap4)
    time1 = time.time()
    # Lap_expand = block_diagonal(attr)
    Lap_expand = torch.block_diag(attr[0],attr[1],attr[2],attr[3]).to_sparse_csr()
    cost_time = time1 - time.time()
    features_expand = torch.block_diag(features,features,features,features)
    for i in range(degree):
        features_expand = features_expand - h_low * torch.spmm(Lap_expand, torch.spmm(Lap_expand, features_expand))
    h, w = features.size(0), features.size(1)
    features_out = torch.cat([features_expand[0:h,0:w],features_expand[h:2*h,w:2*w],features_expand[2*h:3*h,2*w:3*w],features_expand[3*h:4*h,3*w:4*w]],dim=1)
    precompute_time = perf_counter()-t
    return features_out, precompute_time

def dgc_precompute_11gaussian(features, adj, degree, T, biasl, biash):
    # print(degree)
    device = adj.device
    if degree == 0 or T == 0.:
        return features, 0.
    t1 = time.time()
    h_low = T * 1.0 / degree
    eye = sp.eye(adj.shape[0]).tocoo()
    eye = sparse_mx_to_torch_sparse_tensor(eye).float().to(device)
    attr = []
    for idx in range(11):
        attr.append(((1 - biasl - (biash-biasl)*0.1*idx) * eye - adj))
    Lap_expand = coo_block_diag(attr)
    Lap_expand = Lap_expand.to_sparse_csr()
    features_expand = torch.vstack([features for _ in range(11)])
    for i in range(degree):
        features_expand = features_expand - h_low * torch.spmm(Lap_expand, torch.spmm(Lap_expand, features_expand))
    features_out = features_expand.reshape(11, features.size(0), -1).permute(1,0,2).reshape(features.size(0),-1)
    precompute_time = time.time()-t1
    return features_out, precompute_time


def dgc_precompute_11gaussian2(features, adj, degree, T, biasl, biash):
    ### 用BMM不会出现随着Iteration Number增加，矩阵计算变慢的情况
    if degree == 0 or T == 0.:
        return features, 0.
    t = perf_counter()
    h_low = T * 1.0 / degree
    eye = sp.eye(adj.shape[0]).tocoo()
    eye = sparse_mx_to_torch_sparse_tensor(eye).float().to(adj.device)
    # adj_sp = ssp.coo_matrix((adj.values().cpu().numpy(), adj.indices().cpu().numpy()), shape=adj.shape)
    Lap_expand = torch.stack([((1 - biasl - (biash-biasl)*0.1*idx) * eye - adj) for idx in range(11)])
    # t1 = time.time()
    features_expand = torch.stack([features for _ in range(11)])

    for i in range(degree):
        features_expand = features_expand - h_low * torch.bmm(Lap_expand, torch.bmm(Lap_expand, features_expand))
        # if i % 10 == 0:
        #     print(time.time()-t1)
        #     t1 = time.time()
        #     features_expand = torch.stack([features for _ in range(11)])
    features_out = features_expand.permute(1,0,2).reshape(features.size(0),-1)
    precompute_time = perf_counter()-t
    return features_out, precompute_time

def dgc_precompute_11gaussian_iter(features, adj, degree, T, biasl, biash):
    # print(degree)
    if degree == 0 or T == 0.:
        return features, 0.
    t = time.time()
    h = T * 1.0 / degree
    eye = sp.eye(adj.shape[0]).tocoo()
    sparse_tensor = sparse_mx_to_torch_sparse_tensor(eye)
    eye = sparse_tensor.to(features.device)
    Lap0 = ((1 - biasl) * eye - adj).to_sparse_csr()
    Lap1 = ((1 - biasl - (biash-biasl)*0.1*1) * eye - adj).to_sparse_csr()
    Lap2 = ((1 - biasl - (biash-biasl)*0.1*2) * eye - adj).to_sparse_csr()
    Lap3 = ((1 - biasl - (biash-biasl)*0.1*3) * eye - adj).to_sparse_csr()
    Lap4 = ((1 - biasl - (biash-biasl)*0.1*4) * eye - adj).to_sparse_csr()
    Lap5 = ((1 - biasl - (biash-biasl)*0.1*5) * eye - adj).to_sparse_csr()
    Lap6 = ((1 - biasl - (biash-biasl)*0.1*6) * eye - adj).to_sparse_csr()
    Lap7 = ((1 - biasl - (biash-biasl)*0.1*7) * eye - adj).to_sparse_csr()
    Lap8 = ((1 - biasl - (biash-biasl)*0.1*8) * eye - adj).to_sparse_csr()
    Lap9 = ((1 - biasl - (biash-biasl)*0.1*9) * eye - adj).to_sparse_csr()
    Lap10 = ((1 - biash) * eye - adj).to_sparse_csr()

    # Lap_expand = torch.sparse.blkdiag(Lap1, Lap2, Lap3, Lap4)
    features0, features1, features2, features3, features4, features5, features6, features7, features8, features9, features10 = features, features, features, features, features, features, features, features, features, features, features
    t1 = time.time()
    for i in range(degree):
        features0 = features0 - h * torch.spmm(Lap0, torch.spmm(Lap0, features0))
        features1 = features1 - h * torch.spmm(Lap1, torch.spmm(Lap1, features1))
        features2 = features2 - h * torch.spmm(Lap2, torch.spmm(Lap2, features2))
        features3 = features3 - h * torch.spmm(Lap3, torch.spmm(Lap3, features3))
        features4 = features4 - h * torch.spmm(Lap4, torch.spmm(Lap4, features4))
        features5 = features5 - h * torch.spmm(Lap5, torch.spmm(Lap5, features5))
        features6 = features6 - h * torch.spmm(Lap6, torch.spmm(Lap6, features6))
        features7 = features7 - h * torch.spmm(Lap7, torch.spmm(Lap7, features7))
        features8 = features8 - h * torch.spmm(Lap8, torch.spmm(Lap8, features8))
        features9 = features9 - h * torch.spmm(Lap9, torch.spmm(Lap9, features9))
        features10 = features10 - h * torch.spmm(Lap10, torch.spmm(Lap10, features10))
        if i % 20 == 0:
            print(time.time()-t1)
            t1 = time.time()
            features0, features1, features2, features3, features4, features5, features6, features7, features8, features9, features10 = features, features, features, features, features, features, features, features, features, features, features
    features_out = torch.cat([features0, features1, features2, features3, features4, features5, features6, features7, features8, features9, features10], dim=1)
    precompute_time = time.time() - t
    return features_out, precompute_time



def dgc_precompute_gaussian_band4(features, adj, degree, T, bias, scheme='euler'):
    if degree == 0 or T == 0.:
        return features, 0.
    t = perf_counter()
    h = T * 1.0 / degree
    eye = sp.eye(adj.shape[0]).tocoo()
    eye = sparse_mx_to_torch_sparse_tensor(eye).float().to(adj.device)
    Lap = (1-bias) * eye - adj
    Lap_sqr = torch.sparse.mm(Lap, Lap)
    op = eye - h * Lap_sqr
    for i in range(degree):
        features = torch.spmm(op, features)
    precompute_time = perf_counter()-t
    return features, precompute_time

def dgc_precompute_draw(features, adj, degree, T):
    t = time.time()
    h = T * 1.0 / degree
    s,_ = torch.linalg.eig(adj.to_dense())
    s = torch.abs(s)
    s = s.cpu().numpy()
    x = np.linspace(1,40,40)/20
    y = []
    for i in range(len(x)):
        if i == 0:
            tmp_s = (s >= (x[i]-0.05))*(s<=(x[i]))
        else:
            tmp_s = (s > (x[i]-0.05))*(s<=(x[i]))
        y.append(np.sum(tmp_s))
    plt.plot(x, y, color='b', linestyle='-')
    plt.yscale('log')
    plt.xlabel("Eigenvalue", fontsize=18)
    plt.ylabel("Ratio", fontsize=18)
    plt.ylim(0,1e4)
    plt.xlim(0,2.0)
    plt.show()



def dgc_precompute_euler(features, adj, degree, T):
    t = time.time()
    h = T * 1.0 / degree
    eye = sp.eye(adj.shape[0]).tocoo()
    eye = sparse_mx_to_torch_sparse_tensor(eye).float().to(adj.device)
    op = ((1-h) * eye + h * adj).to_sparse_csr() ### CSR要比COO快，因为COO无法进行快速矩阵运算
    for i in range(degree):
        features = torch.spmm(op, features)
    precompute_time = time.time()-t
    return features, precompute_time

def dgc_precompute_rk(features, adj, degree=2, T=None):
    h = 1.0 if T is None else float(T) / degree
    # print(T,degree,h)
    sp_one = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(sp.eye(adj.shape[0]))).cuda()
    L = sp_one - adj
    t0 = perf_counter()
    for i in range(degree):
        k1 = torch.spmm(-L, features)
        k2 = torch.spmm(-L, features + 1/2.0 * h * k1)
        k3 = torch.spmm(-L, features + 1/2.0 * h * k2)
        k4 = torch.spmm(-L, features + h * k3)
        features += 1/6.0 * h * (k1 + 2*k2 + 2*k3 + k4)
    precompute_time = perf_counter()-t0
    return features, precompute_time

def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)

def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir+"reddit_adj.npz")
    data = np.load(dataset_dir+"reddit.npz")

    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']

def load_reddit_data(data_path="data/", normalization="AugNormAdj", cuda=True, aug=1.0):
    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ("data/")
    labels = np.zeros(adj.shape[0])
    labels[train_index]  = y_train
    labels[val_index]  = y_val
    labels[test_index]  = y_test
    adj = adj + adj.T
    train_adj = adj[train_index, :][:, train_index]
    features = torch.FloatTensor(np.array(features))
    features = (features-features.mean(dim=0))/features.std(dim=0)

    # adj_normalizer = fetch_normalization(normalization)
    adj_normalizer = partial(aug_normalized_adjacency, aug=aug)
    adj = adj_normalizer(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    train_adj = adj_normalizer(train_adj)
    train_adj = sparse_mx_to_torch_sparse_tensor(train_adj).float()
    labels = torch.LongTensor(labels)
    if cuda:
        adj = adj.cuda()
        train_adj = train_adj.cuda()
        features = features.cuda()
        labels = labels.cuda()
    return adj, train_adj, features, labels, train_index, val_index, test_index


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500, Flag=0):
    # Set new random planetoid splits:
    # * round(train_rate*len(data)/num_classes) * num_classes labels for training
    # * val_rate*len(data) labels for validation
    # * rest labels for testing

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    if Flag is 0:
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(rest_index[:val_lb], size=data.num_nodes)
        data.test_mask = index_to_mask(
            rest_index[val_lb:], size=data.num_nodes)
    else:
        val_index = torch.cat([i[percls_trn:percls_trn+val_lb]
                               for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn+val_lb:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(rest_index, size=data.num_nodes)
    return data

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_npz(file_name, is_sparse=True):
    if not file_name.endswith('.npz'):
        file_name += '.npz'

    with np.load(file_name) as loader:
        # loader = dict(loader)
        if is_sparse:

            adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                        loader['adj_indptr']), shape=loader['adj_shape'])

            if 'attr_data' in loader:
                features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                             loader['attr_indptr']), shape=loader['attr_shape'])
            else:
                features = None

            labels = loader.get('labels')

        else:
            adj = loader['adj_data']

            if 'attr_data' in loader:
                features = loader['attr_data']
            else:
                features = None

            labels = loader.get('labels')

    return adj, features, labels

def get_adj(dataset, require_lcc=False):
    print('reading %s...' % dataset)
    _A_obs, _X_obs, _z_obs = load_npz(r'data/%s.npz' % dataset)
    _A_obs = _A_obs + _A_obs.T
    _A_obs = _A_obs.tolil()
    _A_obs[_A_obs > 1] = 1

    if _X_obs is None:
        _X_obs = np.eye(_A_obs.shape[0])

    # require_lcc= False
    if require_lcc:
        lcc = largest_connected_components(_A_obs)

        _A_obs = _A_obs[lcc][:,lcc]
        _X_obs = _X_obs[lcc]
        _z_obs = _z_obs[lcc]

        assert _A_obs.sum(0).A1.min() > 0, "Graph contains singleton nodes"

    # whether to set diag=0?
    _A_obs.setdiag(0)
    _A_obs = _A_obs.astype("float32").tocsr()
    _A_obs.eliminate_zeros()

    assert np.abs(_A_obs - _A_obs.T).sum() == 0, "Input graph is not symmetric"
    assert _A_obs.max() == 1 and len(np.unique(_A_obs[_A_obs.nonzero()].A1)) == 1, "Graph must be unweighted"

    return _A_obs, _X_obs, _z_obs

def largest_connected_components(adj, n_components=1):
    """Select the largest connected components in the graph.
    Parameters
    """
    _, component_indices = sp.csgraph.connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep



def load_data(dataset="cora", val_size=0.1, test_size=0.1):

    print('Loading {} dataset...'.format(dataset))
    adj, features, labels = get_adj(dataset)
    features = sp.csr_matrix(features, dtype=np.float32)

    return adj, features, labels


def preprocess(adj, features, labels, preprocess_adj=False, preprocess_feature=False, sparse=False):
    if preprocess_adj == True:
        adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))

    if preprocess_feature:
        features = normalize_f(features)

    labels = torch.LongTensor(labels)
    if sparse:
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        features = sparse_mx_to_torch_sparse_tensor(features)
        # adj = adj
        # features = features
    else:
        features = torch.FloatTensor(np.array(features.todense()))
        adj = torch.FloatTensor(adj.todense())

    return adj, features, labels


def normalize_feature(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(r_mat_inv)
    return mx

def normalize_adj_tensor(adj, sparse=False):
    if sparse:
        adj = to_scipy(adj)
        mx = normalize_adj(adj.tolil())
        return sparse_mx_to_torch_sparse_tensor(mx).cuda()
    else:
        mx = adj + torch.eye(adj.shape[0]).cuda()
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def to_scipy(sparse_tensor):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    values = sparse_tensor._values()
    indices = sparse_tensor._indices()
    return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()))

def get_train_val_test(idx, train_size, val_size, test_size, stratify):

    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=None,
                                                   train_size=train_size + val_size,
                                                   test_size=test_size,
                                                   stratify=stratify)

    if stratify is not None:
        stratify = stratify[idx_train_and_val]

    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=None,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)

    return idx_train, idx_val, idx_test


def unravel_index(index, array_shape):
    rows = index // array_shape[1]
    cols = index % array_shape[1]
    return rows, cols

def likelihood_ratio_filter(node_pairs, modified_adjacency, original_adjacency, d_min, threshold=0.004):
    """
    Filter the input node pairs based on the likelihood ratio test proposed by Zügner et al. 2018, see
    https://dl.acm.org/citation.cfm?id=3220078. In essence, for each node pair return 1 if adding/removing the edge
    between the two nodes does not violate the unnoticeability constraint, and return 0 otherwise. Assumes unweighted
    and undirected graphs.
    """

    N = int(modified_adjacency.shape[0])

    original_degree_sequence = original_adjacency.sum(0)
    current_degree_sequence = modified_adjacency.sum(0)

    # Concatenate the degree sequences
    concat_degree_sequence = torch.cat((current_degree_sequence, original_degree_sequence))

    # Compute the log likelihood values of the original, modified, and combined degree sequences.
    ll_orig, alpha_orig, n_orig, sum_log_degrees_original = degree_sequence_log_likelihood(original_degree_sequence, d_min)
    ll_current, alpha_current, n_current, sum_log_degrees_current = degree_sequence_log_likelihood(
        current_degree_sequence, d_min)

    ll_comb, alpha_comb, n_comb, sum_log_degrees_combined = degree_sequence_log_likelihood(concat_degree_sequence, d_min)

    # Compute the log likelihood ratio
    current_ratio = -2 * ll_comb + 2 * (ll_orig + ll_current)

    # Compute new log likelihood values that would arise if we add/remove the edges corresponding to each node pair.

    new_lls, new_alphas, new_ns, new_sum_log_degrees = updated_log_likelihood_for_edge_changes(node_pairs,
                                                                                               modified_adjacency, d_min)

    # Combination of the original degree distribution with the distributions corresponding to each node pair.
    n_combined = n_orig + new_ns
    new_sum_log_degrees_combined = sum_log_degrees_original + new_sum_log_degrees
    alpha_combined = compute_alpha(n_combined, new_sum_log_degrees_combined, d_min)

    new_ll_combined = compute_log_likelihood(n_combined, alpha_combined, new_sum_log_degrees_combined, d_min)
    new_ratios = -2 * new_ll_combined + 2 * (new_lls + ll_orig)

    # Allowed edges are only those for which the resulting likelihood ratio measure is < than the threshold
    allowed_edges = new_ratios < threshold
    try:
        filtered_edges = node_pairs[allowed_edges.cpu().numpy().astype(np.bool)]
    except:
        filtered_edges = node_pairs[allowed_edges.numpy().astype(np.bool)]

    allowed_mask = torch.zeros(modified_adjacency.shape)
    allowed_mask[filtered_edges.T] = 1
    allowed_mask += allowed_mask.t()
    return allowed_mask, current_ratio


def degree_sequence_log_likelihood(degree_sequence, d_min):
    """
    Compute the (maximum) log likelihood of the Powerlaw distribution fit on a degree distribution.

    """
    # Determine which degrees are to be considered, i.e. >= d_min.

    D_G = degree_sequence[(degree_sequence >= d_min.item())]
    try:
        sum_log_degrees = torch.log(D_G).sum()
    except:
        sum_log_degrees = np.log(D_G).sum()
    n = len(D_G)

    alpha = compute_alpha(n, sum_log_degrees, d_min)
    ll = compute_log_likelihood(n, alpha, sum_log_degrees, d_min)
    return ll, alpha, n, sum_log_degrees

def updated_log_likelihood_for_edge_changes(node_pairs, adjacency_matrix, d_min):

    # For each node pair find out whether there is an edge or not in the input adjacency matrix.

    edge_entries_before = adjacency_matrix[node_pairs.T]

    degree_sequence = adjacency_matrix.sum(1)

    D_G = degree_sequence[degree_sequence >= d_min.item()]
    sum_log_degrees = torch.log(D_G).sum()
    n = len(D_G)


    deltas = -2 * edge_entries_before + 1
    d_edges_before = degree_sequence[node_pairs]

    d_edges_after = degree_sequence[node_pairs] + deltas[:, None]

    # Sum the log of the degrees after the potential changes which are >= d_min
    sum_log_degrees_after, new_n = update_sum_log_degrees(sum_log_degrees, n, d_edges_before, d_edges_after, d_min)

    # Updated estimates of the Powerlaw exponents
    new_alpha = compute_alpha(new_n, sum_log_degrees_after, d_min)
    # Updated log likelihood values for the Powerlaw distributions
    new_ll = compute_log_likelihood(new_n, new_alpha, sum_log_degrees_after, d_min)

    return new_ll, new_alpha, new_n, sum_log_degrees_after


def update_sum_log_degrees(sum_log_degrees_before, n_old, d_old, d_new, d_min):
    # Find out whether the degrees before and after the change are above the threshold d_min.
    old_in_range = d_old >= d_min
    new_in_range = d_new >= d_min
    d_old_in_range = d_old * old_in_range.float()
    d_new_in_range = d_new * new_in_range.float()

    # Update the sum by subtracting the old values and then adding the updated logs of the degrees.
    sum_log_degrees_after = sum_log_degrees_before - (torch.log(torch.clamp(d_old_in_range, min=1))).sum(1) \
                                 + (torch.log(torch.clamp(d_new_in_range, min=1))).sum(1)

    # Update the number of degrees >= d_min
    new_n = n_old - (old_in_range!=0).sum(1) + (new_in_range!=0).sum(1)
    new_n = new_n.float()
    return sum_log_degrees_after, new_n

def compute_alpha(n, sum_log_degrees, d_min):

    try:
        alpha =  1 + n / (sum_log_degrees - n * torch.log(d_min - 0.5))
    except:
        alpha =  1 + n / (sum_log_degrees - n * np.log(d_min - 0.5))
    return alpha

def compute_log_likelihood(n, alpha, sum_log_degrees, d_min):
    # Log likelihood under alpha
    try:
        ll = n * torch.log(alpha) + n * alpha * torch.log(d_min) + (alpha + 1) * sum_log_degrees
    except:
        ll = n * np.log(alpha) + n * alpha * np.log(d_min) + (alpha + 1) * sum_log_degrees

    return ll

def ravel_multiple_indices(ixs, shape, reverse=False):
    """
    "Flattens" multiple 2D input indices into indices on the flattened matrix, similar to np.ravel_multi_index.
    Does the same as ravel_index but for multiple indices at once.
    Parameters
    """
    if reverse:
        return ixs[:, 1] * shape[1] + ixs[:, 0]

    return ixs[:, 0] * shape[1] + ixs[:, 1]


