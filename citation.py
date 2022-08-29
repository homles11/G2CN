import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_citation, sgc_precompute, set_seed, gen_rand_split
from models import get_model
from metrics import accuracy
import pickle as pkl
from args import get_citation_args
from time import perf_counter


def train_regression(model,
                     train_features, train_labels,
                     val_features, val_labels,
                     epochs=100, weight_decay=5e-6,
                     lr=0.2, dropout=0.):

    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    t = perf_counter()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_features)
        loss_train = F.cross_entropy(output, train_labels)
        loss_train.backward()
        optimizer.step()
    train_time = perf_counter()-t

    with torch.no_grad():
        model.eval()
        output = model(val_features)
        acc_val = accuracy(output, val_labels)

    return model, acc_val, train_time

def test_regression(model, test_features, test_labels):
    model.eval()
    return accuracy(model(test_features), test_labels)

def main():

    args = get_citation_args()

    if args.tuned:
        if args.model == "SGC":
            with open("{}-tuning/{}.txt".format(args.model, args.dataset), 'rb') as f:
                args.weight_decay = pkl.load(f)['weight_decay']
                print("using tuned weight decay: {}".format(args.weight_decay))
        else:
            raise NotImplemented

    # setting random seeds
    set_seed(args.seed, args.cuda)

    adj, features, labels, idx_train, idx_val, idx_test = load_citation(args.dataset, args.normalization, args.cuda)

    if args.random_split:
        split_path = f'data/data_splits/{args.dataset}_idx.pt'
        try:
            idx_train, idx_val, idx_test = torch.load(split_path)
            print('data_split loaded from', split_path)
        except:
            idx_train, idx_val, idx_test = gen_rand_split(len(raw_features), device=raw_features.device)
            torch.save([idx_train, idx_val, idx_test], split_path)
            print('gen data split and save to', split_path)

    # import pdb; pdb.set_trace()

    model = get_model(args.model, features.size(1), labels.max().item()+1, args.hidden, args.dropout, args.cuda)

    if args.model == "SGC": features, precompute_time = sgc_precompute(features, adj, args.degree)
    print("{:.4f}s".format(precompute_time))

    if args.model == "SGC":
        model, acc_val, train_time = train_regression(model, features[idx_train], labels[idx_train], features[idx_val], labels[idx_val],
                        args.epochs, args.weight_decay, args.lr, args.dropout)
        acc_test = test_regression(model, features[idx_test], labels[idx_test])

    print("Validation Accuracy: {:.4f} Test Accuracy: {:.4f}".format(acc_val, acc_test))
    print("Pre-compute time: {:.4f}s, train time: {:.4f}s, total: {:.4f}s".format(precompute_time, train_time, precompute_time+train_time))

if __name__ == '__main__':
    main()