import argparse
import torch

def get_citation_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-6,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=128,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', type=str, default="computers",
                        help='Dataset to use.')
    parser.add_argument('--model', type=str, default="SGC",
                        help='model to use.')
    parser.add_argument('--feature', type=str, default="mul",
                        choices=['mul', 'cat', 'adj'],
                        help='feature-type')
    parser.add_argument('--normalization', type=str, default='AugNormAdj',
                       choices=['AugNormAdj'],
                       help='Normalization method for the adjacency matrix.')
    parser.add_argument('--degree', type=int, default=2,
                        help='degree of the approximation.')
    parser.add_argument('--per', type=int, default=-1,
                        help='Number of each nodes so as to balance.')
    parser.add_argument('--experiment', type=str, default="base-experiment",
                        help='feature-type')
    parser.add_argument('--tuned', action='store_true', help='use tuned hyperparams')
    parser.add_argument('-n', '--noise-ratio', type=float, default=0)
    parser.add_argument('-T', '--T', type=float, default=2)
    parser.add_argument('--epsilon', type=float, default=0.0)
    parser.add_argument('--aug', type=float, default=1.0)
    parser.add_argument('--random-split', action='store_true', help='use tuned hyperparams')
    parser.add_argument('--scheme', type=str, default="euler")
    parser.add_argument('--method', type=str, default="tsne")
    parser.add_argument('--trial', type=str, default="1")
    parser.add_argument('--pos-dim', type=int, default=2)
    parser.add_argument('--max-evals', type=int, default=100)
    parser.add_argument('--threshold', type=float, default=0.45)
    parser.add_argument('--K',type=int, default=10)
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args
