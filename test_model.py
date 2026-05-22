import os

from parse import parse_args
from torch.utils.data import DataLoader
from prettytable import PrettyTable
import time

import numpy as np
import copy
import pickle

from utils import *
from evaluation import *
from model import *
from dataprocess import *

if __name__ == '__main__':
    args = parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.path = os.getcwd()

    seed_everything(args.seed)

    """build dataset"""
    train_cf, val_cf, test_cf, user_dict, args.n_users, clicked_set, adj, user_id_map, reverse_map, label_map, popularity_map = load_data(
        args)

    print(args.n_users, train_cf.shape[0] + val_cf.shape[0], test_cf.shape[0],
          (train_cf.shape[0] + val_cf.shape[0] + test_cf.shape[0]) / args.n_users)

    if (args.neg_in_val_test == 1):  
        clicked_set = user_dict['train_user_set']

    """build model"""
    if args.model == 'LightGCN':
        model = LightGCN(args).to(args.device)


    adj_sp_norm, deg = normalize_edge(adj[:2], args.n_users)
    edge_index, edge_weight = adj_sp_norm._indices(), adj_sp_norm._values()

    model.adj_sp_norm = adj_sp_norm.to(args.device)
    model.edge_index = edge_index.to(args.device)
    model.edge_weight = edge_weight.to(args.device)
    model.deg = deg.to(args.device)

    row, col = edge_index
    args.user_dict = user_dict


    if "fusion" in args.model:
        save_model_filename = args.path + '/trained_model/' + args.dataset + '/' + args.model + "_1.0_" + str(
            args.seed) + '.pkl'
    else:
        if args.reweight_flag == 0:
            save_model_filename = args.path + '/trained_model/' + args.dataset + '/' + args.model + "_" + str(
                args.seed) + '.pkl'
        elif args.reweight_flag == 1:

            save_model_filename = args.path + '/trained_model_reweight/' + args.model + "_" + str(
                args.seed) + '_' + str(args.penalty) + '.pkl'
        else:
            save_model_filename = args.path + '/trained_model_reweight5/' + args.model + "_" + str(
                args.seed) + '_' + str(args.penalty) + '.pkl'
            print(save_model_filename)
    model.load_state_dict(torch.load(save_model_filename))
    model.eval()

    user_embs = model.generate()

    test_per_user(user_embs, user_dict, args)
