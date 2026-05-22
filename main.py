import os

from parse import parse_args
from torch.utils.data import Subset, DataLoader
from prettytable import PrettyTable
import time
import gc

import numpy as np
import copy
import pickle

from utils import *
from evaluation import *
from model import *
from dataprocess import *


def run(model, optimizer, train_cf, clicked_set, user_dict, adj, args, label_map):
    test_avg_best, early_stop_count = -float('inf'), 0

    adj_sp_norm, deg = normalize_edge(adj[:2], args.n_users)
    edge_index, edge_weight = adj_sp_norm._indices(), adj_sp_norm._values()

    model.adj_sp_norm = adj_sp_norm.to(args.device)
    model.edge_index = edge_index.to(args.device)
    model.edge_weight = edge_weight.to(args.device)
    model.deg = deg.to(args.device)
    row, col = edge_index
    args.user_dict = user_dict

    losses, recall, ndcg, precision, hit_ratio, F1, rec_percentage, being_rec_percentage, system_score, rec_nums = defaultdict(
        list), defaultdict(
        list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(
        list), defaultdict(list), defaultdict(list), defaultdict(list)
    in_rec_percentage, in_being_rec_percentage, in_system_score, active_rec_percentage, active_being_rec_percentage, active_system_score = defaultdict(
        list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    start = time.time()

    total_losses = []

    for epoch in range(args.epochs):
        # print(epoch)
        neg_cf = neg_sample_before_epoch(train_cf, clicked_set, args)

        dataset = Dataset(
            users=train_cf[:, 0], pos_items=train_cf[:, 1], neg_items=neg_cf, args=args)

        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                collate_fn=dataset.collate_batch,
                                pin_memory=args.pin_memory)  # organzie the dataloader based on re-sampled negative pairs

        """training"""
        model.train()
        loss = 0

        batch_losses = []
        for i, batch in enumerate(dataloader):

            batch = batch_to_gpu(batch, args.device)

            user_embs, pos_item_embs, neg_item_embs, user_embs0, pos_item_embs0, neg_item_embs0 = model(batch)
            if args.model == "LightGCN" or "CAGCN" in args.model:
                bpr_loss = cal_bpr_loss_light_gcn(user_embs, pos_item_embs, neg_item_embs)
            else:
                bpr_loss = cal_bpr_loss(user_embs, pos_item_embs, neg_item_embs)

            # l2 regularization
            l2_loss = cal_l2_loss(
                user_embs0, pos_item_embs0, neg_item_embs0, user_embs0.shape[0])
            batch_loss = bpr_loss + args.l2 * l2_loss
            batch_losses.append(batch_loss.item())

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            del user_embs, pos_item_embs, neg_item_embs, user_embs0, pos_item_embs0, neg_item_embs0

            gc.collect()

            loss += batch_loss.item()

        # ******************evaluation****************
        total_losses.append(loss / (i + 1))

        if not epoch % 10:
            model.eval()
            res = PrettyTable()
            res.field_names = ["Time", "Epoch", "Training_loss",
                               "Recall", "NDCG", "Precision", "Hit_ratio", "F1",
                               "rec_percentage", "being_rec_percentage", "system_score", "rec_nums",
                               "in_rec_percentage", "in_being_rec_percentage", "in_system_score",
                               "active_rec_percentage", "active_being_rec_percentage", "active_system_score"]

            user_embs = model.generate()
            test_res = test(user_embs, user_dict, args, label_map, flag='val')
            res.add_row(
                [format(time.time() - start, '.4f'), epoch, format(loss / (i + 1), '.4f'), test_res['Recall'],
                 test_res['NDCG'],
                 test_res['Precision'], test_res['Hit_ratio'], test_res['F1'], test_res['Recommendation_Percentage'],
                 test_res['Being_Recommended_Percentage'], test_res['System_Score'], test_res['Rec_Nums_Inactive'],
                 test_res['In_Recommendation_Percentage'], test_res['In_Being_Recommended_Percentage'],
                 test_res['In_System_Score'], test_res['Active_Recommendation_Percentage'],
                 test_res['Active_Being_Recommended_Percentage'], test_res['Active_System_Score']])

            print(res)
            with open(args.path + "/log_without_reweight.txt", "a") as log_file:
                log_file.write(res.get_string() + "\n")
            for k in args.topks:
                rec_percentage[k].append(test_res['Recommendation_Percentage'])
                being_rec_percentage[k].append(test_res['Being_Recommended_Percentage'])
                system_score[k].append(test_res['System_Score'])
                in_rec_percentage[k].append(test_res['In_Recommendation_Percentage'])
                in_being_rec_percentage[k].append(test_res['In_Being_Recommended_Percentage'])
                in_system_score[k].append(test_res['In_System_Score'])
                active_rec_percentage[k].append(test_res['Active_Recommendation_Percentage'])
                active_being_rec_percentage[k].append(test_res['Active_Being_Recommended_Percentage'])
                active_system_score[k].append(test_res['Active_System_Score'])
                rec_nums[k].append(test_res['Rec_Nums_Inactive'])
                recall[k].append(test_res['Recall'])
                ndcg[k].append(test_res['NDCG'])
                precision[k].append(test_res['Precision'])
                hit_ratio[k].append(test_res['Hit_ratio'])
                F1[k].append(test_res['F1'])
                losses[k].append(loss / (i + 1))

            # *********************************************************
            # 3 relates to topk=20

            utility_measurement = calculate_percentages_avg(test_res['Recommendation_Percentage'],
                                                            test_res['Being_Recommended_Percentage'])

            if utility_measurement > test_avg_best:
                test_avg_best = utility_measurement
                early_stop_count = 0

            if args.save:
                torch.save(model.state_dict(), os.getcwd() +
                           '/trained_model/' + args.dataset + '/' + args.model + "_" + str(args.seed) + '.pkl')

    for k in args.topks:
        save_path = os.path.join(os.getcwd(), str(k) + "training_metrics_noreweight.png")
        plot_training_metrics(losses, rec_percentage, being_rec_percentage, system_score, rec_nums, k, save_path)
    print("Best validation: ", test_avg_best)


if __name__ == '__main__':
    args = parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.path = os.getcwd()

    seed_everything(args.seed)
    print("Configurations:", args)

    """build dataset"""
    train_cf, val_cf, test_cf, user_dict, args.n_users, clicked_set, adj, user_id_map, reverse_map, label_map = load_data(
        args)

    print(args.n_users, train_cf.shape[0] + val_cf.shape[0], test_cf.shape[0],
          (train_cf.shape[0] + val_cf.shape[0] + test_cf.shape[0]) / args.n_users)

    if (args.neg_in_val_test == 1):
        clicked_set = user_dict['train_user_set']

    """build model"""
    if args.model == 'GraphConv':
        model = GraphConv(args).to(args.device)
    elif args.model == 'LightGCN':
        model = LightGCN(args).to(args.device)

    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    paths = [args.path + "/trained_model/", args.path + "/trained_model/" + args.dataset]
    for p in paths:
        if not os.path.exists(p):
            os.mkdir(p)
            print("path has been created: ", p)

    run(model, optimizer, train_cf, clicked_set, user_dict, adj, args, label_map)
