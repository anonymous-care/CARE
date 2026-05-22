import numpy as np
import torch
import pickle
import os
from utils import *
import networkx as nx
from collections import defaultdict, Counter
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

np.set_printoptions(precision=4)


def create_graph(test_data, pred_data):
    G = nx.Graph()

    for user_id, (true_users, pred_users) in enumerate(zip(test_data, pred_data)):
        user_node = f"user_{user_id}"
        for other_user in true_users:
            G.add_edge(user_node, f"user_{other_user}")

        for other_user in pred_users:
            G.add_edge(user_node, f"user_{other_user}")

    return G


# not used currently
def calculate_recommendation_percentages(user_id, test_data, pred_data):
    pred_data = pred_data[:, :20]
    actual_interacted_users = set(test_data[user_id])
    recommended_users = set(pred_data[user_id])

    if len(actual_interacted_users) == 0:
        return None, None

    correctly_recommended_users = actual_interacted_users.intersection(recommended_users)
    total_recommended_users = len(recommended_users)

    if total_recommended_users == 0:
        return None, None

    user_recommendation_percentage = (len(correctly_recommended_users) / total_recommended_users) * 100

    users_recommended_to = [u for u in range(len(pred_data)) if user_id in pred_data[u] and u != user_id]

    correctly_predicted_users = sum([1 for u in users_recommended_to if user_id in test_data[u]])

    total_other_users = sum([1 for u in range(len(test_data)) if user_id in test_data[u]])

    user_being_recommended_percentage = (
        (correctly_predicted_users / total_other_users) * 100 if total_other_users > 0 else 0
    )

    return user_recommendation_percentage, user_being_recommended_percentage


def calculate_recommendation_percentages_all_users(test_data, pred_data, label_map, popularity_map):
    pred_data = np.vstack(pred_data)[:, :20]
    num_users = pred_data.shape[0]

    all_rec_percentages, all_being_rec_percentages = [], []

    active_rec_percentages, active_being_rec_percentages = [], []
    in_rec_percentages, in_being_rec_percentages = [], []

    inactive_recommendation_counts = {}

    active_pop_rec = []
    active_pop_being = []
    active_nor_rec = []
    active_nor_being = []
    inactive_pop_rec = []
    inactive_pop_being = []
    inactive_nor_rec = []
    inactive_nor_being = []

    for user_id in range(num_users):
        actual_interacted_users = set(test_data[user_id]) if user_id in test_data else set()
        recommended_users = set(map(int, pred_data[user_id]))

        if len(actual_interacted_users) == 0 or len(recommended_users) == 0:
            continue

        correctly_recommended = actual_interacted_users.intersection(recommended_users)
        rec_percentage = (len(correctly_recommended) / len(recommended_users)) * 100
        all_rec_percentages.append(rec_percentage)

        users_recommended_to = [u for u in range(num_users) if u != user_id and user_id in map(int, pred_data[u])]
        correctly_predicted = sum(1 for u in users_recommended_to if user_id in test_data.get(u, set()))
        total_other = sum(1 for u in range(num_users) if user_id in test_data.get(u, set()))
        if total_other > 0:
            being_percentage = (correctly_predicted / total_other) * 100
            all_being_rec_percentages.append(being_percentage)
        else:
            being_percentage = 0

        if label_map.get(user_id, -1) == 1:
            rec_count = sum(1 for rec_list in pred_data for rec_user in rec_list if rec_user == user_id)
            inactive_recommendation_counts[user_id] = rec_count

        user_label = label_map.get(user_id, 1)
        pop_tag = popularity_map.get(user_id, 0)

        if user_label == 2:
            active_rec_percentages.append(rec_percentage)
            if total_other > 0:
                active_being_rec_percentages.append(being_percentage)
            if pop_tag == 1:
                active_pop_rec.append(rec_percentage)
                if total_other > 0:
                    active_pop_being.append(being_percentage)
            else:
                active_nor_rec.append(rec_percentage)
                if total_other > 0:
                    active_nor_being.append(being_percentage)
        elif user_label == 1:
            in_rec_percentages.append(rec_percentage)
            if total_other > 0:
                in_being_rec_percentages.append(being_percentage)

            if pop_tag == 1:
                inactive_pop_rec.append(rec_percentage)
                if total_other > 0:
                    inactive_pop_being.append(being_percentage)
            else:
                inactive_nor_rec.append(rec_percentage)
                if total_other > 0:
                    inactive_nor_being.append(being_percentage)

    global_rec = np.mean(all_rec_percentages) if all_rec_percentages else 0
    global_being = np.mean(all_being_rec_percentages) if all_being_rec_percentages else 0
    global_system = 0.5 * (global_rec + global_being)

    active_rec = np.mean(active_rec_percentages) if active_rec_percentages else 0
    active_being = np.mean(active_being_rec_percentages) if active_being_rec_percentages else 0
    active_system = 0.5 * (active_rec + active_being)

    inactive_rec = np.mean(in_rec_percentages) if in_rec_percentages else 0
    inactive_being = np.mean(in_being_rec_percentages) if in_being_rec_percentages else 0
    inactive_system = 0.5 * (inactive_rec + inactive_being)

    active_pop_rec_avg = np.mean(active_pop_rec) if active_pop_rec else 0
    active_pop_being_avg = np.mean(active_pop_being) if active_pop_being else 0
    active_pop_system = 0.5 * (active_pop_rec_avg + active_pop_being_avg)

    active_nor_rec_avg = np.mean(active_nor_rec) if active_nor_rec else 0
    active_nor_being_avg = np.mean(active_nor_being) if active_nor_being else 0
    active_nor_system = 0.5 * (active_nor_rec_avg + active_nor_being_avg)

    inactive_pop_rec_avg = np.mean(inactive_pop_rec) if inactive_pop_rec else 0
    inactive_pop_being_avg = np.mean(inactive_pop_being) if inactive_pop_being else 0
    inactive_pop_system = 0.5 * (inactive_pop_rec_avg + inactive_pop_being_avg)

    inactive_nor_rec_avg = np.mean(inactive_nor_rec) if inactive_nor_rec else 0
    inactive_nor_being_avg = np.mean(inactive_nor_being) if inactive_nor_being else 0
    inactive_nor_system = 0.5 * (inactive_nor_rec_avg + inactive_nor_being_avg)

    avg_inactive_recommendations = np.mean(
        list(inactive_recommendation_counts.values())) if inactive_recommendation_counts else 0

    results = {
        "Rec_percentage": global_rec,
        "Being_rec_percentage": global_being,
        "System_score": global_system,
        "Active_Rec_percentage": active_rec,
        "Active_Being_rec_percentage": active_being,
        "Active_System_score": active_system,
        "In_Rec_percentage": inactive_rec,
        "In_Being_rec_percentage": inactive_being,
        "In_System_score": inactive_system,
        "Rec_Nums_Inactive": avg_inactive_recommendations,
        "Active_Pop_Rec_percentage": active_pop_rec_avg,
        "Active_Pop_Being_rec_percentage": active_pop_being_avg,
        "Active_Pop_System_score": active_pop_system,
        "Active_Nor_Rec_percentage": active_nor_rec_avg,
        "Active_Nor_Being_rec_percentage": active_nor_being_avg,
        "Active_Nor_System_score": active_nor_system,
        "Inactive_Pop_Rec_percentage": inactive_pop_rec_avg,
        "Inactive_Pop_Being_rec_percentage": inactive_pop_being_avg,
        "Inactive_Pop_System_score": inactive_pop_system,
        "Inactive_Nor_Rec_percentage": inactive_nor_rec_avg,
        "Inactive_Nor_Being_rec_percentage": inactive_nor_being_avg,
        "Inactive_Nor_System_score": inactive_nor_system
    }

    return results


def calculate_percentages_avg(r, b_r):
    return 0.5 * r + 0.5 * b_r


def getLabel(test_data, pred_data):
    r = []

    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)

    return np.array(r).astype('float')


def Hit_at_k(r, k):
    right_pred = r[:, :k].sum(axis=1)

    return 1. * (right_pred > 0)


def RecallPrecision_ATk(test_data, r, k):
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.divide(right_pred, recall_n, out=np.zeros_like(right_pred, dtype=float), where=recall_n > 0)
    precis = right_pred / precis_n
    return {'Recall': recall, 'Precision': precis}


def NDCGatK_r(test_data, r, k):
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix

    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = np.sum(pred_data * (1. / np.log2(np.arange(2, k + 2))), axis=1)

    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg

    return ndcg


# not used currently
def get_weight(userLabel):
    w1 = 0
    # weight for recommand
    w2 = 0
    # weight for being_recommand
    if userLabel == 1:
        w1 = 0.3
        w2 = 0.7
    elif userLabel == 2:
        w1 = 0.5
        w2 = 0.5
    return w1, w2


def test_one_batch(X, topks, label_map, batch_users):
    sorted_users = X[0].numpy()
    groundTrue = X[1]
    userSet = set(label_map.keys())
    inactive_users = 0
    active_users = 0
    rec_nums = []

    r = getLabel(groundTrue, sorted_users)

    pre, recall, ndcg, hit_ratio, F1 = [], [], [], [], []
    for k in topks:
        ret = RecallPrecision_ATk(groundTrue, r, k)
        ndcgs = NDCGatK_r(groundTrue, r, k)
        hit_ratios = Hit_at_k(r, k)

        hit_ratio.append(sum(hit_ratios))
        pre.append(sum(ret['Precision']))
        recall.append(sum(ret['Recall']))
        ndcg.append(sum(ndcgs))

        temp = ret['Precision'] + ret['Recall']
        temp[temp == 0] = float('inf')
        F1s = 2 * ret['Precision'] * ret['Recall'] / temp

        F1.append(sum(F1s))

    return {
        'Recall': np.array(recall),
        'Precision': np.array(pre),
        'NDCG': np.array(ndcg),
        'F1': np.array(F1),
        'Hit_ratio': np.array(hit_ratio)
    }


def test(user_embs, user_dict, args, label_map, popularity_map, flag='val'):
    results = {'Precision': np.zeros(len(args.topks)),
               'Recall': np.zeros(len(args.topks)),
               'NDCG': np.zeros(len(args.topks)),
               'Hit_ratio': np.zeros(len(args.topks)),
               'F1': np.zeros(len(args.topks)),
               "Rec_percentage": 0.0,
               "Being_rec_percentage": 0.0,
               "System_score": 0.0,
               "Active_Rec_percentage": 0.0,
               "Active_Being_rec_percentage": 0.0,
               "Active_System_score": 0.0,
               "In_Rec_percentage": 0.0,
               "In_Being_rec_percentage": 0.0,
               "In_System_score": 0.0,
               "Rec_Nums_Inactive": 0.0,
               "Active_Pop_Rec_percentage": 0.0,
               "Active_Pop_Being_rec_percentage": 0.0,
               "Active_Pop_System_score": 0.0,
               "Active_Nor_Rec_percentage": 0.0,
               "Active_Nor_Being_rec_percentage": 0.0,
               "Active_Nor_System_score": 0.0,
               "Inactive_Pop_Rec_percentage": 0.0,
               "Inactive_Pop_Being_rec_percentage": 0.0,
               "Inactive_Pop_System_score": 0.0,
               "Inactive_Nor_Rec_percentage": 0.0,
               "Inactive_Nor_Being_rec_percentage": 0.0,
               "Inactive_Nor_System_score": 0.0
               }

    train_user_set = user_dict['train_user_set']
    val_user_set = user_dict['val_user_set']
    num_inactive = 0
    keys = label_map.keys()
    for key in keys:
        if label_map[key] == 1:
            num_inactive += 1

    if flag == "test":
        test_user_set = user_dict['test_user_set']
        test_users = torch.tensor(list(train_user_set.keys()))
    elif flag == "val":
        test_user_set = user_dict['val_user_set']
        test_users = torch.tensor(list(train_user_set.keys()))

    with torch.no_grad():
        users_list = []
        ratings_list = []
        groundTruth_items_list = []

        for batch_users in minibatch(test_users, batch_size=4096):
            batch_users = batch_users.to(args.device)
            if flag == "val":
                clicked_items = [train_user_set[user.item()] for user in batch_users]
            elif flag == "test":
                clicked_items = []
                for user in batch_users:
                    clicked_items_for_user_in_train = train_user_set[user.item()]
                    clicked_items_for_user_in_val = val_user_set[user.item()]
                    clicked_items.append(
                        np.concatenate([clicked_items_for_user_in_train, clicked_items_for_user_in_val]))

            exclude_index = []
            exclude_items = []

            for range_i, items in enumerate(clicked_items):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)

            rating_batch = torch.matmul(user_embs[batch_users], user_embs.t())
            rating_batch[exclude_index, exclude_items] = -(1 << 10)

            groundTruth_items = [test_user_set[user.item()] for user in batch_users]

            rating_K = torch.topk(rating_batch, k=max(args.topks))[1].cpu()

            users_list.append(batch_users)
            ratings_list.append(rating_K)
            groundTruth_items_list.append(groundTruth_items)

        X = zip(ratings_list, groundTruth_items_list, users_list)
        pre_results = []

        for x in X:
            pre_results.append(test_one_batch(x, args.topks, label_map, batch_users))

        for result in pre_results:
            results['Recall'] += result['Recall']
            results['Precision'] += result['Precision']
            results['NDCG'] += result['NDCG']
            results['F1'] += result['F1']
            results['Hit_ratio'] += result['Hit_ratio']

        rec_stats = calculate_recommendation_percentages_all_users(test_user_set, ratings_list, label_map,
                                                                   popularity_map)

        results.update(rec_stats)
        results['Recall'] /= len(test_users)
        results['Precision'] /= len(test_users)
        results['NDCG'] /= len(test_users)
        results['F1'] /= len(test_users)
        results['Hit_ratio'] /= len(test_users)

    return results


# grouping by ratio
def split_groups_horizontally(filename, user_id_map, group_num=10):
    ratio_range = np.linspace(0, 1, group_num + 1)
    id_to_ratio = dict()

    with open(filename, "r") as f:
        for line in f.readlines():
            line_split = line.rstrip().split()
            uid = int(line_split[0])
            ratio = float(line_split[1])
            newuid = -1
            if uid in user_id_map.keys():
                newuid = user_id_map[uid]
            id_to_ratio[newuid] = ratio

    groups = []
    for i, v in enumerate(ratio_range[:-1]):
        start_value = v
        end_value = ratio_range[i + 1]
        groups.append([k for k, v in filter(lambda x: x[1] >= start_value and x[1] < end_value, id_to_ratio.items())])

    final_end_value = ratio_range[-1]
    groups[-1].extend([k for k, v in filter(lambda x: x[1] == final_end_value, id_to_ratio.items())])

    return groups


# group by label
def split_groups_horizontally_label(filename, user_id_map):
    with open(filename, "rb") as f:
        id_to_ratio = pickle.load(f)

    mapped_id_to_ratio = {
        user_id_map.get(uid, -1): ratio
        for uid, ratio in id_to_ratio.items()
        if uid in user_id_map
    }

    group_1 = [k for k, v in mapped_id_to_ratio.items() if v == 1]  
    group_2 = [k for k, v in mapped_id_to_ratio.items() if v == 2] 

    return [group_1, group_2]


def test_per_user(user_embs, user_dict, args):

    train_user_set = user_dict['train_user_set']
    val_user_set = user_dict['val_user_set']
    test_user_set = user_dict['test_user_set']

    test_users = torch.tensor(list(train_user_set.keys()), dtype=torch.long)

    users_list = []
    ratings_list = []
    ratings_values_list = []

    n_users = int(user_embs.shape[0])
    topK = int(max(args.topks))

    with torch.no_grad():
        print("[test_per_user] n_users(emb) =", n_users, "| topK =", topK)
        print("[test_per_user] max(test_users) =", int(torch.max(test_users)) if len(test_users) > 0 else -1)
        print("[test_per_user] num_eval_users =", len(test_users))

        for batch_users in minibatch(test_users, batch_size=args.test_batch_size):
            batch_users = batch_users.to(args.device)

            clicked_items = []
            for u in batch_users:
                uid = int(u.item())
                tr = train_user_set.get(uid, [])
                va = val_user_set.get(uid, [])
                if len(tr) == 0 and len(va) == 0:
                    clicked_items.append(np.array([], dtype=np.int64))
                else:
                    clicked_items.append(np.concatenate([np.array(tr, dtype=np.int64),
                                                        np.array(va, dtype=np.int64)]))

            exclude_index = []
            exclude_items = []
            for row_i, items in enumerate(clicked_items):
                if items is None or len(items) == 0:
                    continue
                exclude_index.extend([row_i] * len(items))
                exclude_items.extend(items.tolist())

            rating_batch = torch.matmul(user_embs[batch_users], user_embs.t())  

            if len(exclude_index) > 0:
                ex_i = np.asarray(exclude_index, dtype=np.int64)
                ex_j = np.asarray(exclude_items, dtype=np.int64)

                mask = (ex_j >= 0) & (ex_j < n_users)
                ex_i = ex_i[mask]
                ex_j = ex_j[mask]

                if len(ex_i) > 0:
                    rating_batch[torch.from_numpy(ex_i).to(args.device),
                                 torch.from_numpy(ex_j).to(args.device)] = -(1 << 10)

            rating_values, rating_K = torch.topk(rating_batch, k=topK, dim=1) 
            rating_K = rating_K.cpu()
            rating_values = rating_values.cpu()

            users_list.append(batch_users.detach().cpu())
            ratings_list.append(rating_K)
            ratings_values_list.append(rating_values)

    users_np = torch.cat(users_list, dim=0).numpy().astype(np.int64)             
    ratings_mat = torch.cat(ratings_list, dim=0).numpy().astype(np.int64)           
    values_mat = torch.cat(ratings_values_list, dim=0).numpy().astype(np.float32) 

    all_rating = np.full((n_users, topK), -1, dtype=np.int64)
    all_value = np.full((n_users, topK), -1e10, dtype=np.float32)

    all_rating[users_np] = ratings_mat
    all_value[users_np] = values_mat

    if args.reweight_flag == 0:
        save_dir = os.path.join(args.path, "test_logs", args.dataset)
        prefix = os.path.join(save_dir, f"{args.model}_{args.seed}_")
    else:
        save_dir = os.path.join(args.path, "test_logs_reweight5", args.dataset)
        prefix = os.path.join(save_dir, f"{args.model}_{args.seed}_")

    os.makedirs(save_dir, exist_ok=True)

    np.save(prefix + "rating_list.npy", all_rating)
    np.save(prefix + "rating_value_list.npy", all_value)
    np.save(prefix + "user_ids.npy", np.arange(n_users, dtype=np.int64))
    np.save(prefix + "Recall.npy", np.array([]))
    np.save(prefix + "Precision.npy", np.array([]))
    np.save(prefix + "F1.npy", np.array([]))
    np.save(prefix + "NDCG.npy", np.array([]))
    np.save(prefix + "Hit_ratio.npy", np.array([]))
    np.save(prefix + "rec_percentage.npy", np.array(0.0))
    np.save(prefix + "being_rec_percentage.npy", np.array(0.0))
    np.save(prefix + "system_score.npy", np.array(0.0))

    filled = int((all_rating[:, 0] != -1).sum())
    print("[test_per_user] Finish saving records (uid-indexed).")
    print("[test_per_user] saved prefix:", prefix)
    print("[test_per_user] rating_list shape:", all_rating.shape,
          "| filled_users:", filled, "/", n_users,
          "| eval_users:", len(users_np))

import os
import numpy as np
import torch


def _neumf_score(model, users_1d: torch.LongTensor, items_1d: torch.LongTensor) -> torch.Tensor:

    if hasattr(model, "predict"):
        s = model.predict(users_1d, items_1d)
    else:
        s = model(users_1d, items_1d)
    if s.dim() > 1:
        s = s.view(-1)
    return s

def test_neumf(model, user_dict, args, label_map, popularity_map, flag="val"):


    train_user_set = user_dict["train_user_set"]
    val_user_set = user_dict["val_user_set"]
    test_user_set = user_dict["test_user_set"]

    eval_users = torch.tensor(list(train_user_set.keys()), dtype=torch.long)
    n_users = int(args.n_users)
    device = args.device

    topK = int(max(args.topks))
    cand_chunk = int(getattr(args, "cand_chunk", 4096))
    batch_users_size = int(getattr(args, "test_batch_size", 4096))

    model.eval()

    all_rating = np.full((n_users, topK), -1, dtype=np.int64)
    all_value  = np.full((n_users, topK), -1e10, dtype=np.float32)

    candidates_all = torch.arange(n_users, device=device, dtype=torch.long)

    @torch.no_grad()
    def _topk_merge(cur_scores, cur_items, new_scores, new_items, k):
        scores = torch.cat([cur_scores, new_scores], dim=1)
        items  = torch.cat([cur_items,  new_items],  dim=1)
        topv, topi = torch.topk(scores, k=k, dim=1)
        top_items = torch.gather(items, 1, topi)
        return topv, top_items

    with torch.no_grad():

        for bu in minibatch(eval_users, batch_size=batch_users_size):
            bu = bu.to(device)
            B = bu.shape[0]

            running_scores = torch.full((B, topK), -1e10, device=device)
            running_items  = torch.full((B, topK), -1, device=device, dtype=torch.long)

            exclude_sets = []
            bu_list = bu.detach().cpu().tolist()
            for u in bu_list:
                if flag == "val":
                    tr = train_user_set.get(u, [])
                    ex = set(tr)
                    ex.add(u)
                else:
                    tr = train_user_set.get(u, [])
                    va = val_user_set.get(u, [])
                    ex = set(tr) | set(va)
                    ex.add(u)
                exclude_sets.append(ex)

            for start in range(0, n_users, cand_chunk):
                cand = candidates_all[start:start + cand_chunk] 
                scores = model.score_user_candidates(bu, cand)  
                scores = scores.clone()

                cand_cpu = cand.detach().cpu().numpy()
                for i_u, ex in enumerate(exclude_sets):
                    if not ex:
                        continue
                    m = np.isin(cand_cpu, list(ex))
                    if m.any():
                        idx = torch.from_numpy(np.where(m)[0]).to(device)
                        scores[i_u, idx] = -1e10

                k2 = min(topK, scores.shape[1])
                topv, topi = torch.topk(scores, k=k2, dim=1)
                top_items = cand[topi]

                running_scores, running_items = _topk_merge(
                    running_scores, running_items, topv, top_items, topK
                )

            bu_cpu = np.asarray(bu_list, dtype=np.int64)
            all_rating[bu_cpu] = running_items.detach().cpu().numpy().astype(np.int64)
            all_value[bu_cpu]  = running_scores.detach().cpu().numpy().astype(np.float32)

    if int(getattr(args, "reweight_flag", 0)) == 0:
        save_dir = os.path.join(args.path, "test_logs", args.dataset)
    else:
        save_dir = os.path.join(args.path, "test_logs_reweight5", args.dataset)

    os.makedirs(save_dir, exist_ok=True)
    prefix = os.path.join(save_dir, f"{args.model}_{args.seed}_")
    np.save(prefix + "rating_list.npy", all_rating)
    np.save(prefix + "rating_value_list.npy", all_value)
    np.save(prefix + "user_ids.npy", np.arange(n_users, dtype=np.int64))

    gt = val_user_set if flag == "val" else test_user_set

    eval_users_np = np.asarray(list(train_user_set.keys()), dtype=np.int64)
    pred_for_eval = all_rating[eval_users_np, :int(max(args.topks))]
    groundTruth   = [gt.get(int(u), []) for u in eval_users_np]

    r = getLabel(groundTruth, pred_for_eval)

    results = {
        "Precision": np.zeros(len(args.topks)),
        "Recall": np.zeros(len(args.topks)),
        "NDCG": np.zeros(len(args.topks)),
        "Hit_ratio": np.zeros(len(args.topks)),
        "F1": np.zeros(len(args.topks)),
    }

    for i_k, k in enumerate(args.topks):
        ret = RecallPrecision_ATk(groundTruth, r, k)
        ndcgs = NDCGatK_r(groundTruth, r, k)
        hit_ratios = Hit_at_k(r, k)

        pre = ret["Precision"]
        rec = ret["Recall"]
        tmp = pre + rec
        tmp[tmp == 0] = float("inf")
        f1 = 2 * pre * rec / tmp

        results["Precision"][i_k] = float(np.mean(pre))
        results["Recall"][i_k] = float(np.mean(rec))
        results["NDCG"][i_k] = float(np.mean(ndcgs))
        results["Hit_ratio"][i_k] = float(np.mean(hit_ratios))
        results["F1"][i_k] = float(np.mean(f1))

    rec_stats = calculate_recommendation_percentages_all_users(
        gt, [pred_for_eval], label_map, popularity_map
    )
    results.update(rec_stats)

    filled = int((all_rating[:, 0] != -1).sum())
    print(f"[test_neumf] saved prefix={prefix} | rating_list shape={all_rating.shape} | filled_users={filled}/{n_users}")

    return results

@torch.no_grad()
def test_per_user_neumf(model, user_dict, args):
    train_user_set = user_dict["train_user_set"]
    val_user_set = user_dict["val_user_set"]

    test_users = torch.tensor(list(train_user_set.keys()), dtype=torch.long)
    device = args.device

    n_users = int(args.n_users)
    topK = 20  

    cand_chunk = int(getattr(args, "cand_chunk", 4096))   
    batch_size = int(getattr(args, "test_batch_size", 4096)) 

    model.eval()

    all_rating = np.full((n_users, topK), -1, dtype=np.int64)
    all_value  = np.full((n_users, topK), -1e10, dtype=np.float32)

    candidates_all = torch.arange(n_users, device=device, dtype=torch.long)

    def topk_merge(cur_vals, cur_inds, new_vals, new_inds, k):

        vals = torch.cat([cur_vals, new_vals], dim=1)
        inds = torch.cat([cur_inds, new_inds], dim=1)
        topv, topi = torch.topk(vals, k=k, dim=1)
        top_inds = inds.gather(1, topi)
        return topv, top_inds

    def build_exclude_pairs(batch_users_cpu):
        ex_rows = []
        ex_cols = []
        for row_i, u in enumerate(batch_users_cpu):
            tr = train_user_set.get(u, [])
            va = val_user_set.get(u, [])
            if len(tr) > 0:
                ex_rows.extend([row_i] * len(tr))
                ex_cols.extend(tr)
            if len(va) > 0:
                ex_rows.extend([row_i] * len(va))
                ex_cols.extend(va)
            ex_rows.append(row_i)
            ex_cols.append(u)
        if len(ex_rows) == 0:
            return None, None
        ex_rows = np.asarray(ex_rows, dtype=np.int64)
        ex_cols = np.asarray(ex_cols, dtype=np.int64)
        m = (ex_cols >= 0) & (ex_cols < n_users)
        return ex_rows[m], ex_cols[m]

    for bu in minibatch(test_users, batch_size=batch_size):
        bu = bu.to(device)
        B = bu.shape[0]

        bu_cpu = bu.detach().cpu().numpy().astype(np.int64)
        ex_rows_np, ex_cols_np = build_exclude_pairs(bu_cpu)

        running_vals = torch.full((B, topK), -1e30, device=device)
        running_inds = torch.full((B, topK), -1, device=device, dtype=torch.long)

        for start in range(0, n_users, cand_chunk):
            end = min(n_users, start + cand_chunk)
            cand = candidates_all[start:end]
            C = cand.shape[0]

            u_rep = bu.unsqueeze(1).expand(B, C).reshape(-1)
            v_rep = cand.unsqueeze(0).expand(B, C).reshape(-1)

            scores = _neumf_score(model, u_rep, v_rep).view(B, C)

            if ex_rows_np is not None and len(ex_rows_np) > 0:
                in_block = (ex_cols_np >= start) & (ex_cols_np < end)
                if np.any(in_block):
                    rr = torch.from_numpy(ex_rows_np[in_block]).to(device)
                    cc = torch.from_numpy(ex_cols_np[in_block] - start).to(device)  
                    scores[rr, cc] = -1e30

            k_blk = min(topK, C)
            blk_vals, blk_local = torch.topk(scores, k=k_blk, dim=1)
            blk_inds = cand[blk_local]  
            running_vals, running_inds = topk_merge(
                running_vals, running_inds, blk_vals, blk_inds, topK
            )

        all_rating[bu_cpu] = running_inds.detach().cpu().numpy().astype(np.int64)
        all_value[bu_cpu]  = running_vals.detach().cpu().numpy().astype(np.float32)

    if args.reweight_flag == 0:
        save_dir = os.path.join(args.path, "test_logs", args.dataset)
    else:
        save_dir = os.path.join(args.path, "test_logs_reweight5", args.dataset)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    prefix = os.path.join(save_dir, f"{args.model}_{args.seed}_")

    np.save(prefix + "rating_list.npy", all_rating)
    np.save(prefix + "rating_value_list.npy", all_value)
    np.save(prefix + "user_ids.npy", np.arange(n_users, dtype=np.int64))
    np.save(prefix + "Recall.npy", np.array([]))
    np.save(prefix + "Precision.npy", np.array([]))
    np.save(prefix + "F1.npy", np.array([]))
    np.save(prefix + "NDCG.npy", np.array([]))
    np.save(prefix + "Hit_ratio.npy", np.array([]))
    np.save(prefix + "rec_percentage.npy", np.array(0.0))
    np.save(prefix + "being_rec_percentage.npy", np.array(0.0))
    np.save(prefix + "system_score.npy", np.array(0.0))

    filled = int((all_rating[:, 0] != -1).sum())
    print("[test_per_user_neumf] saved prefix:", prefix)
    print("[test_per_user_neumf] rating_list shape:", all_rating.shape,
          "| filled_users:", filled, "/", n_users,
          "| eval_users:", len(test_users))
