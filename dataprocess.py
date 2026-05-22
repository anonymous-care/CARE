import pickle
import random
import pandas as pd
import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import Dataset as BaseDataset
from torch_geometric.utils import add_remaining_self_loops, degree

class Dataset(BaseDataset):
    def __init__(self, users, pos_items, neg_items, args):
        self.users = users
        self.pos_items = pos_items
        self.neg_items = neg_items
        self.args = args
        # print("dataset number:", len(self.users))

    def _get_feed_dict(self, index):

        feed_dict = {
            'users': self.users[index],
            'pos_items': self.pos_items[index],
            'neg_items': self.neg_items[index],
        }

        return feed_dict

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        return self._get_feed_dict(index)


    def collate_batch(self, feed_dicts):

        feed_dict = dict()

        feed_dict['users'] = torch.LongTensor([d['users'] for d in feed_dicts])
        feed_dict['pos_items'] = torch.LongTensor(
            [d['pos_items'] for d in feed_dicts])

        feed_dict['neg_items'] = torch.LongTensor(
            np.stack([d['neg_items'] for d in feed_dicts]))

        feed_dict['idx'] = torch.cat(
            [feed_dict['users'], feed_dict['pos_items'], feed_dict['neg_items'].view(-1)])

        return feed_dict


def get_user_map(train_data, test_data, val_data):
    all_user_ids = np.unique(np.concatenate([train_data[:, 0], train_data[:, 1],
                                             val_data[:, 0], val_data[:, 1],
                                             test_data[:, 0], test_data[:, 1]]))
    user_id_map = {old_id: new_id for new_id, old_id in enumerate(all_user_ids)}
    reverse_user_id_map = {new_id: old_id for old_id, new_id in user_id_map.items()}
    return all_user_ids, user_id_map, reverse_user_id_map

def get_user_label_map(user_id_map, label):
    user_label_map = {}
    keySet = set(label.keys())
    userSet = set(user_id_map.keys())
    for key in keySet:
        if key in userSet:
            user_label_map[user_id_map[key]] = label[key]
    return user_label_map

def get_popularity_label_map(user_id_map, label):
    user_label_map = {}
    keySet = set(label.keys())
    userSet = set(user_id_map.keys())
    for key in keySet:
        if key in userSet:
            user_label_map[user_id_map[key]] = label[key]
    return user_label_map


def process(train_data, val_data, test_data):
    all_user_ids, user_id_map, reverse_user_id_map = get_user_map(train_data, test_data, val_data)
    n_users = len(all_user_ids)
    train_data[:, 0] = np.array([user_id_map[user_id] for user_id in train_data[:, 0]])
    train_data[:, 1] = np.array([user_id_map[user_id] for user_id in train_data[:, 1]])

    val_data[:, 0] = np.array([user_id_map[user_id] for user_id in val_data[:, 0]])
    val_data[:, 1] = np.array([user_id_map[user_id] for user_id in val_data[:, 1]])

    test_data[:, 0] = np.array([user_id_map[user_id] for user_id in test_data[:, 0]])
    test_data[:, 1] = np.array([user_id_map[user_id] for user_id in test_data[:, 1]])

    train_user_set, val_user_set, test_user_set = defaultdict(list), defaultdict(list), defaultdict(list)
    for u1, u2, _ in train_data:
        train_user_set[int(u1)].append(int(u2))
        train_user_set[int(u2)].append(int(u1))
    for u1, u2, _ in val_data:
        val_user_set[int(u1)].append(int(u2))
        val_user_set[int(u2)].append(int(u1))
    for u1, u2, _ in test_data:
        test_user_set[int(u1)].append(int(u2))
        test_user_set[int(u2)].append(int(u1))
    return n_users, train_user_set, val_user_set, test_user_set

def process_adj(data_cf, n_users):
    cf = data_cf.copy()
    cf_ = cf.copy()
    cf_[:, 0], cf_[:, 1] = cf[:, 1], cf[:, 0]
    cf_ = np.concatenate([cf, cf_], axis=0)
    return torch.LongTensor(cf_).t()

def load_data(args):
    print('reading train/val/test user-item set ...')
    train_f = args.path + '/output/' + 'train/'+ 'edges.txt'
    val_f = args.path + '/output/' + 'val/'+ 'edges.txt'
    test_f = args.path + '/output/' + 'test/'+ 'edges.txt'
    file_popularity_map = args.path +"/output/userPopularityLabel.pkl"
    label_f = args.path + '/output/' + 'userActiveLabel.pkl'
    with open(train_f, 'r') as file:
        for i, line in enumerate(file, start=1):
            try:
                values = [int(value) for value in line.split()]
            except ValueError as e:
                print(f"Error at Line {i}: {e}")
                print(f"Line content: {line.strip()}")
    with open(label_f, 'rb') as f:
        label_cf = pickle.load(f)
    popularity_map_temp = np.load(file_popularity_map, allow_pickle=True)
    train_cf = np.loadtxt(open(train_f, "r"), dtype = object)
    val_cf = np.loadtxt(open(val_f, "r"), dtype = object)
    test_cf = np.loadtxt(open(test_f, "r"), dtype = object)
    train_cf = train_cf.astype(int)
    val_cf = val_cf.astype(int)
    test_cf = test_cf.astype(int)
    all_user_ids, user_id_map, reverse_user_id_map = get_user_map(train_cf, test_cf, val_cf)
    print("Train: ", train_cf.shape)
    print("Val: ", val_cf.shape)
    print("Test: ", test_cf.shape)
    label_map = get_user_label_map(user_id_map, label_cf)
    popularity_map = dict()
    popularity_map = get_popularity_label_map(user_id_map, popularity_map_temp)
    n_users, train_user_set, val_user_set, test_user_set = process(train_cf, val_cf, test_cf)

    print('building the adj mat ...')
    adj = process_adj(train_cf, n_users)

    user_dict = {
        'train_user_set': train_user_set,
        'val_user_set': val_user_set,
        'test_user_set': test_user_set
    }

    clicked_set = defaultdict(list)
    for key in user_dict:
        for user in user_dict[key]:
            clicked_set[user].extend(user_dict[key][user])

    print('Finish loading dataset', args.dataset)
    return train_cf, val_cf, test_cf, user_dict, n_users, clicked_set, adj, user_id_map, reverse_user_id_map, label_map, popularity_map


def normalize_edge(edge_index, n_users):
    row, col = edge_index
    deg = degree(col)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    return torch.sparse.FloatTensor(edge_index, edge_weight, (n_users, n_users)), deg
