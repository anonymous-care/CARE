import numpy as np
import random
import torch
import scipy.sparse as sp
from torch import nn
from torch_geometric.utils import add_remaining_self_loops, degree
from torch_scatter import scatter_max, scatter_add
import seaborn as sns
import matplotlib.pyplot as plt
import os


def neg_sample_before_epoch(train_cf, clicked_set, args):
    neg_cf = np.random.randint(
        0, args.n_users, (train_cf.shape[0], args.K))

    for i in range(train_cf.shape[0]):
        user_clicked_set = clicked_set[train_cf[i, 0]]

        for j in range(args.K):
            while (neg_cf[i, j] in user_clicked_set or neg_cf[i, j] == train_cf[i, 0]):
                neg_cf[i, j] = np.random.randint(
                    0, args.n_users)

    return neg_cf


def batch_to_gpu(batch, device):
    for c in batch:
        batch[c] = batch[c].to(device)

    return batch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def minibatch(*tensors, batch_size):
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def knn_adj(adj_sp_norm, args):
    adj_sp_norm = adj_sp_norm.to_dense()
    top_adj_sp_norm, _ = torch.topk(adj_sp_norm, args.knn)
    low, high = top_adj_sp_norm[:, -1], top_adj_sp_norm[:, 0]
    mask = ((adj_sp_norm >= low.unsqueeze(1)) *
            (adj_sp_norm <= high.unsqueeze(1))).float()
    adj_sp_norm = adj_sp_norm * mask
    adj_sp_norm = torch.triu(adj_sp_norm, diagonal=1)

    edge_index = adj_sp_norm.nonzero()
    adj_sp_norm = torch.sparse.FloatTensor(
        edge_index.t(), adj_sp_norm[edge_index[:, 0], edge_index[:, 1]],
        (args.n_users + args.n_items, args.n_users + args.n_items))

    return adj_sp_norm


def ratio(train_cf, n_users):
    user_link_num = torch.tensor(
        [(train_cf[:, 0] == i).sum() for i in range(n_users)])

    link_ratio = n_users / user_link_num
    link_ratio = link_ratio / link_ratio.sum()

    return link_ratio[train_cf[:, 0]]


def cal_bpr_loss(user_embs, pos_item_embs, neg_item_embs, link_ratios=None):
    pos_scores = torch.sum(
        torch.mul(user_embs, pos_item_embs), axis=1)

    neg_scores = torch.sum(torch.mul(user_embs.unsqueeze(
        dim=1), neg_item_embs), axis=-1)

    bpr_loss = torch.mean(
        torch.log(1 + torch.exp((neg_scores - pos_scores.unsqueeze(dim=1))).sum(dim=1)))

    return bpr_loss


def cal_bpr_loss_light_gcn(user_embs, pos_item_embs, neg_item_embs, link_ratios=None):
    if neg_item_embs.dim() == 2:
        neg_item_embs = neg_item_embs.unsqueeze(1)

    pos_scores = (user_embs * pos_item_embs).sum(dim=1)
    neg_scores = (user_embs.unsqueeze(1) * neg_item_embs).sum(dim=2)

    diff = neg_scores - pos_scores.unsqueeze(1)
    bpr_loss = torch.nn.functional.softplus(diff).mean()

    return bpr_loss

def cal_bpr_loss_neumf(model, users, pos_items, neg_items):

    pos_score = model.score_pairs(users, pos_items)

    B, K = neg_items.shape
    users_rep = users.view(B, 1).expand(B, K).reshape(-1)   
    neg_flat = neg_items.reshape(-1)                        
    neg_score = model.score_pairs(users_rep, neg_flat).view(B, K) 

    loss = F.softplus(neg_score - pos_score.view(B, 1)).mean()
    return loss


def cal_bpr_loss_reweight(weights, user_embs, pos_item_embs, neg_item_embs, link_ratios=None):
    if neg_item_embs.dim() == 2:
        neg_item_embs = neg_item_embs.unsqueeze(1)

    pos_scores = (user_embs * pos_item_embs).sum(dim=1)
    neg_scores = (user_embs.unsqueeze(1) * neg_item_embs).sum(dim=2)

    diff = neg_scores - pos_scores.unsqueeze(1)
    per_sample_loss = torch.log1p(torch.exp(diff)).sum(dim=1)

    if weights.dim() > 1:
        weights = weights.squeeze() 

    loss = torch.mean(weights * per_sample_loss)
    return loss


def cal_bpr_loss_light_gcn_reweight_right_1(pos_weights, neg_weights, user_embs, pos_item_embs, neg_item_embs):
    pos_scores = torch.sum(user_embs * pos_item_embs, dim=1)  
    neg_scores = torch.sum(user_embs.unsqueeze(dim=1) * neg_item_embs, dim=-1) 

    pos_scores_weighted = pos_weights * pos_scores.unsqueeze(dim=1)  
    neg_scores_weighted = neg_weights * neg_scores  
    bpr_loss = torch.mean(nn.functional.softplus(neg_scores_weighted - pos_scores_weighted))

    return bpr_loss


def cal_bpr_loss_light_gcn_reweight_right_2(weights, user_embs, pos_item_embs, neg_item_embs, link_ratios=None):
    pos_scores = torch.sum(torch.mul(user_embs, pos_item_embs), axis=1)
    neg_scores = torch.sum(torch.mul(user_embs.unsqueeze(dim=1), neg_item_embs), axis=-1)

    bpr_loss = torch.mean(weights.unsqueeze(dim=1) * nn.functional.softplus(neg_scores - pos_scores))

    return bpr_loss


def cal_bpr_loss_light_gcn_reweight_right_3(pos_weights, neg_weights, user_embs, pos_item_embs, neg_item_embs):
    pos_scores = torch.sum(torch.mul(user_embs, pos_item_embs), axis=1)

    neg_scores = torch.sum(torch.mul(user_embs.unsqueeze(dim=1), neg_item_embs), axis=-1)

    softplus_result = nn.functional.softplus(neg_scores - pos_scores.unsqueeze(dim=1))

    weighted_softplus = pos_weights.unsqueeze(dim=1) * neg_weights * softplus_result

    bpr_loss = torch.mean(weighted_softplus)

    return bpr_loss



def cal_bpr_loss_light_gcn_reweight(weights, user_embs, pos_item_embs, neg_item_embs, link_ratios=None):
    if neg_item_embs.dim() == 2:
        neg_item_embs = neg_item_embs.unsqueeze(1)

    pos_scores = (user_embs * pos_item_embs).sum(dim=1)
    neg_scores = (user_embs.unsqueeze(1) * neg_item_embs).sum(dim=2)

    diff = neg_scores - pos_scores.unsqueeze(1)
    loss_mat = torch.nn.functional.softplus(diff)

    if weights.dim() > 1:
        weights = weights.squeeze()

    weighted = loss_mat * weights.unsqueeze(1)

    return weighted.sum() / (weights.sum() * loss_mat.size(1))


def cal_l2_loss(user_embs, pos_item_embs, neg_item_embs, batch_size):
    return 0.5 * (user_embs.norm(2).pow(2) + pos_item_embs.norm(2).pow(2) + neg_item_embs.norm(2).pow(2)) / batch_size




def softmax(src, index, num_nodes):
    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()

    out = out / (
            scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)

    return out


def plot_training_metrics(losses, rec_percentage, being_rec_percentage, system_score, rec_nums, top_k, save_path=None):
    epochs = range(len(losses[top_k]))

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.plot(epochs, losses[top_k], label="Loss", color='blue')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(epochs, rec_percentage[top_k], label="Recommendation %", color='red')
    plt.xlabel("Epochs")
    plt.ylabel("Recommendation %")
    plt.title("Recommendation Percentage Over Time")
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(epochs, being_rec_percentage[top_k], label="Being Recommended %", color='green')
    plt.xlabel("Epochs")
    plt.ylabel("Being Recommended %")
    plt.title("Being Recommended Percentage Over Time")
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.plot(epochs, system_score[top_k], label="System Score", color='purple')
    plt.xlabel("Epochs")
    plt.ylabel("System Score")
    plt.title("System Score Over Time")
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(epochs, rec_nums[top_k], label="Rec Nums (Inactive)", color='orange')
    plt.xlabel("Epochs")
    plt.ylabel("Rec Nums Inactive")
    plt.title("Inactive Users' Recommendations Over Time")
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()
