import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
import time
import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from matplotlib.lines import Line2D
from pathlib import Path
import csv
import json


def plot_four_panel(
    x_values,
    results_total,
    metrics,               
    ylabels,                
    filename,
    xlabel="Lambda",
    flags=("no_reweight", "reweight"),
    flag_styles=None,
    panel_tags=(" (a)", " (b)", " (c)", " (d)"),
    figsize=(10.5, 6.6),
):

    assert len(metrics) == 4 and len(ylabels) == 4, "Need exactly 4 metrics and 4 ylabels."

    if flag_styles is None:
        flag_styles = {"no_reweight": "-", "reweight": "--"}

    ratios = sorted(results_total[flags[0]][metrics[0]].keys())

    cmap = plt.colormaps.get_cmap("tab10")
    n = len(ratios)
    ratio_colors = {r: cmap(i / max(1, n - 1)) for i, r in enumerate(ratios)}

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    axes = axes.ravel()

    for i, (ax, metric, ylabel) in enumerate(zip(axes, metrics, ylabels)):
        for flag in flags:
            ls = flag_styles[flag]
            for ratio in ratios:
                y = results_total[flag][metric][ratio]
                ax.plot(
                    x_values, y,
                    color=ratio_colors[ratio],
                    linestyle=ls,
                    marker="o",
                    markersize=3.5,
                    linewidth=2.0,
                    alpha=0.95,
                )

        ax.set_ylabel(ylabel, fontsize=13)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", labelsize=11)

        if panel_tags is not None:
            ax.text(
                0.02, -0.20,
                panel_tags[i],
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=12
            )

    axes[2].set_xlabel(xlabel, fontsize=13)
    axes[3].set_xlabel(xlabel, fontsize=13)

    ratio_handles = [
        Line2D([0], [0],
               color=ratio_colors[r], lw=2, marker='o', markersize=4,
               label=f"r={r}")
        for r in ratios
    ]
    flag_handles = [
        Line2D([0], [0],
               color="black", lw=2, linestyle=flag_styles[f],
               label=f"{f}")
        for f in flags
    ]

    spacer = Line2D([0], [0], color="none", label="")  
    handles = ratio_handles + [spacer] + flag_handles

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(ratio_handles) + 1 + len(flag_handles),
        frameon=False,
        fontsize=11,
        handlelength=2.2,
        columnspacing=1.2,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    plt.show()
    print(f"Saved combined plot: {filename}")

def save_plot_data(x_values, results_total, metric, outdir="logs"):
    outdir = Path(outdir)

    if outdir.exists() and not outdir.is_dir():
        raise RuntimeError(f"[save_plot_data] outdir='{outdir}' exists but is not a directory.")

    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / f"{metric}_20.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["flag", "ratio", "lambda", "metric", "value"])
        for flag in ["no_reweight", "reweight"]:
            for ratio, y_list in results_total[flag][metric].items():
                for lam, val in zip(x_values, y_list):
                    writer.writerow([flag, ratio, lam, metric, float(val)])

    for flag in ["no_reweight", "reweight"]:
        for ratio, y_list in results_total[flag][metric].items():
            arr = np.asarray(y_list, dtype=float)
            np.save(outdir / f"{metric}__{flag}__ratio{ratio}_20.npy", arr)

    print(f"[LOG] Saved {metric} data to {csv_path} and per-series NPYs.")

def save_all_results_blob(lambda_values, results_total, outdir="logs"):

    Path(outdir).mkdir(parents=True, exist_ok=True)

    json_safe = {}
    for flag, flag_dict in results_total.items():
        json_safe[flag] = {}
        for metric, metric_dict in flag_dict.items():
            json_safe[flag][metric] = {}
            for ratio, y_list in metric_dict.items():
                json_safe[flag][metric][str(ratio)] = [float(v) for v in y_list]
    with open(Path(outdir) / "results_total_20.json", "w") as f:
        json.dump({
            "lambda_values": list(lambda_values),
            "results_total": json_safe
        }, f)
    print(f"[LOG] Saved JSON snapshot -> {Path(outdir) / 'results_total.json'}")

    flat = {}
    for flag, flag_dict in results_total.items():
        for metric, metric_dict in flag_dict.items():
            for ratio, y_list in metric_dict.items():
                key = f"{metric}__{flag}__ratio{ratio}"
                flat[key] = np.asarray(y_list, dtype=float)
    flat["lambda_values"] = np.asarray(lambda_values, dtype=float)
    np.savez_compressed(Path(outdir) / "results_total_20.npz", **flat)
    print(f"[LOG] Saved NPZ snapshot  -> {Path(outdir) / 'results_total.npz'}")


def plot_results(x_values, results_total, metric,
                 x_label, y_label, filename):

    # currently don't use

    flag_styles = {
        'no_reweight': '-',
        'reweight': '--',
    }

    ratios = sorted(results_total['no_reweight'][metric].keys())

    cmap = plt.colormaps.get_cmap("tab10")
    n = len(ratios)
    ratio_colors = {
        r: cmap(i / max(1, n - 1))  
        for i, r in enumerate(ratios)
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for flag, ls in flag_styles.items():
        for ratio in ratios:
            y = results_total[flag][metric][ratio]
            ax.plot(
                x_values, y,
                color=ratio_colors[ratio],
                linestyle=ls,
                marker='o',
                markersize=4,
                linewidth=2,
                alpha=0.95
            )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f'{y_label} vs {x_label}')
    ax.grid(True, alpha=0.3)

    ratio_handles = [
        Line2D([0], [0],
               color=ratio_colors[r], lw=2, marker='o', markersize=4,
               label=f'ratio={r}')
        for r in ratios
    ]
    flag_handles = [
        Line2D([0], [0],
               color='black', lw=2, linestyle=flag_styles[f],
               label=f)
        for f in flag_styles
    ]

    leg1 = ax.legend(
        handles=ratio_handles,
        title='Ratio (color)',
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        fontsize=9,
        title_fontsize=9
    )
    ax.add_artist(leg1)

    ax.legend(
        handles=flag_handles,
        title='Setting (linestyle)',
        loc='upper left',
        bbox_to_anchor=(1.02, 0.55),
        frameon=False,
        fontsize=9,
        title_fontsize=9
    )

    fig.tight_layout()
    fig.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()
    print(f"Saved plot: {filename}")

def load_result(method_name, seed, topk_index, flag='reweight'):
    if flag == "reweight":
        prefix = "./test_logs_reweight5/Twitter/"
    # elif flag == "reweight2":
    #     prefix = "./test_logs_reweight5/Twitter/"
    else:
        prefix = "./test_logs/Twitter/"
    rating_filename = prefix + method_name + "_" + str(seed) + "_rating_list.npy"
    rating_value_filename = prefix + method_name + "_" + str(seed) + "_rating_value_list.npy"
    high_rating_items = np.load(rating_filename, allow_pickle=True)
    ratings = np.load(rating_value_filename, allow_pickle=True)

    file_f1 = prefix + method_name + "_" + str(seed) + "_F1.npy"
    file_NDCG = prefix + method_name + "_" + str(seed) + "_NDCG.npy"
    file_hit_ratio = prefix + method_name + "_" + str(seed) + "_Hit_ratio.npy"
    file_recall = prefix + method_name + "_" + str(seed) + "_Recall.npy"
    file_precision = prefix + method_name + "_" + str(seed) + "_Precision.npy"
    file_rec_percentage = prefix + method_name + "_" + str(seed) + "_rec_percentage.npy"
    file_being_rec_percentage = prefix + method_name + "_" + str(seed) + "_being_rec_percentage.npy"
    file_system_score = prefix + method_name + "_" + str(seed) + "_system_score.npy"
    file_user = prefix + method_name + "_" + str(seed) + "_user_ids.npy"
    f1 = np.load(file_f1, allow_pickle=True)
    ndcg = np.load(file_NDCG, allow_pickle=True)
    hit_ratio = np.load(file_hit_ratio, allow_pickle=True)
    recall = np.load(file_recall, allow_pickle=True)
    precision = np.load(file_precision, allow_pickle=True)
    rec_percentage = np.load(file_rec_percentage, allow_pickle=True)
    being_rec_percentage = np.load(file_being_rec_percentage, allow_pickle=True)
    system_score = np.load(file_system_score, allow_pickle=True)
    results = dict()
    results['F1'] = f1
    results['NDCG'] = ndcg
    results['Hit_ratio'] = hit_ratio
    results['Recall'] = recall
    results['Precision'] = precision
    results['Rec_percentage'] = rec_percentage
    results['Being_rec_percentage'] = being_rec_percentage
    results['System_score'] = system_score

    rec_user_list = np.load(file_user, allow_pickle=True)
    return high_rating_items, ratings, results, rec_user_list


def get_user_map(train_data, test_data, val_data):
    all_user_ids = np.unique(np.concatenate([train_data[:, 0], train_data[:, 1],
                                             val_data[:, 0], val_data[:, 1],
                                             test_data[:, 0], test_data[:, 1]]))
    user_id_map = {old_id: new_id for new_id, old_id in enumerate(all_user_ids)}
    reverse_user_id_map = {new_id: old_id for old_id, new_id in user_id_map.items()}
    return all_user_ids, user_id_map, reverse_user_id_map


def get_user_map_train(train_data):
    all_user_ids = np.unique(np.concatenate([train_data[:, 0], train_data[:, 1]]))
    user_id_map = {old_id: new_id for new_id, old_id in enumerate(all_user_ids)}
    reverse_user_id_map = {new_id: old_id for old_id, new_id in user_id_map.items()}
    return all_user_ids, user_id_map, reverse_user_id_map


def obtain_user_dict(seed):
    train_f = "./output/train" + "/edges.txt"
    val_f = "./output/val" + "/edges.txt"
    test_f = "./output/test" + "/edges.txt"
    train_cf = np.loadtxt(open(train_f, "r"))
    val_cf = np.loadtxt(open(val_f, "r"))
    test_cf = np.loadtxt(open(test_f, "r"))
    train_cf = train_cf.astype(int)
    val_cf = val_cf.astype(int)
    test_cf = test_cf.astype(int)
    all_user_ids, user_id_map, reverse_user_id_map = get_user_map(train_cf, test_cf, val_cf)
    train_cf[:, 0] = np.array([user_id_map[user_id] for user_id in train_cf[:, 0]])
    train_cf[:, 1] = np.array([user_id_map[user_id] for user_id in train_cf[:, 1]])

    val_cf[:, 0] = np.array([user_id_map[user_id] for user_id in val_cf[:, 0]])
    val_cf[:, 1] = np.array([user_id_map[user_id] for user_id in val_cf[:, 1]])

    test_cf[:, 0] = np.array([user_id_map[user_id] for user_id in test_cf[:, 0]])
    test_cf[:, 1] = np.array([user_id_map[user_id] for user_id in test_cf[:, 1]])

    n_users = len(all_user_ids)

    train_user_set, val_user_set, test_user_set = defaultdict(
        list), defaultdict(list), defaultdict(list)
    for u1, u2, _ in train_cf:
        train_user_set[int(u1)].append(int(u2))
        train_user_set[int(u2)].append(int(u1))
    for u1, u2, _ in val_cf:
        val_user_set[int(u1)].append(int(u2))
        val_user_set[int(u2)].append(int(u1))
    for u1, u2, _ in test_cf:
        test_user_set[int(u1)].append(int(u2))
        test_user_set[int(u2)].append(int(u1))

    user_dict = {
        'train_user_set': train_user_set,
        'val_user_set': val_user_set,
        'test_user_set': test_user_set,
    }
    return user_dict, n_users


def Hit_at_k(r, k):
    right_pred = r[:, :k].sum(axis=1)

    return 1. * (right_pred > 0)


def RecallPrecision_ATk(test_data, r, k):

    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = right_pred / recall_n
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


def getLabel(test_data, pred_data):
    r = []

    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)

    return np.array(r).astype('float')

# not use now
def calculate_recommendation_percentages(user_id, test_data, pred_data):

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


def calculate_all_users_percentages(test_data, pred_data):

    recommendation_percentages = []
    being_recommended_percentages = []

    for user_id in range(len(test_data)):
        result = calculate_recommendation_percentages(user_id, test_data, pred_data)

        if result[0] is not None:
            rec_percentage, being_rec_percentage = result
            recommendation_percentages.append(rec_percentage)
            being_recommended_percentages.append(being_rec_percentage)

    return recommendation_percentages, being_recommended_percentages


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



def calculate_percentages_avg(rec, being_rec, weight_1=1 / 2, weight_2=1 / 2):
    res = weight_1 * rec + weight_2 * being_rec
    return res


def minibatch(*tensors, batch_size):
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def train_activeness_distribution_from_train_edges(file_train_edges, label_map):
    train_edges = np.loadtxt(open(file_train_edges, "r")).astype(int)
    all_user_ids, user_id_map, reverse_user_id_map = get_user_map_train(train_edges)
    train_edges[:, 0] = np.array([user_id_map[user_id] for user_id in train_edges[:, 0]])
    train_edges[:, 1] = np.array([user_id_map[user_id] for user_id in train_edges[:, 1]])
    edgelist = defaultdict(list)
    for u, v, _ in train_edges:
        edgelist[int(u)].append(int(v))

    inactive_ratio = dict()
    for u in edgelist.keys():
        v_list = edgelist[u]
        v_label = [label_map.get(t, 1) for t in v_list]

        # print(len(all_user_ids))
        # print(len(label_map.keys()))
        inactiveness = [int(t == 1) for t in v_label]  # opposite gender: 1, same gender: 0
        inactive_ratio[u] = np.mean(inactiveness)
    return inactive_ratio


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

def recommendation_activeness_distribution(recommendation_list, label_map, topk=50):
    inactive_ratio = dict()
    for u in range(len(recommendation_list)):
        v_list = recommendation_list[u][:topk]
        v_activeness = [label_map.get(t, 1) for t in v_list]
        inactives = [int(t == 1) for t in v_activeness]
        inactive_ratio[u] = np.mean(inactives)
    return inactive_ratio


def inconsistency_per_seed(seed, recommendation_list, user_id_map, label_map):
    file_edges = "./output/train/edges.txt"
    file_label_map = "./output/userActiveLabel.pkl"
    file_inactive_ratio = "./output/activeRatio.txt"

    label_map_temp = np.load(file_label_map, allow_pickle=True)

    label_map = dict()
    for k in label_map_temp.keys():
        label_map[k] = label_map_temp.get(k)
    label_map = get_user_label_map(user_id_map, label_map)
    train_dict = train_activeness_distribution_from_train_edges(file_edges, label_map)

    recommendation_dict = recommendation_activeness_distribution(recommendation_list, label_map)

    groups = split_groups_horizontally(file_inactive_ratio, user_id_map, group_num=3)
    group_inconsistency = []
    for i, g in enumerate(groups):
        t = []
        for uid in g:
            rec = recommendation_dict.get(uid, -1)
            train = train_dict.get(uid, -1)
            difference = abs(rec - train)
            t.append(difference)
        group_inconsistency.append(np.mean(t))
    return group_inconsistency


def absolute(u, v):
    return abs(u - v)


def selection_criterion(u_ratio, existing_activeness, candidate_users, candidate_activeness, candidate_rating, lambda_):
    if len(existing_activeness) == 0:
        return 0

    inactive_num = np.sum(np.array([u == 1 for u in existing_activeness]).astype(int))

    max_score = -10000
    max_index = 0
    for i in range(len(candidate_users)):
        c_activeness = candidate_activeness[i]
        if c_activeness == 1:
            v_ratio = (1 + inactive_num) / (len(existing_activeness) + 1)
        else:
            v_ratio = inactive_num / (len(existing_activeness) + 1)

        f = absolute(u_ratio, v_ratio)
        r = candidate_rating[i]
        score = (1 - lambda_) * r - (lambda_) * f

        if score > max_score:
            max_score = score
            max_index = i

        if 1 in candidate_activeness[0:i + 1] and 2 in candidate_activeness[0: i + 1]:
            break
    return max_index


class rerank():
    def __init__(self, method_name, ratio, seed, flag):
        self.user_dict, self.n_user = obtain_user_dict(seed=seed)
        self.seed = seed
        self.ratio = float(ratio)

        file_train_edges = "./output/train" + "/edges.txt"
        file_val_edges = "./output/val" + "/edges.txt"
        file_test_edges = "./output/test" + "/edges.txt"
        file_user_inactive_ratio = "./output/activeRatio.txt"
        file_label_map = "./output/userActiveLabel.pkl"
        file_popularity_map = "./output/userPopularityLabel.pkl"
        train_edges = np.loadtxt(open(file_train_edges, "r")).astype(int)
        all_user_ids, user_id_map, reverse_user_id_map = get_user_map_train(train_edges)
        label_map_temp = np.load(file_label_map, allow_pickle=True)
        popularity_map_temp = np.load(file_popularity_map, allow_pickle=True)
        self.user_id_map = user_id_map
        label_map = dict()
        label_map = get_user_label_map(user_id_map, label_map_temp)
        popularity_map = dict()
        popularity_map = get_popularity_label_map(user_id_map, popularity_map_temp)
        self.label_map = label_map
        self.popularity_map = popularity_map

        self.recommendation_list, self.recommendation_ratings, results, rec_user_list = load_result(
            method_name=method_name, seed=seed, topk_index=3, flag=flag)
        self.train_dict = train_activeness_distribution_from_train_edges(file_train_edges, self.label_map)

    def rerank(self):
        results_selective = dict()
        results_selective_val = dict()
        items_lists_selective = dict()  
        inconsistencies = dict()
        for t in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            r, u, list_ = self.evaluate_recommendation_selective(lambda_=t, flag="test")
            results_selective[t] = r
            items_lists_selective[t] = list_
            inconsistencies[t] = inconsistency_per_seed(self.seed, list_, self.user_id_map, self.label_map)
            print(str(t) + ":", "Inconsistency =", np.array(inconsistencies[t]))

        return results_selective, inconsistencies

    # utils
    def rerank_selective(self, topk, lambda_, users):

        train_dict = self.train_dict
        label_map = self.label_map
        rec_items = self.recommendation_list
        rec_ratings = self.recommendation_ratings

        recommendation_rerank = []
        for uid in users:
            uid = uid.item()
            if uid in self.label_map.keys() and self.label_map[uid] == 1:
                recommendation_rerank.append(rec_items[uid][:topk])
                continue

            inactive_ratio_train = self.ratio
            if uid >= len(rec_items):
                recommendation_rerank.append([-1] * topk)
                continue
            rec_items_t = rec_items[uid][:topk]
            rec_activeness = [label_map.get(u, 1) for u in rec_items_t]
            inactive_ratio_rec = np.mean([t == 1 for t in rec_activeness])

            users_activeness = []
            candidates = list(rec_items[uid])
            candidates_activeness = [label_map.get(u, 1) for u in candidates]
            u_ratings = list(rec_ratings[uid])
            max_r = max(u_ratings)
            min_r = min(u_ratings)
            u_ratings = [(t - min_r) / (max_r - min_r) for t in u_ratings]

            u_recommendation_rerank = []
            for _ in range(topk):
                selected_user = selection_criterion(inactive_ratio_train, users_activeness, candidate_users=candidates, \
                                                    candidate_activeness=candidates_activeness,
                                                    candidate_rating=u_ratings,
                                                    lambda_=lambda_)
                u_recommendation_rerank.append(candidates[selected_user])
                users_activeness.append(candidates_activeness[selected_user])
                del candidates[selected_user]
                del candidates_activeness[selected_user]
                del u_ratings[selected_user]
            recommendation_rerank.append(u_recommendation_rerank)
        return np.array(recommendation_rerank)

    def evaluate_recommendation_selective(self, lambda_, flag='test'):
        train_dict = self.train_dict
        user_dict = self.user_dict
        id_label_map = self.label_map
        popularity_map= self.popularity_map
        n_users = self.n_user

        train_user_set = user_dict['train_user_set']
        val_user_set = user_dict['val_user_set']

        if flag == "test":
            test_user_set = user_dict['test_user_set']
            test_users = torch.tensor(list(train_user_set.keys()))
        elif flag == "val":
            test_user_set = user_dict['val_user_set']
            test_users = torch.tensor(list(train_user_set.keys()))

        recalls_ = []
        precisions_ = []
        f1s_ = []
        ndcgs_ = []
        hits_ = []

        ratings = 0
        grounds = 0

        with torch.no_grad():
            users_list = []
            ratings_list = []
            ratings_values_list = []
            groundTruth_items_list = []

            for batch_users in minibatch(test_users, batch_size=4096):
                if flag == "val":
                    clicked_items = [train_user_set.get(user.item(), []) for user in batch_users]
                elif flag == "test":
                    clicked_items = []
                    for user in batch_users:
                        clicked_items_for_user_in_train = train_user_set[user.item()]
                        clicked_items_for_user_in_val = val_user_set[user.item()]
                        clicked_items.append(
                            np.concatenate([clicked_items_for_user_in_train, clicked_items_for_user_in_val]))

                groundTruth_items = [test_user_set.get(user.item(), []) for user in batch_users]

                exclude_index = []
                exclude_items = []

                for range_i, items in enumerate(clicked_items):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)

                rating_K = self.rerank_selective(lambda_=lambda_, topk=50, users=batch_users)

                users_list.append(batch_users)
                ratings_list.append(rating_K)
                groundTruth_items_list.append(groundTruth_items)
                ratings += len(rating_K)
                grounds += len(groundTruth_items)

            X = zip(ratings_list, groundTruth_items_list, users_list)

            for i, (sorted_users, groundTrue, user_tensor) in enumerate(X):
                sorted_items = sorted_users
                r = getLabel(groundTrue, sorted_items)

                for iter_ in range(len(groundTrue)):
                    pre, recall, ndcg, hit_ratio, F1, rec_percentage, being_rec_percentage, system_score = [], [], [], [], [], [], [], []
                    for k in [20]: # k
                        ret = RecallPrecision_ATk([groundTrue[iter_]], r[iter_].reshape(1, -1), k)
                        ndcgs = NDCGatK_r([groundTrue[iter_]], r[iter_].reshape(1, -1), k)
                        hit_ratios = Hit_at_k(r[iter_].reshape(1, -1), k)

                        hit_ratio.append(sum(hit_ratios))
                        pre.append(sum(ret['Precision']))
                        recall.append(sum(ret['Recall']))
                        ndcg.append(sum(ndcgs))

                        temp = ret['Precision'] + ret['Recall']
                        temp[temp == 0] = float('inf')
                        F1s = 2 * ret['Precision'] * ret['Recall'] / temp

                        F1.append(sum(F1s))


                    recalls_.append(np.mean(recall))
                    precisions_.append(np.mean(pre))
                    f1s_.append(np.mean(F1))
                    ndcgs_.append(np.mean(ndcg))
                    hits_.append(np.mean(hit_ratio))

        rec_stats = calculate_recommendation_percentages_all_users(test_user_set, ratings_list, id_label_map, popularity_map)


        results = dict()
        results['F1'] = np.array(f1s_).mean(axis=0)
        results['NDCG'] = np.array(ndcgs_).mean(axis=0)
        results['Hit_ratio'] = np.array(hits_).mean(axis=0)
        results['Recall'] = np.array(recalls_).mean(axis=0)
        results['Precision'] = np.array(precisions_).mean(axis=0)
        results.update(rec_stats)
        print("Rec Percentage =", results['Rec_percentage'])
        print("Being Rec Percentage =", results['Being_rec_percentage'])
        print("System Score =", results["System_score"])
        print("Rec Nums =",     results["Rec_Nums_Inactive"])
        print("In Rec Percentage =", results['In_Rec_percentage'])
        print("In Being Rec Percentage =", results['In_Being_rec_percentage'])
        print("In System Score =", results["In_System_score"])
        print("Active Rec Percentage =", results['Active_Rec_percentage'])
        print("Active Being Rec Percentage =", results['Active_Being_rec_percentage'])
        print("Active System Score =", results["Active_System_score"])
        print("Active Popular Rec Percentage =", results.get("Active_Pop_Rec_percentage", "N/A"))
        print("Active Popular Being Rec Percentage =", results.get("Active_Pop_Being_rec_percentage", "N/A"))
        print("Active Popular System Score =", results.get("Active_Pop_System_score", "N/A"))
        print("Active Normal Rec Percentage =", results.get("Active_Nor_Rec_percentage", "N/A"))
        print("Active Normal Being Rec Percentage =", results.get("Active_Nor_Being_rec_percentage", "N/A"))
        print("Active Normal System Score =", results.get("Active_Nor_System_score", "N/A"))
        print("Inactive Popular Rec Percentage =", results.get("Inactive_Pop_Rec_percentage", "N/A"))
        print("Inactive Popular Being Rec Percentage =", results.get("Inactive_Pop_Being_rec_percentage", "N/A"))
        print("Inactive Popular System Score =", results.get("Inactive_Pop_System_score", "N/A"))
        print("Inactive Normal Rec Percentage =", results.get("Inactive_Nor_Rec_percentage", "N/A"))
        print("Inactive Normal Being Rec Percentage =", results.get("Inactive_Nor_Being_rec_percentage", "N/A"))
        print("Inactive Normal System Score =", results.get("Inactive_Nor_System_score", "N/A"))

        ratings_list = np.concatenate(ratings_list, axis=0)
        return results, users_list, np.array(ratings_list)


def worker_rerank(task):
    method_name, ratio, seed, flag, lambda_values = task
    rnk = rerank(method_name=method_name, ratio=ratio, seed=seed, flag=flag)
    user_performance, _ = rnk.rerank()  # user_performance 为 dict, 键为 lambda 值，每项是指标字典
    return (ratio, flag, user_performance)



def main(method_name="LightGCN", ratios=(0.2, 0.3, 0.5, 0.6), num_workers=4, out_fig="four_panel_main_metrics.png"):
    lambda_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    metrics = [
        "Rec_percentage",
        "Being_rec_percentage",
        "In_Being_rec_percentage",
        "Rec_Nums_Inactive",
    ]

    flags = ["no_reweight", "reweight"]

    results_total = {
        flag: {metric: {ratio: [] for ratio in ratios} for metric in metrics}
        for flag in flags
    }

    tasks = []
    for ratio in ratios:
        for flag in flags:
            tasks.append((method_name, ratio, 1, flag, lambda_values))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker_rerank, task) for task in tasks]
        for future in as_completed(futures):
            ratio, flag, user_performance = future.result()

            for lam in lambda_values:
                for metric in metrics:
                    results_total[flag][metric][ratio].append(user_performance[lam][metric])

    plot_four_panel(
        lambda_values,
        results_total,
        metrics=metrics,
        ylabels=[
            "Rec-Acc@20 (All)",
            "Being-Acc@20 (All)",
            "Being-Acc@20 (At-risk)",
            "Exposure-Times (At-risk)",
        ],
        filename=out_fig,
        xlabel=r"$\lambda$",
        panel_tags=None,  
    )

    save_all_results_blob(lambda_values, results_total, outdir="logs/"+ method_name) 



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate rerank results")
    parser.add_argument("--option", help="Optional ratio", type=float, default=0.5)
    parser.add_argument("--seed", help="Optional seed", type=int, default=1)
    parser.add_argument("--num_workers", help="Number of workers", type=int, default=4)
    args = parser.parse_args()
    # main(method_name='LightGCN', ratios=[0.2, 0.3, 0.5, 0.6], num_workers=args.num_workers)
    # main(method_name="LightGCN", ratios=(0.2,0.3,0.5,0.6), num_workers=4, out_fig="four_panel_main_metrics.png")
    # main(method_name="MF", ratios=(0.2,0.3,0.5,0.6), num_workers=4, out_fig="four_panel_main_metrics_MF.png")
    main(method_name="NeuMF", ratios=(0.2,0.3,0.5,0.6), num_workers=4, out_fig="four_panel_main_metrics_NeuMF.png")
# main(method_name='LightGCN', ratios=[0.2, 0.3, 0.5, 0.6], flag='no_reweight')
# main(method_name='LightGCN', ratios=[0.2, 0.3, 0.5, 0.6], flag='reweight')
