import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from evaluation import (
    getLabel, RecallPrecision_ATk, NDCGatK_r, Hit_at_k,
    calculate_recommendation_percentages_all_users
)
from rerank_parrel import (  
    obtain_user_dict,                  
    get_user_map_train,
    get_user_label_map,
    get_popularity_label_map,
)

import pickle


def load_logs(base_dir: str, method: str, seed: int):
    prefix = os.path.join(base_dir, f"{method}_{seed}_")
    rating_list = np.load(prefix + "rating_list.npy", allow_pickle=True).astype(np.int64)
    rating_val  = np.load(prefix + "rating_value_list.npy", allow_pickle=True).astype(np.float32)
    user_ids    = np.load(prefix + "user_ids.npy", allow_pickle=True).astype(np.int64)
    return rating_list, rating_val, user_ids


def save_logs(base_dir: str, method: str, seed: int, rating_list, rating_val, user_ids):
    os.makedirs(base_dir, exist_ok=True)
    prefix = os.path.join(base_dir, f"{method}_{seed}_")
    np.save(prefix + "rating_list.npy", rating_list)
    np.save(prefix + "rating_value_list.npy", rating_val)
    np.save(prefix + "user_ids.npy", user_ids)


def compute_train_degree(user_dict, n_users: int):
    train_user_set = user_dict["train_user_set"]
    deg = np.zeros(n_users, dtype=np.int64)
    for u, nbrs in train_user_set.items():
        if 0 <= int(u) < n_users:
            deg[int(u)] = len(nbrs)
    return deg


def degree_debias_rerank(rating_list, rating_val, deg, alpha: float):
    """
    rating_list: [n_users, topK] candidate ids
    rating_val : [n_users, topK] raw scores
    deg       : [n_users]
    return reranked list + adjusted score
    """
    n_users, topK = rating_list.shape
    penalty = alpha * np.log(deg.astype(np.float32) + 1.0)  
    cand = rating_list
    cand_pen = penalty[cand]  

    # adjusted score
    adj = rating_val - cand_pen

    order = np.argsort(-adj, axis=1)  
    new_list = np.take_along_axis(cand, order, axis=1)
    new_val  = np.take_along_axis(adj,  order, axis=1)
    return new_list, new_val


def load_label_and_popularity_maps(output_dir: str, user_id_map: dict):
    """
    adjust uid
    """
    label_pkl = os.path.join(output_dir, "userActiveLabel.pkl")
    pop_pkl   = os.path.join(output_dir, "userPopularityLabel.pkl")

    label_map_temp = {}
    popularity_map_temp = {}

    if os.path.exists(label_pkl):
        with open(label_pkl, "rb") as f:
            label_map_temp = pickle.load(f)
    if os.path.exists(pop_pkl):
        with open(pop_pkl, "rb") as f:
            popularity_map_temp = pickle.load(f)

    label_map = get_user_label_map(user_id_map, label_map_temp) if len(label_map_temp) else {}
    pop_map   = get_popularity_label_map(user_id_map, popularity_map_temp) if len(popularity_map_temp) else {}
    return label_map, pop_map


def eval_at_20(pred_mat_20, user_dict, label_map, pop_map, flag="test"):
    """
    top_K = 20
    """
    train_user_set = user_dict["train_user_set"]
    gt_set = user_dict["test_user_set"] if flag == "test" else user_dict["val_user_set"]

    eval_users = np.array(list(train_user_set.keys()), dtype=np.int64)
    groundTruth = [gt_set.get(int(u), []) for u in eval_users]

    r = getLabel(groundTruth, pred_mat_20)
    k = 20
    ret = RecallPrecision_ATk(groundTruth, r, k)
    ndcgs = NDCGatK_r(groundTruth, r, k)
    hit_ratios = Hit_at_k(r, k)

    pre = ret["Precision"]
    rec = ret["Recall"]
    tmp = pre + rec
    tmp[tmp == 0] = float("inf")
    f1 = 2 * pre * rec / tmp

    results = {
        "Precision@20": float(np.mean(pre)),
        "Recall@20": float(np.mean(rec)),
        "NDCG@20": float(np.mean(ndcgs)),
        "Hit@20": float(np.mean(hit_ratios)),
        "F1@20": float(np.mean(f1)),
    }

    rec_stats = calculate_recommendation_percentages_all_users(
        gt_set,
        [pred_mat_20],      
        label_map,
        pop_map
    )
    results.update(rec_stats)
    return results


def plot_four_panel_eval(alphas, results_by_flag, out_png):

    metrics = [
        ("Rec_percentage", "Rec-Acc@20 (All)"),
        ("Being_rec_percentage", "Being-Acc@20 (All)"),
        ("In_Being_rec_percentage", "Being-Acc@20 (At-risk)"),
        ("Rec_Nums_Inactive", "Exposure-Times (At-risk)"),
    ]
    flags = ["no_reweight", "reweight"]
    flag_styles = {"no_reweight": "-", "reweight": "--"}

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6), sharex=True)
    axes = axes.ravel()

    for i, (m, ylabel) in enumerate(metrics):
        ax = axes[i]
        for flag in flags:
            y = results_by_flag[flag][m]
            ax.plot(alphas, y, linestyle=flag_styles[flag], marker="o", linewidth=2.0)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", labelsize=11)

    axes[2].set_xlabel(r"$\alpha$", fontsize=13)
    axes[3].set_xlabel(r"$\alpha$", fontsize=13)

    # legend
    handles = [
        Line2D([0], [0], color="black", lw=2, linestyle=flag_styles["no_reweight"], label="no_reweight"),
        Line2D([0], [0], color="black", lw=2, linestyle=flag_styles["reweight"], label="reweight"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=2, frameon=False, fontsize=11)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("[OK] figure ->", out_png)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Twitter")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--method", type=str, default="NeuMF") 
    parser.add_argument("--eval_flag", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--alphas", type=str, default="0,0.1,0.2,0.3,0.5,0.8,1.0")
    parser.add_argument("--base_dir", type=str, default=".", help="project root (contains test_logs/ output/)")
    args = parser.parse_args()

    root = os.path.abspath(args.base_dir)
    output_dir = os.path.join(root, "output")

    user_dict, n_users = obtain_user_dict(seed=args.seed)

    train_edges = np.loadtxt(open(os.path.join(output_dir, "train", "edges.txt"), "r")).astype(int)
    _, user_id_map, _ = get_user_map_train(train_edges)

    label_map, pop_map = load_label_and_popularity_maps(output_dir, user_id_map)

    deg = compute_train_degree(user_dict, n_users)

    alphas = [float(x) for x in args.alphas.split(",")]

    out_dir = os.path.join(root, "eval_degreefair", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"eval_degreefair_20_{args.method}_seed{args.seed}.csv")
    out_png = os.path.join(out_dir, f"degreefair_4panel_20_{args.method}_seed{args.seed}.png")

    results_by_flag = {
        "no_reweight": {
            "Rec_percentage": [],
            "Being_rec_percentage": [],
            "In_Being_rec_percentage": [],
            "Rec_Nums_Inactive": [],
        },
        "reweight": {
            "Rec_percentage": [],
            "Being_rec_percentage": [],
            "In_Being_rec_percentage": [],
            "Rec_Nums_Inactive": [],
        }
    }

    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "method", "flag", "seed", "alpha", "eval_flag",
                "Precision@20", "Recall@20", "NDCG@20", "Hit@20", "F1@20",
                "Rec_percentage", "Being_rec_percentage", "System_score",
                "Active_Rec_percentage", "Active_Being_rec_percentage", "Active_System_score",
                "In_Rec_percentage", "In_Being_rec_percentage", "In_System_score",
                "Rec_Nums_Inactive"
            ])

        for alpha in alphas:
            for flag in ["no_reweight", "reweight"]:
                base_logs_dir = os.path.join(root, "test_logs_reweight5" if flag == "reweight" else "test_logs", args.dataset)
                rating_list, rating_val, user_ids = load_logs(base_logs_dir, args.method, args.seed)
                need_n = max(rating_list.shape[0], int(np.max(rating_list)) + 1)
                if len(deg) < need_n:
                    deg = np.pad(deg, (0, need_n - len(deg)), mode="constant", constant_values=0)
                method_out = f"{args.method}_degreefair_a{alpha:g}"
                out_logs_dir = os.path.join(root, "test_logs_degreefair", args.dataset, flag)
                new_list, new_val = degree_debias_rerank(rating_list, rating_val, deg, alpha)
                save_logs(out_logs_dir, method_out, args.seed, new_list, new_val, user_ids)

                train_users = np.array(list(user_dict["train_user_set"].keys()), dtype=np.int64)
                pred20 = new_list[train_users, :20]
                res = eval_at_20(pred20, user_dict, label_map, pop_map, flag=args.eval_flag)

                results_by_flag[flag]["Rec_percentage"].append(float(res.get("Rec_percentage", 0.0)))
                results_by_flag[flag]["Being_rec_percentage"].append(float(res.get("Being_rec_percentage", 0.0)))
                results_by_flag[flag]["In_Being_rec_percentage"].append(float(res.get("In_Being_rec_percentage", 0.0)))
                results_by_flag[flag]["Rec_Nums_Inactive"].append(float(res.get("Rec_Nums_Inactive", 0.0)))

                w.writerow([
                    args.method, flag, args.seed, alpha, args.eval_flag,
                    res["Precision@20"], res["Recall@20"], res["NDCG@20"], res["Hit@20"], res["F1@20"],
                    res.get("Rec_percentage", 0.0), res.get("Being_rec_percentage", 0.0), res.get("System_score", 0.0),
                    res.get("Active_Rec_percentage", 0.0), res.get("Active_Being_rec_percentage", 0.0), res.get("Active_System_score", 0.0),
                    res.get("In_Rec_percentage", 0.0), res.get("In_Being_rec_percentage", 0.0), res.get("In_System_score", 0.0),
                    res.get("Rec_Nums_Inactive", 0.0),
                ])

    out_txt = os.path.join(out_dir, f"eval_degreefair_20_{args.method}_seed{args.seed}.txt")
    with open(out_txt, "w") as f:
        f.write(f"method={args.method}, seed={args.seed}, eval_flag={args.eval_flag}\n")
        f.write(f"alphas={alphas}\n\n")
        for flag in ["no_reweight", "reweight"]:
            f.write(f"[{flag}]\n")
            for m in ["Rec_percentage", "Being_rec_percentage", "In_Being_rec_percentage", "Rec_Nums_Inactive"]:
                f.write(f"  {m}: {results_by_flag[flag][m]}\n")
            f.write("\n")
    print("[OK] csv ->", out_csv)
    print("[OK] txt ->", out_txt)

    plot_four_panel_eval(alphas, results_by_flag, out_png)


if __name__ == "__main__":
    main()
