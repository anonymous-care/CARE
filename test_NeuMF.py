# test_NeuMF.py
import os
import torch
import numpy as np

from parse import parse_args
from utils import seed_everything, minibatch
from dataprocess import load_data
from model import NeuMF


@torch.no_grad()
def test_per_user_neumf(model, user_dict, args):

    train_user_set = user_dict["train_user_set"]
    val_user_set   = user_dict["val_user_set"]

    test_users = torch.tensor(list(train_user_set.keys()), dtype=torch.long)

    device = args.device
    model.eval()

    n_users = int(args.n_users)
    topK = int(max(args.topks))

    cand_chunk = int(getattr(args, "cand_chunk", 4096))
    batch_users_size = int(getattr(args, "test_batch_size", 1024))

    all_rating = np.full((n_users, topK), -1, dtype=np.int64)
    all_value  = np.full((n_users, topK), -1e10, dtype=np.float32)

    candidates_all = torch.arange(n_users, device=device, dtype=torch.long)

    @torch.no_grad()
    def topk_merge(cur_scores, cur_items, new_scores, new_items, k):

        scores = torch.cat([cur_scores, new_scores], dim=1)
        items  = torch.cat([cur_items,  new_items],  dim=1)
        topv, topi = torch.topk(scores, k=k, dim=1)
        top_items = torch.gather(items, 1, topi)
        return topv, top_items

    with torch.no_grad():
        for bu in minibatch(test_users, batch_size=batch_users_size):
            bu = bu.to(device)
            B = bu.shape[0]

            running_scores = torch.full((B, topK), -1e10, device=device)
            running_items  = torch.full((B, topK), -1, device=device, dtype=torch.long)

            exclude = []
            bu_list = bu.tolist()
            for u in bu_list:
                tr = train_user_set.get(u, [])
                va = val_user_set.get(u, [])
                exclude.append(set(tr) | set(va) | {u})

            for start in range(0, n_users, cand_chunk):
                cand = candidates_all[start:start + cand_chunk]  
                scores = model.score_user_candidates(bu, cand)   
                scores = scores.clone()

                cand_cpu = cand.detach().cpu().numpy()
                for i in range(B):
                    ex = exclude[i]
                    if not ex:
                        continue
                    m = np.isin(cand_cpu, list(ex))
                    if m.any():
                        idx = torch.from_numpy(np.where(m)[0]).to(device)
                        scores[i, idx] = -1e10

                kk = min(topK, scores.shape[1])
                topv, topi = torch.topk(scores, k=kk, dim=1)
                top_items = cand[topi] 

                running_scores, running_items = topk_merge(
                    running_scores, running_items, topv, top_items, topK
                )


            bu_cpu = bu.detach().cpu().numpy().astype(np.int64)
            all_rating[bu_cpu] = running_items.detach().cpu().numpy().astype(np.int64)
            all_value[bu_cpu]  = running_scores.detach().cpu().numpy().astype(np.float32)

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
    print("[test_per_user_neumf] saved prefix:", prefix)
    print("[test_per_user_neumf] rating_list shape:", all_rating.shape,
          "| filled_users:", filled, "/", n_users,
          "| eval_users:", len(test_users))


def main():
    args = parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.path = os.getcwd()
    seed_everything(args.seed)

    train_cf, val_cf, test_cf, user_dict, args.n_users, clicked_set, adj, user_id_map, reverse_map, label_map, popularity_map = load_data(args)

    if args.model != "NeuMF":
        raise ValueError("test_NeuMF.py expects --model NeuMF")

    model = NeuMF(args).to(args.device)

    if args.reweight_flag == 0:
        ckpt = os.path.join(args.path, "trained_model", args.dataset, f"{args.model}_{args.seed}.pkl")
    else:
        ckpt = os.path.join(args.path, "trained_model_reweight5", f"{args.model}_{args.seed}_{args.penalty}.pkl")

    print("[LOAD]", ckpt)
    model.load_state_dict(torch.load(ckpt, map_location=args.device))
    model.eval()

    test_per_user_neumf(model, user_dict, args)


if __name__ == "__main__":
    main()
