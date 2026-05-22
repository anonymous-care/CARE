# main_neumf.py
import os
import time
import gc
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from prettytable import PrettyTable

from parse import parse_args
from utils import *
from evaluation import *
from model import *
from dataprocess import *


def bpr_softplus_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """
    pos_scores: [B]
    neg_scores: [B, K]
    """
    pos = pos_scores.view(-1, 1)            
    loss_mat = F.softplus(neg_scores - pos) 
    return loss_mat.mean()


def run(model, optimizer, train_cf, clicked_set, user_dict, args, label_map, popularity_map):
    args.user_dict = user_dict
    start = time.time()

    for epoch in range(args.epochs):
        neg_cf = neg_sample_before_epoch(train_cf, clicked_set, args)

        dataset = Dataset(users=train_cf[:, 0], pos_items=train_cf[:, 1], neg_items=neg_cf, args=args)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=dataset.collate_batch,
            pin_memory=args.pin_memory
        )

        model.train()
        epoch_loss = 0.0
        num_steps = 0

        for batch in dataloader:
            num_steps += 1
            batch = batch_to_gpu(batch, args.device)

            users = batch["users"]       
            pos_items = batch["pos_items"]  
            neg_items = batch["neg_items"] 

            u0, p0, n0, _, _, _ = model(batch)

            pos_scores = model.score_pairs(users, pos_items) 

            B, K = neg_items.shape
            uu = users.view(B, 1).expand(B, K).reshape(-1)   
            vv = neg_items.reshape(-1)                       
            neg_scores = model.score_pairs(uu, vv).view(B, K)

            bpr_loss = bpr_softplus_loss(pos_scores, neg_scores)

            l2_loss = cal_l2_loss(u0, p0, n0, u0.shape[0])

            batch_loss = bpr_loss + args.l2 * l2_loss

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            epoch_loss += float(batch_loss.item())

            del u0, p0, n0, pos_scores, neg_scores
            gc.collect()

        avg_loss = epoch_loss / max(1, num_steps)

        if epoch % 10 == 0:
            model.eval()
            res = PrettyTable()
            res.field_names = ["Time", "Epoch", "Training_loss",
                               "Recall", "NDCG", "Precision", "Hit_ratio", "F1",
                               "rec_percentage", "being_rec_percentage", "system_score", "rec_nums",
                               "in_rec_percentage", "in_being_rec_percentage", "in_system_score",
                               "active_rec_percentage", "active_being_rec_percentage", "active_system_score",
                               "Active_Pop_Rec_percentage", "Active_Pop_Being_rec_percentage", "Active_Pop_System_score",
                               "Active_Nor_Rec_percentage", "Active_Nor_Being_rec_percentage", "Active_Nor_System_score",
                               "Inactive_Pop_Rec_percentage", "Inactive_Pop_Being_rec_percentage", "Inactive_Pop_System_score",
                               "Inactive_Nor_Rec_percentage", "Inactive_Nor_Being_rec_percentage", "Inactive_Nor_System_score"
                               ]

            test_res = test_neumf(model, user_dict, args, label_map, popularity_map, flag="val")

            res.add_row(
                [format(time.time() - start, ".4f"), epoch, format(avg_loss, ".4f"),
                 test_res["Recall"], test_res["NDCG"], test_res["Precision"],
                 test_res["Hit_ratio"], test_res["F1"],
                 test_res["Rec_percentage"], test_res["Being_rec_percentage"],
                 test_res["System_score"], test_res["Rec_Nums_Inactive"],
                 test_res["In_Rec_percentage"], test_res["In_Being_rec_percentage"], test_res["In_System_score"],
                 test_res["Active_Rec_percentage"], test_res["Active_Being_rec_percentage"], test_res["Active_System_score"],
                 test_res.get("Active_Pop_Rec_percentage", 0.0), test_res.get("Active_Pop_Being_rec_percentage", 0.0),
                 test_res.get("Active_Pop_System_score", 0.0),
                 test_res.get("Active_Nor_Rec_percentage", 0.0), test_res.get("Active_Nor_Being_rec_percentage", 0.0),
                 test_res.get("Active_Nor_System_score", 0.0),
                 test_res.get("Inactive_Pop_Rec_percentage", 0.0), test_res.get("Inactive_Pop_Being_rec_percentage", 0.0),
                 test_res.get("Inactive_Pop_System_score", 0.0),
                 test_res.get("Inactive_Nor_Rec_percentage", 0.0), test_res.get("Inactive_Nor_Being_rec_percentage", 0.0),
                 test_res.get("Inactive_Nor_System_score", 0.0)]
            )

            print(res)
            with open(args.path + args.save_path, "a") as log_file:
                log_file.write(res.get_string() + "\n")

            if args.save:
                save_dir = os.path.join(os.getcwd(), "trained_model", args.dataset)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{args.model}_{args.seed}.pkl")
                torch.save(model.state_dict(), save_path)
                print("save success:", save_path)

    print("Finish training NeuMF.")


if __name__ == "__main__":
    args = parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.path = os.getcwd()
    seed_everything(args.seed)
    print("Configurations:", args)

    train_cf, val_cf, test_cf, user_dict, args.n_users, clicked_set, adj, user_id_map, reverse_map, label_map, popularity_map = load_data(args)

    if args.neg_in_val_test == 1:
        clicked_set = user_dict["train_user_set"]

    if args.model != "NeuMF":
        raise ValueError("main_neumf.py expects --model NeuMF")

    if not hasattr(args, "cand_chunk"):
        args.cand_chunk = 4096
    if not hasattr(args, "test_batch_size"):
        args.test_batch_size = 4096

    model = NeuMF(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    run(model, optimizer, train_cf, clicked_set, user_dict, args, label_map, popularity_map)
