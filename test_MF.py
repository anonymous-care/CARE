import os
import torch

from parse import parse_args
from utils import *
from evaluation import *
from model import *
from dataprocess import *


def _patch_os_mkdir_to_ignore_exists():

    _orig = os.mkdir

    def _safe_mkdir(path):
        try:
            _orig(path)
        except FileExistsError:
            pass

    os.mkdir = _safe_mkdir


if __name__ == '__main__':
    args = parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.path = os.getcwd()
    seed_everything(args.seed)

    train_cf, val_cf, test_cf, user_dict, args.n_users, clicked_set, adj, user_id_map, reverse_map, label_map, popularity_map = load_data(args)

    if args.neg_in_val_test == 1:
        clicked_set = user_dict['train_user_set']

    if args.model == "MF":
        model = MF(args).to(args.device)
    else:
        raise ValueError("test_MF.py expects --model MF")

    if args.reweight_flag == 0:
        ckpt = os.path.join(args.path, "trained_model", args.dataset, f"{args.model}_{args.seed}.pkl")
    else:
        ckpt = os.path.join(args.path, "trained_model_reweight5", f"{args.model}_{args.seed}_{args.penalty}.pkl")

    print("[LOAD]", ckpt)
    model.load_state_dict(torch.load(ckpt, map_location=args.device))
    model.eval()

    user_embs = model.generate()
    test_per_user(user_embs, user_dict, args)
