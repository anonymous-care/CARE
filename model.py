import torch
import torch.nn as nn
import numpy as np
from torch.nn import Linear
import torch.nn.functional as F
from torch_scatter import scatter
from utils import softmax


class GraphConv(nn.Module):

    def __init__(self, args):
        super(GraphConv, self).__init__()
        self.args = args

    def forward(self, embed, adj_sp_norm, edge_index, edge_weight, deg):
        agg_embed = embed
        embs = [embed]

        row, col = edge_index

        for hop in range(self.args.n_hops):
            out = agg_embed[row] * edge_weight.unsqueeze(-1)
            agg_embed = scatter(
                out, col, dim=0, dim_size=self.args.n_users, reduce='add')
            embs.append(agg_embed)

        embs = torch.stack(embs, dim=1)

        return embs

class LightGCN(nn.Module):
    def __init__(self, args):
        super(LightGCN, self).__init__()

        self.args = args

        self._init_weight()
        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.embeds = nn.Parameter(initializer(torch.empty(
            self.args.n_users, self.args.embedding_dim)))


    def _init_model(self):
        if self.args.model == 'LightGCN':
            return GraphConv(self.args)


    def batch_generate(self, user, pos_user, neg_user):
        user_gcn_embs = self.gcn(
            self.embeds, self.adj_sp_norm, self.edge_index, self.edge_weight, self.deg)

        user_gcn_embs = self.pooling(user_gcn_embs)

        user_embs = user_gcn_embs[user]
        pos_user_embs = user_gcn_embs[pos_user]
        neg_user_embs = user_gcn_embs[neg_user]

        return user_embs, pos_user_embs, neg_user_embs
    def forward(self, batch=None):
        user = batch['users']
        pos_user = batch['pos_items']
        neg_user = batch['neg_items']

        user_embs, pos_user_embs, neg_user_embs = self.batch_generate(
            user, pos_user, neg_user)

        return user_embs, pos_user_embs, neg_user_embs, self.embeds[user], self.embeds[pos_user], self.embeds[neg_user]

    def pooling(self, embeddings):
        if self.args.aggr == 'mean':
            return embeddings.mean(dim=1)
        elif self.args.aggr == 'sum':
            return embeddings.sum(dim=1)
        elif self.args.aggr == 'concat':
            return embeddings.view(embeddings.shape[0], -1)
        else:
            return embeddings[:, -1, :]

    def generate(self):
        user_gcn_embs = self.gcn(
            self.embeds, self.adj_sp_norm, self.edge_index, self.edge_weight, self.deg)

        user_embs = self.pooling(user_gcn_embs)

        return user_embs

    def generate_layers(self):
        return self.gcn(self.embeds, self.adj_sp_norm, self.edge_index, self.edge_weight, self.deg)

    def propagate(self):

        user_gcn_embs = self.gcn(
            self.embeds, self.adj_sp_norm, self.edge_index, self.edge_weight, self.deg
        )
        user_embs_all = self.pooling(user_gcn_embs)
        return user_embs_all


class MF(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_users = args.n_users
        self.emb_dim = args.embedding_dim
        self.user_embedding = nn.Embedding(self.n_users, self.emb_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.01)

    def forward(self, batch):
        users = batch["users"]        
        pos_items = batch["pos_items"]  
        neg_items = batch["neg_items"]

        u = self.user_embedding(users)  
        p = self.user_embedding(pos_items)  
        n = self.user_embedding(neg_items)  

        return u, p, n, u, p, n

    @torch.no_grad()
    def generate(self):
        return self.user_embedding.weight


class SASRec(nn.Module):
    def __init__(self, n_items, hidden=64, maxlen=50, n_layers=2, n_heads=2, dropout=0.2):
        super().__init__()
        self.n_items = n_items
        self.hidden = hidden
        self.maxlen = maxlen

        self.item_emb = nn.Embedding(n_items, hidden, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden)
        self.dropout = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(hidden)

    def forward(self, seq):

        B, L = seq.size()
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)

        x = self.item_emb(seq) + self.pos_emb(pos)
        x = self.dropout(self.ln(x))

        causal = torch.triu(torch.ones(L, L, device=seq.device), diagonal=1).bool()
        out = self.encoder(x, mask=causal)
        return out

    def predict_next(self, seq, candidates):

        out = self.forward(seq)      
        h = out[:, -1, :]               
        cand_emb = self.item_emb(candidates)
        score = (cand_emb * h.unsqueeze(1)).sum(-1)
        return score


class NeuMF(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_users = args.n_users
        self.emb_dim = args.embedding_dim

        self.gmf_user = nn.Embedding(self.n_users, self.emb_dim)
        self.gmf_item = nn.Embedding(self.n_users, self.emb_dim)

        self.mlp_user = nn.Embedding(self.n_users, self.emb_dim)
        self.mlp_item = nn.Embedding(self.n_users, self.emb_dim)

        mlp_hidden = getattr(args, "mlp_hidden", [128, 64]) 
        layers = []
        in_dim = self.emb_dim * 2
        for h in mlp_hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.mlp = nn.Sequential(*layers)

        final_in = self.emb_dim + (mlp_hidden[-1] if len(mlp_hidden) > 0 else self.emb_dim * 2)
        self.pred = nn.Linear(final_in, 1)
        for emb in [self.gmf_user, self.gmf_item, self.mlp_user, self.mlp_item]:
            nn.init.normal_(emb.weight, std=0.01)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def score_pairs(self, users: torch.LongTensor, items: torch.LongTensor) -> torch.Tensor:

        gu = self.gmf_user(users)        
        gi = self.gmf_item(items)         
        gmf_vec = gu * gi                 

        mu = self.mlp_user(users)     
        mi = self.mlp_item(items)        
        mlp_in = torch.cat([mu, mi], dim=-1)  
        mlp_vec = self.mlp(mlp_in)           

        x = torch.cat([gmf_vec, mlp_vec], dim=-1)  
        s = self.pred(x).squeeze(-1)         
        return s

    def score_user_candidates(self, users: torch.LongTensor, candidates: torch.LongTensor) -> torch.Tensor:

        B = users.shape[0]
        C = candidates.shape[0]

        uu = users.view(B, 1).expand(B, C).reshape(-1)          
        vv = candidates.view(1, C).expand(B, C).reshape(-1)    
        s = self.score_pairs(uu, vv).view(B, C)               
        return s

    def forward(self, batch):
        users = batch["users"]         
        pos_items = batch["pos_items"]  
        neg_items = batch["neg_items"] 

        u0 = self.gmf_user(users)
        p0 = self.gmf_item(pos_items)
        n0 = self.gmf_item(neg_items)  
        return u0, p0, n0, u0, p0, n0
