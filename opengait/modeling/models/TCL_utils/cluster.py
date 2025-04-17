import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class PhaseClusterAdapter(nn.Module):
    """
    PhaseClusterAdapter, but clustering on the flattened (c, p) space:
      - x: [batch, c, seq_len, parts]
      - merge c & parts → D = c * parts
      - cluster prototypes: [K, D]
      - output: [batch, c, parts] + cluster_indices [batch, seq_len]
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_parts: int,
        num_clusters: int = 4,
        similarity: str = 'cosine'
    ):
        super().__init__()
        self.in_channels  = in_channels
        self.num_parts    = num_parts
        self.num_clusters = num_clusters
        self.similarity   = similarity
        self.hidden_dim   = hidden_dim

        self.in_projector = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
        )

        # clustering in the flattened (c * p) space
        D = hidden_dim * num_parts
        self.cluster_prototypes = nn.Parameter(torch.randn(num_clusters, D))
        self._init_weights()

        # project K * c → c for each part (length = p)
        self.out_projector = nn.Sequential(
            nn.Conv1d(hidden_dim * num_clusters, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
            nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False),
        )

        # start with a softer temperature
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def _init_weights(self):
        nn.init.orthogonal_(self.cluster_prototypes)

    def forward(self, x: torch.Tensor):
        """
        Args:
          x: [B, C, S, P]
        Returns:
          out:              [B, C, P]
          cluster_indices:  [B, S]
        """
        B, C_in, S, P = x.shape

        # 1) project to hidden dim
        x = rearrange(x, 'b c s p -> (b p) c s').contiguous()  # [B*P, C_in, S]
        x = self.in_projector(x)                # [B*P, C_h, S]
        x = rearrange(x, '(b p) d s -> b d s p', b=B, p=P).contiguous()  # [B, C_h, S, P]
        C_h = x.shape[1]
        D = C_h * P

        # 1) flatten channel & part dims → [B, S, D]
        x_flat = rearrange(x, 'b c s p -> b s (c p)').contiguous()

        # 2) compute similarity [B, S, K]
        if self.similarity == 'cosine':
            x_norm   = F.normalize(x_flat, dim=-1)               # [B, S, D]
            proto_nm = F.normalize(self.cluster_prototypes, dim=-1)  # [K, D]
            sim = torch.einsum('bsd,kd->bsk', x_norm, proto_nm)
        else:
            x_exp   = x_flat.unsqueeze(2)                         # [B, S, 1, D]
            proto_exp = self.cluster_prototypes.unsqueeze(0)      # [1, K, D]
            dist = ((x_exp - proto_exp) ** 2).sum(-1)             # [B, S, K]
            sim  = -dist

        # 3) soft assignment (with STE during training)
        soft = F.softmax(sim / self.temperature, dim=-1)         # [B, S, K]
        with torch.no_grad():
            hard_idx       = soft.argmax(dim=-1, keepdim=True)       # [B, S, 1]
            hard_assign    = torch.zeros_like(soft).scatter_(-1, hard_idx, 1.0)
        assignment = (hard_assign - soft).detach() + soft

        self.soft_assignment = soft                               # save for losses / viz
        cluster_indices      = soft.argmax(dim=-1)                # [B, S]

        # 4) compute cluster-wise pooled features: [B, K, D]
        #    cluster_feats[b,k] = sum_{t=1..S} assignment[b,t,k] * x_flat[b,t]
        cluster_feats = torch.einsum('bsk,bsd->bkd', assignment, x_flat)

        # 5) reshape to [B, K*C, P]
        clustered = rearrange(
            cluster_feats,
            'b k (c p) -> b (k c) p',
            c=C_h, p=P
        )

        # 6) project back to [B, C, P]
        out = self.out_projector(clustered)

        return out, cluster_indices

    def diversity_loss(self):
        """
        Diversity loss, switchable by self.similarity:
          - 'euclidean': mean_{i≠j} ||p_i - p_j||^2
          - 'cosine'   : mean_{i≠j} (⟨p_i,p_j⟩)²  after L2 normalization
        """
        K = self.num_clusters
        prototypes = self.cluster_prototypes  # [K, D]

        if self.similarity == 'cosine':
            # Cosine: encourage orthogonality
            proto_norm = F.normalize(prototypes, dim=-1)     # [K, D]
            sim_mat    = torch.matmul(proto_norm, proto_norm.t())  # [K, K]
            I          = torch.eye(K, device=sim_mat.device)
            # (sim - I)^2 zeroes out diagonal, penalizes off-diagonals
            return (sim_mat - I).pow(2).mean()
        else:  # 'euclidean'
            # Euclidean: pairwise squared distance
            dist = torch.cdist(prototypes, prototypes, p=2)  # [K, K]
            dist_sq = dist.pow(2)
            mask = ~torch.eye(K, device=dist_sq.device, dtype=torch.bool)
            return dist_sq.masked_select(mask).mean()

    def balance_loss(self):
        """
        Encourage clusters to be equally used:
        KL( avg_soft || uniform )
        """
        avg_prob = self.soft_assignment.mean(dim=(0, 1))      # [K]
        uni      = torch.full_like(avg_prob, 1.0/self.num_clusters)
        return F.kl_div(avg_prob.log(), uni, reduction='batchmean')


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange
# from ...modules import SeparateFCs  # 请根据实际路径修改导入

# class PhaseClusterAdapter(nn.Module):
#     def __init__(self, in_channels, num_parts, num_clusters=4, num_heads=4, similarity='cosine'):
#         super().__init__()
#         self.in_channels = in_channels
#         self.num_parts = num_parts
#         self.num_clusters = num_clusters
#         self.similarity = similarity  # 'euclidean' or 'cosine'

#         self.cluster_prototypes = nn.Parameter(torch.randn(num_parts, num_clusters, in_channels))
#         # self.linear = SeparateFCs(parts_num=num_parts,
#         #                           in_channels=in_channels * num_clusters,
#         #                           out_channels=in_channels)
#         self.out_projector = nn.Sequential(
#             nn.Conv1d(in_channels * num_clusters, out_channels=in_channels, kernel_size=1),
#             nn.BatchNorm1d(in_channels),
#             nn.GELU(),
#             nn.Conv1d(in_channels, out_channels=in_channels, kernel_size=1),
#         )
#         self.temperature = nn.Parameter(torch.tensor(0.07))  # learnable temperature

#     def _init_weights(self):
#         # pass
#         nn.init.orthogonal_(self.cluster_prototypes)

#     def forward(self, x, training=True):
#         """
#         Args:
#             x: Tensor of shape [n, c, s, p]
#             training: whether to use STE soft assignment
#         Returns:
#             out: [n, c, p]  - output features per part
#             cluster_indices: [n, s] - one cluster label per frame
#         """
#         n, c, s, p = x.shape

#         # compute similarity
#         x = rearrange(x, 'n c s p -> p n s c').contiguous()  # [p, n, s, c]

#         if self.similarity == 'cosine':
#             x_normed = F.normalize(x, dim=-1)
#             proto_normed = F.normalize(self.cluster_prototypes, dim=-1)
#             sim = torch.einsum('pnsc,pkc->pnsk', x_normed, proto_normed)  # [p, n, s, K]
#         else:  # euclidean
#             x_exp = x.unsqueeze(2)  # [p, n, 1, s, c]
#             proto_exp = self.cluster_prototypes.unsqueeze(1).unsqueeze(3)  # [p, 1, K, 1, c]
#             dist_squared = ((x_exp - proto_exp) ** 2).sum(-1)  # [p, n, K, s]
#             sim = -dist_squared.permute(0, 1, 3, 2).contiguous()  # [p, n, s, K]

#         # Soft assignment (STE if training)
#         soft_assignment = F.softmax(sim / self.temperature, dim=-1)  # [p, n, s, K]

#         if training:
#             with torch.no_grad():
#                 hard_idx = soft_assignment.argmax(dim=-1, keepdim=True)
#                 hard_assignment = torch.zeros_like(soft_assignment).scatter_(-1, hard_idx, 1.0)
#             assignment = (hard_assignment - soft_assignment).detach() + soft_assignment  # STE
#         else:
#             hard_idx = sim.argmax(dim=-1, keepdim=True)
#             assignment = torch.zeros_like(sim).scatter_(-1, hard_idx, 1.0)

#         self.soft_assignment = soft_assignment  # for inspection/visualization

#         # Soft assignment 平均 + argmax -> [n, s]
#         cluster_indices = soft_assignment.mean(dim=0).argmax(dim=-1)  # [n, s]

#         # 聚类结果重新组织为 [p, n, K, s, 1]
#         assignment = rearrange(assignment, 'p n s k -> p n k s').unsqueeze(-1)  # [p, n, K, s, 1]
#         x_expanded = x.unsqueeze(2)  # [p, n, 1, s, c]
#         clustered = assignment * x_expanded  # [p, n, K, s, c]

#         # temporal pooling per cluster
#         clustered = clustered.max(dim=-2)[0]  # [p, n, K, c]
#         clustered = rearrange(clustered, 'p n k c -> n (k c) p')  # [n, K*c, p]

#         # project to final feature
#         out = self.out_projector(clustered)  # [n, c, p]

#         return out, cluster_indices

#     # def diversity_loss(self):
#     #     prototypes = F.normalize(self.cluster_prototypes, dim=-1)  # [p, K, c]
#     #     similarity_matrix = torch.einsum('pkc,pqc->pkq', prototypes, prototypes)  # [p, K, K]
#     #     identity = torch.eye(self.num_clusters, device=similarity_matrix.device).unsqueeze(0)  # [1, K, K]
#     #     diversity_loss = (similarity_matrix - identity).pow(2).mean()
#     #     return diversity_loss

#     def diversity_loss(self, margin=3.0):
#         prototypes = self.cluster_prototypes  # [p, K, c]
#         p, K, c = prototypes.shape

#         # pairwise distance
#         proto1 = prototypes.unsqueeze(2)  # [p, K, 1, c]
#         proto2 = prototypes.unsqueeze(1)  # [p, 1, K, c]
#         dist = ((proto1 - proto2) ** 2).sum(-1)  # [p, K, K]

#         # mask out diagonal
#         mask = ~torch.eye(K, dtype=torch.bool, device=prototypes.device).unsqueeze(0)  # [1, K, K]
#         dist_off_diag = dist.masked_select(mask).view(p, K, K - 1)  # [p, K, K-1]

#         # hinge loss to push pairwise distance above margin
#         loss = F.relu(margin - dist_off_diag).mean()  # the smaller the distance, the larger the loss

#         return loss

#     def balance_loss(self):
#         # [p, n, s, K] -> [K]
#         avg_prob = self.soft_assignment.mean(dim=(0,1,2))
#         uni = torch.full_like(avg_prob, 1.0/self.num_clusters)
#         return F.kl_div(avg_prob.log(), uni, reduction='batchmean')


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange

# class PhaseClusterAdapter(nn.Module):
#     def __init__(self, in_channels, num_parts, num_clusters=4, kmeans_iters=10):
#         super().__init__()
#         self.c = in_channels
#         self.p = num_parts
#         self.num_clusters = num_clusters
#         self.kmeans_iters = kmeans_iters
#         self.out_projecter = nn.Conv1d(in_channels * num_clusters, in_channels, 1, bias=False)

#     @torch.no_grad()
#     def cluster_frames(self, x):
#         """
#         GPU版 KMeans with Euclidean distance (avoids cdist)
#         x: [n, c, s, p]
#         return: cluster_labels: [n, s, p]
#         """
#         n, c, s, p = x.shape
#         x_reshaped = x.permute(0, 3, 2, 1).reshape(-1, s, c)  # [b=n*p, s, c]
#         b = x_reshaped.size(0)

#         # 初始化聚类中心：从每段随机采样 K 帧作为初始质心
#         rand_idx = torch.randint(0, s, (b, self.num_clusters), device=x.device)
#         centroids = torch.gather(x_reshaped, 1, rand_idx.unsqueeze(-1).expand(-1, -1, c))  # [b, K, c]

#         for _ in range(self.kmeans_iters):
#             # 欧氏距离展开式：||x - c||^2 = ||x||^2 + ||c||^2 - 2xTc
#             x_sq = (x_reshaped ** 2).sum(dim=-1, keepdim=True)  # [b, s, 1]
#             c_sq = (centroids ** 2).sum(dim=-1).unsqueeze(1)    # [b, 1, K]
#             xc = torch.matmul(x_reshaped, centroids.transpose(1, 2))  # [b, s, K]

#             dist = x_sq + c_sq - 2 * xc  # [b, s, K]
#             labels = dist.argmin(dim=-1)  # [b, s]

#             # one-hot assignment
#             one_hot = F.one_hot(labels, num_classes=self.num_clusters).float()  # [b, s, K]
#             count = one_hot.sum(dim=1, keepdim=True) + 1e-6  # [b, 1, K]
#             centroids = torch.einsum('bsk,bsc->bkc', one_hot, x_reshaped) / count.transpose(1, 2)

#         # reshape 回原来的 [n, s, p]
#         cluster_labels = labels.view(n, p, s).permute(0, 2, 1).contiguous()
#         return cluster_labels

#     def forward(self, x):
#         """
#         x: [n, c, s, p]
#         return:
#             out: [n, c, p]
#             cluster_labels: [n, s]
#         """
#         n, c, s, p = x.shape
#         cluster_labels = self.cluster_frames(x)  # [n, s, p]

#         out = []
#         for cluster_id in range(self.num_clusters):
#             mask = (cluster_labels == cluster_id).float()  # [n, s, p]
#             mask = rearrange(mask, 'n s p -> n 1 s p')

#             # use masked max pooling with valid check
#             masked_x = x.masked_fill(mask == 0, float('-inf'))
#             pooled = masked_x.max(dim=2).values  # [n, c, p]
#             pooled[pooled == float('-inf')] = 0  # avoid -inf → nan
#             out.append(pooled)

#         out = torch.cat(out, dim=1)  # [n, c*K, p]
#         out = self.out_projecter(out)  # [n, c, p]

#         # 聚类 index 取每帧在所有 part 中的众数
#         cluster_labels = cluster_labels.mode(dim=-1).values  # [n, s]
#         return out, cluster_labels


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange
# from ...modules import SeparateFCs
# # from .attentions import CausalAttentionLayer


# class PhaseClusterAdapter(nn.Module):
#     def __init__(self, in_channels, num_parts, num_clusters=4, num_heads=4, similarity='cosine'):
#         super().__init__()
#         self.in_channels = in_channels
#         self.num_parts = num_parts
#         self.num_clusters = num_clusters
#         self.similarity = similarity.lower()
#         assert self.similarity in ['cosine', 'euclidean'], "similarity must be 'cosine' or 'euclidean'"

#         self.instance_norm = nn.InstanceNorm1d(in_channels, affine=False)
#         # self.temporal_attention = CausalAttentionLayer(dim=in_channels, num_heads=num_heads)
#         self.cluster_prototypes = nn.Parameter(torch.randn(num_parts, num_clusters, in_channels))
#         # self.linear = nn.Conv1d(in_channels * num_clusters, in_channels, 1, bias=False)
#         self.linear = SeparateFCs(parts_num=num_parts,
#                                   in_channels=in_channels * num_clusters,
#                                   out_channels=in_channels)

#     def forward(self, x, training=True):
#         n, c, s, p = x.shape
#         x_part = rearrange(x, 'n c s p -> p n s c')  # [p, n, s, c]

#         # ========== Branch A: InstanceNorm + similarity ==========
#         x_flat = rearrange(x_part, 'p n s c -> (p n) c s')
#         x_normed = self.instance_norm(x_flat)
#         x_normed = rearrange(x_normed, '(p n) c s -> p n s c', p=p, n=n)

#         if self.similarity == 'cosine':
#             x_normed_norm = F.normalize(x_normed, dim=-1)
#             proto_norm = F.normalize(self.cluster_prototypes, dim=-1)
#             sim = torch.einsum('pnsc,pkc->pnsk', x_normed_norm, proto_norm)  # [p, n, s, K]
#         else:  # euclidean
#             x_exp = x_normed.unsqueeze(2)  # [p, n, 1, s, c]
#             proto_exp = self.cluster_prototypes.unsqueeze(1).unsqueeze(3)  # [p, 1, K, 1, c]
#             dist_squared = ((x_exp - proto_exp) ** 2).sum(-1)  # [p, n, K, s]
#             sim = -dist_squared.permute(0, 1, 3, 2).contiguous()  # [p, n, s, K]

#         # Soft + STE assignment
#         if training:
#             soft_assignment = F.softmax(sim / 0.1, dim=-1)
#             with torch.no_grad():
#                 hard_idx = soft_assignment.argmax(dim=-1, keepdim=True)
#                 hard_assignment = torch.zeros_like(soft_assignment).scatter_(-1, hard_idx, 1.0)
#             assignment = (hard_assignment - soft_assignment).detach() + soft_assignment
#         else:
#             hard_idx = sim.argmax(dim=-1, keepdim=True)
#             assignment = torch.zeros_like(sim).scatter_(-1, hard_idx, 1.0)

#         cluster_indices = rearrange(hard_idx.squeeze(-1), 'p n s -> n s p').contiguous()
#         cluster_indices = torch.mode(cluster_indices, dim=-1).values  # [n, s]

#         # ========== Branch B: Attention + Aggregation ==========
#         # x = rearrange(x, 'n c s p -> (n p) s c').contiguous() # [n*p, s, c]
#         # x_attn = self.temporal_attention(x) + x  # [n*p, s, c]
#         # x_attn = rearrange(x_attn, '(n p) s c -> p n s c', p=p, n=n, s=s).contiguous()  # [p, n, s, c]

#         x_attn = rearrange(x, 'n c s p -> p n s c') .contiguous()  # [p, n, s, c]

#         # assign
#         assignment = rearrange(assignment, 'p n s k -> p n k s 1').contiguous()
#         x_exp = x_attn.unsqueeze(2)  # [p, n, 1, s, c]
#         grouped = assignment * x_exp  # [p, n, K, s, c]

#         # temporal pooling
#         clustered = grouped.max(dim=3)[0]  # [p, n, K, c]
#         clustered = rearrange(clustered, 'p n k c -> n (k c) p').contiguous()


        
#         # out projecter
#         out = self.linear(clustered)  # [n, c, p]
#         return out, cluster_indices

#     def _init_weights(self):
#         pass
#         # nn.init.normal_(self.cluster_prototypes, mean=0.0, std=0.02)

#     def diversity_loss(self):
#         prototypes = F.normalize(self.cluster_prototypes, dim=-1)
#         similarity_matrix = torch.einsum('pkc,pqc->pkq', prototypes, prototypes)  # [p, K, K]
#         identity = torch.eye(self.num_clusters, device=similarity_matrix.device).unsqueeze(0)
#         return (similarity_matrix - identity).pow(2).mean()