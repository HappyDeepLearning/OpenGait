import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class PhaseClusterAdapter(nn.Module):
    """
    Spatial PhaseClusterAdapter with 2D conv output projection:
      - Input x: [batch, channels, seq_len, height, width]
      - Outputs:
          out: [batch, channels, height, width]
          cluster_indices: [batch, seq_len]
    """
    def __init__(self,
                 in_channels: int,
                 num_clusters: int = 4,
                 similarity: str = 'cosine',
                 hight=16,
                 width=11):
        super().__init__()
        self.C = in_channels
        self.num_clusters = num_clusters
        self.similarity = similarity.lower()
        # cluster_prototypes will be initialized lazily in forward()
        p = hight * width  # number of spatial parts
        self.cluster_prototypes = nn.Parameter(
            torch.randn(p, self.num_clusters, self.C)
        )

        # output projector: 1×1 2D convs over the (K·C) feature map
        self.out_projector = nn.Sequential(
            nn.Conv2d(self.num_clusters * self.C, self.C, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.C),
            nn.GELU(),
            nn.Conv2d(self.C, self.C, kernel_size=1),
        )
        # learnable temperature for softmax
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def _init_weights(self):
        pass

    def forward(self, x: torch.Tensor, training: bool = True):
        """
        Args:
          x: input features of shape [n, C, S, H, W]
          training: whether to apply Straight‑Through Estimator on hard assignment

        Returns:
          out: aggregated features [n, C, H, W]
          cluster_indices: hard cluster label per frame [n, S]
        """
        n, c, s, h, w = x.shape
        assert c == self.C, f"Expected input channels {self.C}, got {c}"
        p = h * w  # number of spatial parts

        # 1) flatten spatial dims into a single "parts" dimension
        x_flat = x.view(n, c, s, p)                       # [n, C, S, P]
        x_flat = rearrange(x_flat, 'n C S P -> P n S C')  # [P, n, S, C]

        # 2) compute similarity between each part's feature and its K prototypes
        if self.similarity == 'cosine':
            x_norm = F.normalize(x_flat, dim=-1)                  # [P,n,S,C]
            proto_norm = F.normalize(self.cluster_prototypes, dim=-1)  # [P,K,C]
            sim = torch.einsum('pnsc,pkc->pnsk', x_norm, proto_norm)   # [P,n,S,K]
        else:  # euclidean distance
            x_e = x_flat.unsqueeze(2)                              # [P,n,1,S,C]
            p_e = self.cluster_prototypes.unsqueeze(1).unsqueeze(3)  # [P,1,K,1,C]
            dist2 = ((x_e - p_e)**2).sum(-1)                        # [P,n,K,S]
            sim = -dist2.permute(0,1,3,2).contiguous()             # [P,n,S,K]

        # 3) compute soft assignments via temperature‑scaled softmax
        soft = F.softmax(sim / self.temperature, dim=-1)           # [P,n,S,K]
        self.soft_assignment = soft  
        # optionally apply STE for hard assignment
        if training:
            with torch.no_grad():
                idx = soft.argmax(dim=-1, keepdim=True)           # [P,n,S,1]
                one_hot = torch.zeros_like(soft).scatter_(-1, idx, 1.0)
            assign = (one_hot - soft).detach() + soft              # STE
        else:
            idx = sim.argmax(dim=-1, keepdim=True)
            assign = torch.zeros_like(sim).scatter_(-1, idx, 1.0)

        # 4) derive a single cluster index per frame by averaging across parts
        avg_soft = soft.mean(dim=0)        # [n,S,K]
        cluster_indices = avg_soft.argmax(dim=-1)  # [n,S]

        # 5) aggregate features per cluster by max‑pooling over the time dimension
        a = rearrange(assign, 'P n S K -> P n K S').unsqueeze(-1)  # [P,n,K,S,1]
        x_e = x_flat.unsqueeze(2)                                  # [P,n,1,S,C]
        clustered = (a * x_e).max(dim=3)[0]                        # [P,n,K,C]
        # reshape into [n, K*C, H, W]
        clustered = rearrange(clustered, 'P n K C -> n (K C) P')
        clustered = clustered.view(n, self.num_clusters * self.C, h, w)      # [n, K·C, H, W]

        # 6) project concatenated cluster features back to C channels via 2D conv
        out = self.out_projector(clustered)                        # [n, C, H, W]

        return out, cluster_indices

    def diversity_loss(self) -> torch.Tensor:
        """
        Encourage prototypes within each part to be distinct.
        - Cosine mode: minimize squared off-diagonal correlations.
        - Euclidean mode: maximize pairwise distances.
        Returns average diversity loss over all parts.
        """
        # self.cluster_prototypes: [P, K, C]
        P, K, C = self.cluster_prototypes.shape
        proto = self.cluster_prototypes  # [P,K,C]

        if self.similarity == 'cosine':
            # normalize each prototype vector
            proto_n = F.normalize(proto, dim=-1)  # [P,K,C]
            # compute K×K similarity matrix per part
            sim_mat = torch.einsum('pkc,pjc->pkj', proto_n, proto_n)  # [P,K,K]
            # zero out diagonal
            eye = torch.eye(K, device=sim_mat.device).unsqueeze(0)    # [1,K,K]
            loss = ((sim_mat - eye)**2).sum(dim=(1,2)) / (K*(K-1))    # [P]
        else:
            # pairwise distances per part
            # flatten parts × clusters for cdist
            flat = proto.view(P * K, C)
            dist = torch.cdist(flat, flat, p=2).view(P, K, P, K)
            # we only care distances within same part: diagonal over the P dimension
            dist_pp = torch.stack([dist[i, :, i, :] for i in range(P)], dim=0)  # [P,K,K]
            # zero diagonal
            mask = ~torch.eye(K, device=dist_pp.device, dtype=torch.bool)
            dist_sq = dist_pp.pow(2)
            loss = dist_sq.masked_select(mask).view(P, -1).mean(dim=1)  # [P]

        # average over parts
        return loss.mean()

    def balance_loss(self) -> torch.Tensor:
        """
        Encourage clusters to be equally used across batch, frames, and parts.
        Compute KL(avg_soft || uniform).
        """
        # self.soft_assignment: [P, n, S, K]
        if self.soft_assignment is None:
            raise RuntimeError("soft_assignment not computed in forward")
        avg_prob = self.soft_assignment.mean(dim=(0,1,2))  # [K]
        uni = torch.full_like(avg_prob, 1.0 / self.num_clusters)
        # kl_div expects input = log-probs
        return F.kl_div(avg_prob.log(), uni, reduction='batchmean')

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange
# # from ...modules import SeparateFCs  # 请根据实际路径修改导入

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


#     def diversity_loss(self):
#         """
#         Diversity loss, switchable by self.similarity:
#           - 'euclidean': mean_{i≠j} ||p_i - p_j||^2
#           - 'cosine'   : mean_{i≠j} (⟨p_i,p_j⟩)²  after L2 normalization
#         """
#         K = self.num_clusters
#         prototypes = self.cluster_prototypes  # [K, D]

#         if self.similarity == 'cosine':
#             # Cosine: encourage orthogonality
#             proto_norm = F.normalize(prototypes, dim=-1)     # [K, D]
#             sim_mat    = torch.matmul(proto_norm, proto_norm.t())  # [K, K]
#             I          = torch.eye(K, device=sim_mat.device)
#             # (sim - I)^2 zeroes out diagonal, penalizes off-diagonals
#             return (sim_mat - I).pow(2).mean()
#         else:  # 'euclidean'
#             # Euclidean: pairwise squared distance
#             dist = torch.cdist(prototypes, prototypes, p=2)  # [K, K]
#             dist_sq = dist.pow(2)
#             mask = ~torch.eye(K, device=dist_sq.device, dtype=torch.bool)
#             return dist_sq.masked_select(mask).mean()

#     def balance_loss(self):
#         """
#         Encourage clusters to be equally used:
#         KL( avg_soft || uniform )
#         """
#         avg_prob = self.soft_assignment.mean(dim=(0, 1))      # [K]
#         uni      = torch.full_like(avg_prob, 1.0/self.num_clusters)
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
#         self.num_clustersmeans_iters = kmeans_iters
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

#         for _ in range(self.num_clustersmeans_iters):
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