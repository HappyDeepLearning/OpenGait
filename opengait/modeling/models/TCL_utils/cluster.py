# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange
# from ...modules import SeparateFCs
# from .attentions import CausalAttentionLayer


# class PhaseClusterAdapter(nn.Module):
#     def __init__(self, in_channels, num_parts, num_clusters=4, num_heads=4):
#         super().__init__()
#         self.in_channels = in_channels
#         self.num_parts = num_parts
#         self.num_clusters = num_clusters
        
#         # self.in_project = nn.Conv2d(in_channels, hiddern_channels, kernel_size=1)

#         self.cluster_prototypes = nn.Parameter(torch.randn(num_parts, num_clusters, in_channels))
#         # self.temporal_attention = CausalAttentionLayer(dim=in_channels, num_heads=num_heads)
#         self.linear = SeparateFCs(parts_num=num_parts,
#                                   in_channels=in_channels * num_clusters,
#                                   out_channels=in_channels)
    
#     def _init_weights(self):
#         # nn.init.zeros_(self.linear.fc_bin)
#         nn.init.normal_(self.cluster_prototypes, mean=0.0, std=0.02)

#     def forward(self, x, training=True):

#         n, c, s, p = x.shape

#         # computing Cosine similarity
#         x = rearrange(x, 'n c s p -> p n s c').contiguous()  # [p, n, s, c]
#         cluster_prototypes = F.normalize(self.cluster_prototypes, dim=-1)
#         x_norm = F.normalize(x, dim=-1)
#         sim = torch.einsum('pnsc,pkc->pnsk', x_norm, cluster_prototypes)  # [p, n, s, K]

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

#         assignment = rearrange(assignment, 'p n s k -> p n k s').contiguous()  # [p, n, K, s]
#         assignment = assignment.unsqueeze(-1)  # [p, n, K, s, 1]
#         x_expanded = x.unsqueeze(2)  # [p, n, 1, s, c]
#         clustered = assignment * x_expanded  # [p, n, K, s, c]

#         # Temporal Pooling / Attention
#         clustered = clustered.max(dim=-2)[0]  # [p, n, K, c]
#         clustered = rearrange(clustered, 'p n k c -> n (k c) p').contiguous()

#         # clustered = rearrange(clustered, 'p n k s c -> (p n k) s c')
#         # clustered_attn = self.temporal_attention(clustered) + clustered
#         # clustered = clustered_attn.max(dim=1)[0]  # [p*n*k, c]
#         # clustered = rearrange(clustered, '(p n k) c -> n (k c) p', p=self.num_parts, n=n)
        
#         # Final projection layer to [n, c, p]
#         out = self.linear(clustered)  # [n, c, p]
#         return out, cluster_indices

#     def diversity_loss(self):
#         prototypes = F.normalize(self.cluster_prototypes, dim=-1)  # [p, K, c]
#         similarity_matrix = torch.einsum('pkc,pqc->pkq', prototypes, prototypes)  # [p, K, K]
#         identity = torch.eye(self.num_clusters, device=similarity_matrix.device).unsqueeze(0)  # [1, K, K]
#         diversity_loss = (similarity_matrix - identity).pow(2).mean()
#         return diversity_loss

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange
# from ...modules import SeparateFCs
# from .attention import CrossAttention

# class PhaseTokenCrossAttentionAdapter(nn.Module):
#     def __init__(self, in_channels, num_parts, num_clusters=4, num_heads=4):
#         super().__init__()
#         self.in_channels = in_channels
#         self.num_parts = num_parts
#         self.num_clusters = num_clusters
#         self.num_heads = num_heads

#         # Learnable phase tokens: [1, K, C]
#         self.phase_tokens = nn.Parameter(torch.randn(1, num_clusters, in_channels))

#         # Cross Attention: phase tokens as query, input as context
#         self.cross_attn = CrossAttention(
#             query_dim=in_channels,
#             context_dim=in_channels,
#             num_heads=num_heads,
#             qkv_bias=False
#         )

#         # Final projection layer to [n, c, p]
#         self.linear = SeparateFCs(
#             parts_num=num_parts,
#             in_channels=in_channels * num_clusters,
#             out_channels=in_channels
#         )

#     def forward(self, x):
#         """
#         x: [n, c, s, p]
#         Return: [n, c, p], [n, s] cluster_indices for visualization
#         """
#         n, c, s, p = x.shape
#         x = rearrange(x, 'n c s p -> p n s c')  # [p, n, s, c]

#         # Repeat phase tokens for batch and part
#         phase_tokens = self.phase_tokens.expand(p, -1, -1)  # [p, K, c]
#         phase_tokens = rearrange(phase_tokens, 'p k c -> (p) k c')
#         x_input = rearrange(x, 'p n s c -> (p n) s c')

#         # Repeat tokens across batch: [p*n, K, c]
#         phase_tokens = phase_tokens.unsqueeze(1).repeat(1, n, 1, 1)
#         phase_tokens = rearrange(phase_tokens, 'p n k c -> (p n) k c')

#         # Cross attention: [p*n, K, c] = attn([p*n, K, c], context=[p*n, s, c])
#         attended = self.cross_attn(phase_tokens, context=x_input)  # [p*n, K, c]

#         attended = rearrange(attended, '(p n) k c -> n (k c) p', p=p, n=n)
#         out = self.linear(attended)  # [n, c, p]

#         # Compute argmax indices for visualization
#         sim = torch.einsum('pnsc,pkc->pnsk', F.normalize(x, dim=-1), F.normalize(
#             rearrange(self.phase_tokens.expand(p, -1, -1), 'p k c -> p k c'), dim=-1))
#         cluster_indices = sim.argmax(dim=-1)  # [p, n, s]
#         cluster_indices = rearrange(cluster_indices, 'p n s -> n s p')
#         cluster_indices = torch.mode(cluster_indices, dim=-1).values  # [n, s]

#         return out, cluster_indices



import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ...modules import SeparateFCs
from .attentions import CausalAttentionLayer


class PhaseClusterAdapter(nn.Module):
    def __init__(self, in_channels, num_parts, num_clusters=4, num_heads=4, similarity='cosine'):
        super().__init__()
        self.in_channels = in_channels
        self.num_parts = num_parts
        self.num_clusters = num_clusters
        self.similarity = similarity.lower()
        assert self.similarity in ['cosine', 'euclidean'], "similarity must be 'cosine' or 'euclidean'"

        self.instance_norm = nn.InstanceNorm1d(in_channels, affine=False)
        self.temporal_attention = CausalAttentionLayer(dim=in_channels, num_heads=num_heads)
        self.cluster_prototypes = nn.Parameter(torch.randn(num_parts, num_clusters, in_channels))
        # self.linear = nn.Conv1d(in_channels * num_clusters, in_channels, 1, bias=False)
        self.linear = SeparateFCs(parts_num=num_parts,
                                  in_channels=in_channels * num_clusters,
                                  out_channels=in_channels)

    def forward(self, x, training=True):
        n, c, s, p = x.shape
        x_part = rearrange(x, 'n c s p -> p n s c')  # [p, n, s, c]

        # ========== Branch A: InstanceNorm + similarity ==========
        x_flat = rearrange(x_part, 'p n s c -> (p n) c s')
        x_normed = self.instance_norm(x_flat)
        x_normed = rearrange(x_normed, '(p n) c s -> p n s c', p=p, n=n)

        if self.similarity == 'cosine':
            x_normed_norm = F.normalize(x_normed, dim=-1)
            proto_norm = F.normalize(self.cluster_prototypes, dim=-1)
            sim = torch.einsum('pnsc,pkc->pnsk', x_normed_norm, proto_norm)  # [p, n, s, K]
        else:  # euclidean
            x_exp = x_normed.unsqueeze(2)  # [p, n, 1, s, c]
            proto_exp = self.cluster_prototypes.unsqueeze(1).unsqueeze(3)  # [p, 1, K, 1, c]
            dist_squared = ((x_exp - proto_exp) ** 2).sum(-1)  # [p, n, K, s]
            sim = -dist_squared.permute(0, 1, 3, 2).contiguous()  # [p, n, s, K]

        # Soft + STE assignment
        if training:
            soft_assignment = F.softmax(sim / 0.1, dim=-1)
            with torch.no_grad():
                hard_idx = soft_assignment.argmax(dim=-1, keepdim=True)
                hard_assignment = torch.zeros_like(soft_assignment).scatter_(-1, hard_idx, 1.0)
            assignment = (hard_assignment - soft_assignment).detach() + soft_assignment
        else:
            hard_idx = sim.argmax(dim=-1, keepdim=True)
            assignment = torch.zeros_like(sim).scatter_(-1, hard_idx, 1.0)

        cluster_indices = rearrange(hard_idx.squeeze(-1), 'p n s -> n s p').contiguous()
        cluster_indices = torch.mode(cluster_indices, dim=-1).values  # [n, s]

        # ========== Branch B: Attention + Aggregation ==========
        x = rearrange(x, 'n c s p -> (n p) s c').contiguous() # [n*p, s, c]
        x_attn = self.temporal_attention(x) + x  # [n*p, s, c]
        x_attn = rearrange(x_attn, '(n p) s c -> p n s c', p=p, n=n, s=s).contiguous()  # [p, n, s, c]

        assignment = rearrange(assignment, 'p n s k -> p n k s 1').contiguous()
        x_exp = x_attn.unsqueeze(2)  # [p, n, 1, s, c]
        grouped = assignment * x_exp  # [p, n, K, s, c]
        clustered = grouped.max(dim=3)[0]  # [p, n, K, c]
        clustered = rearrange(clustered, 'p n k c -> n (k c) p').contiguous()

        out = self.linear(clustered)  # [n, c, p]
        return out, cluster_indices

    def _init_weights(self):
        pass
        # nn.init.normal_(self.cluster_prototypes, mean=0.0, std=0.02)

    def diversity_loss(self):
        prototypes = F.normalize(self.cluster_prototypes, dim=-1)
        similarity_matrix = torch.einsum('pkc,pqc->pkq', prototypes, prototypes)  # [p, K, K]
        identity = torch.eye(self.num_clusters, device=similarity_matrix.device).unsqueeze(0)
        return (similarity_matrix - identity).pow(2).mean()
