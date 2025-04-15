from inspect import isfunction
from einops import rearrange, repeat
import torch
from torch import nn
import torch.nn.functional as F
import math


def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def get_sinusoid_encoding_table(time_idx: torch.Tensor, dim: int) -> torch.Tensor:
    """
    生成 sin/cos 位置编码（同 Transformer）。
    输入：
        - time_idx: [B, T]，每个时间帧在原始视频中的帧位置
        - dim: embedding 维度
    输出：
        - [B, T, dim] 的位置编码
    """
    device = time_idx.device
    position = time_idx.unsqueeze(-1).float()  # [B, T, 1]
    div_term = torch.exp(torch.arange(0, dim, 2, device=device) * (-torch.log(torch.tensor(10000.0)) / dim))  # [dim//2]
    pe = torch.zeros(*position.shape[:-1], dim, device=device)  # [B, T, dim]
    pe[..., 0::2] = torch.sin(position * div_term)
    pe[..., 1::2] = torch.cos(position * div_term)
    return pe

class CausalAttentionLayer(nn.Module):
    """
    标准 Causal Attention，不使用 window_size。
    每个 query 位置只能看到自己及之前的位置。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, context=None, mask=None) -> torch.Tensor:
        """
        x: [B, N, C], N 是序列长度
        """
        B, N, C = x.shape

        # 投影到 q,k,v
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 1, 3, 4)  # [3, B, N, H, D]
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, N, H, D]

        q = self.q_norm(q)
        k = self.k_norm(k)

        # 合并 batch 和 head => [B*H, N, D]
        q_ = q.permute(0, 2, 1, 3).reshape(B * self.num_heads, N, self.head_dim)
        k_ = k.permute(0, 2, 1, 3).reshape(B * self.num_heads, N, self.head_dim)
        v_ = v.permute(0, 2, 1, 3).reshape(B * self.num_heads, N, self.head_dim)

        # 计算 attention scores => [B*H, N, N]
        scores = torch.bmm(q_, k_.transpose(1, 2)) * (self.head_dim ** -0.5)

        # ===== 构造 causal mask =====
        causal_mask = torch.tril(torch.ones(N, N, device=x.device, dtype=torch.bool))  # [N, N]
        causal_mask = causal_mask.unsqueeze(0).expand(B * self.num_heads, -1, -1)       # [B*H, N, N]

        # 屏蔽未来帧
        scores.masked_fill_(~causal_mask, float('-inf'))

        # attention + dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        # attention 加权
        out_ = torch.bmm(attn, v_)  # [B*H, N, D]

        # reshape 回 [B, N, H, D]
        out_ = out_.reshape(B, self.num_heads, N, self.head_dim).permute(0, 2, 1, 3)

        # 合并 head 维度 => [B, N, C]
        out = out_.reshape(B, N, C)

        # 输出投影
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class LocalAttentionLayer(nn.Module):
    """
    一个简化的局部注意力层，不依赖 xFormers。
    在序列维度上，对每个 query 位置仅与附近 window_size 的位置进行注意力计算。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
        window_sizes: list = [10,],
    ) -> None:
        """
        参数说明:
          dim: 输入特征维度
          num_heads: 多头数量
          qkv_bias: 是否在 qkv 投影中使用 bias
          qk_norm: 若为 True，则对 q, k 分别做 LayerNorm
          window_sizes: 本地注意力窗口大小的list(例如 13 表示前后各 6 帧)
        """
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # 可选的对 q,k 做 norm
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.window_sizes = window_sizes

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, context=None, mask=None) -> torch.Tensor:
        """
        x: [B, N, C], 其中:
           B = batch_size
           N = 序列长度 (time dimension)
           C = embedding dim
        """
        B, N, C = x.shape

        # 1) 投影到 q,k,v => [B, N, 3, num_heads, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        # 2) 变换维度 => (3, B, N, num_heads, head_dim)，分离q,k,v
        qkv = qkv.permute(2, 0, 1, 3, 4)  # => [3, B, N, H, D]
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, N, H, D]

        # 可选：对 q, k 做 norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 3) 在本地窗口上做注意力
        multi_scale_outputs = []
        for window_size in self.window_sizes:
            attn_out = self.local_attention(q, k, v, window_size=window_size)
            multi_scale_outputs.append(attn_out)
        
        out = sum(multi_scale_outputs) / len(multi_scale_outputs)

        # 4) out => [B, N, H, D] => reshape回 [B, N, C]
        out = out.reshape(B, N, self.num_heads * self.head_dim)

        # 5) 输出投影
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def local_attention(self, q, k, v, window_size: int):
        """
        对 [B, N, H, D] 形状的 q,k,v，做局部窗口注意力:
          - 仅在 i +/- (window_size//2) 范围内进行注意力
          - 其余位置用 -inf 屏蔽
        返回 [B, N, H, D]
        """
        B, N, H, D = q.shape
        # 先把 [B, N, H, D] => [B*H, N, D] 方便做矩阵乘
        # 同时合并 batch 和 head
        q_ = q.permute(0, 2, 1, 3).reshape(B*H, N, D)  # [B*H, N, D]
        k_ = k.permute(0, 2, 1, 3).reshape(B*H, N, D)  # [B*H, N, D]
        v_ = v.permute(0, 2, 1, 3).reshape(B*H, N, D)  # [B*H, N, D]

        scale = D ** -0.5
        # scores => [B*H, N, N]
        scores = torch.matmul(q_, k_.transpose(-1, -2)) * scale

        # 构造局部窗口的 mask
        half_window = window_size // 2
        idx = torch.arange(N, device=q_.device)
        i_idx = idx.unsqueeze(1)  # [N,1]
        j_idx = idx.unsqueeze(0)  # [1,N]
        # 若 abs(i-j) > half_window，则屏蔽
        distance = (i_idx - j_idx).abs()
        local_mask = distance <= half_window  # [N,N], True=保留,False=屏蔽

        # 扩展到 [B*H, N, N]
        local_mask = local_mask.unsqueeze(0).expand(B*H, -1, -1)

        max_neg = -torch.finfo(scores.dtype).max
        scores.masked_fill_(~local_mask, max_neg)  # 屏蔽掉不在窗口内的位置

        attn = F.softmax(scores, dim=-1)  # [B*H, N, N]
        attn = self.attn_drop(attn)

        out_ = torch.matmul(attn, v_)  # [B*H, N, D]
        # reshape 回 [B, N, H, D]
        out_ = out_.reshape(B, H, N, D).permute(0, 2, 1, 3)
        return out_

class CrossAttention(nn.Module):
    """
    标准 Cross Attention 模块，支持 context 输入。
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int = None,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        context_dim = default(context_dim, query_dim)
        assert query_dim % num_heads == 0, 'query_dim must be divisible by num_heads'

        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(query_dim, query_dim, bias=qkv_bias)
        self.to_k = nn.Linear(context_dim, query_dim, bias=qkv_bias)
        self.to_v = nn.Linear(context_dim, query_dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(query_dim, query_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x:       [B, N, C]   -> query
        context: [B, M, C]   -> key/value (默认等于 x)
        mask:    [B, M]      -> 可选，对 key 进行 masking，True=保留，False=屏蔽
        """
        B, N, C = x.shape
        context = default(context, x)
        _, M, _ = context.shape

        # Linear Projection
        q = self.to_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]
        k = self.to_k(context).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, M, D]
        v = self.to_v(context).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, M, D]

        # Optionally normalize q, k
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Attention score: [B, H, N, M]
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Apply mask (optional)
        if exists(mask):
            mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, M]
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

        # Softmax + dropout
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.attn_drop(attn)

        # Attention-weighted sum: [B, H, N, D]
        out = torch.matmul(attn, v)

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, N, C)

        # Output projection
        out = self.proj(out)
        out = self.proj_drop(out)
        return out, attn


class CrossAttention2(nn.Module):
    """
    改进版 Cross Attention，支持返回注意力权重
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int = None,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = True,  # 默认开启归一化
        attn_drop: float = 0.1,  # 调整默认值
        proj_drop: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        return_attn: bool = True  # 新增返回注意力权重选项
    ):
        super().__init__()
        context_dim = context_dim or query_dim
        assert query_dim % num_heads == 0, 'query_dim必须能被num_heads整除'
        
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.return_attn = return_attn

        # 初始化线性投影层
        self.to_q = nn.Linear(query_dim, query_dim, bias=qkv_bias)
        self.to_k = nn.Linear(context_dim, query_dim, bias=qkv_bias)
        self.to_v = nn.Linear(context_dim, query_dim, bias=qkv_bias)

        # 归一化层
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        # 注意力机制
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(query_dim, query_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # # 参数初始化
        # self._init_weights()

    def _init_weights(self):
        """Xavier初始化增强稳定性"""
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        if self.to_q.bias is not None:
            nn.init.constant_(self.to_q.bias, 0)
        if self.to_k.bias is not None:
            nn.init.constant_(self.to_k.bias, 0)
        if self.to_v.bias is not None:
            nn.init.constant_(self.to_v.bias, 0)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, 
               query: torch.Tensor, 
               context: torch.Tensor = None, 
               mask: torch.Tensor = None) -> torch.Tensor:
        """
        输入:
            query:  [B, N, C]  -> phase tokens
            context: [B, M, C] -> 视频特征
            mask:   [B, M]     -> 时序mask
        输出:
            output: [B, N, C]
            attn_weights: [B, H, N, M] (当return_attn=True时)
        """
        B, N, C = query.shape
        context = context if context is not None else query
        _, M, _ = context.shape

        # 线性投影 + 多头拆分
        q = self.to_q(query).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, N, D]
        k = self.to_k(context).view(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, M, D]
        v = self.to_v(context).view(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, M, D]

        # 归一化处理
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 注意力分数计算
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, M]

        # 掩码处理
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, M]
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

        # 注意力权重计算
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights).contiguous()

        # 注意力加权求和
        output = (attn_weights @ v).permute(0, 2, 1, 3).contiguous()  # [B, N, H, D]
        output = output.view(B, N, C)  # [B, N, C]

        # 最终投影
        output = self.proj(output)
        output = self.proj_drop(output)

        return (output, attn_weights) if self.return_attn else output

class TopKCrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.,
        proj_drop=0.,
        prob_topk=None,
    ):
        super().__init__()
        context_dim = context_dim or query_dim
        assert query_dim % num_heads == 0, 'query_dim must be divisible by num_heads'

        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.prob_topk = prob_topk

        self.to_q = nn.Linear(query_dim, query_dim, bias=qkv_bias)
        self.to_k = nn.Linear(context_dim, query_dim, bias=qkv_bias)
        self.to_v = nn.Linear(context_dim, query_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(query_dim, query_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context=None, mask=None):
        B, N, C = x.shape
        context = context if context is not None else x
        _, M, _ = context.shape
        H, D = self.num_heads, self.head_dim

        q = self.to_q(x).view(B, N, H, D).transpose(1, 2)  # [B, H, N, D]
        k = self.to_k(context).view(B, M, H, D).transpose(1, 2)  # [B, H, M, D]
        v = self.to_v(context).view(B, M, H, D).transpose(1, 2)  # [B, H, M, D]

        scores = (q @ k.transpose(-1, -2)) * self.scale  # [B, H, N, M]

        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], float('-inf'))

        if self.prob_topk and self.prob_topk > 0:
            K = min(self.prob_topk, M)  # 防止K超过M

            if self.training:
                with torch.no_grad():
                    attn_prob = F.softmax(scores, dim=-1)  # [B, H, N, M]
                    
                    # 逐个query检查概率是否有效
                    invalid_mask = attn_prob.sum(dim=-1, keepdim=True) == 0  # [B,H,N,1]
                    attn_prob[invalid_mask.expand_as(attn_prob)] = 1.0 / M

                    idx = torch.multinomial(
                        attn_prob.view(-1, M), num_samples=K, replacement=False
                    ).view(B, H, N, K)

                # 创建mask并将未选中位置置为-inf
                mask_topk = torch.full_like(scores, float('-inf'))
                selected_scores = scores.gather(dim=-1, index=idx)
                mask_topk.scatter_(-1, idx, selected_scores)

            else:
                topk_vals, topk_idx = torch.topk(scores, K, dim=-1)
                mask_topk = torch.full_like(scores, float('-inf'))
                mask_topk.scatter_(-1, topk_idx, topk_vals)

            scores = mask_topk

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj_drop(self.proj(out))

        return out

class TemporalAdapter(nn.Module):
    def __init__(
        self, 
        input_channels: int, 
        intermed_channels: int, 
        num_heads: int = 4,
        window_sizes: list = [10], 
        prob_topk: int = 450,
        use_time_emb: bool = False,
        time_emb_type: str = 'sinusoidal',  # 可选 'sinusoidal' or 'learnable'
        max_time: int = 1000,
        attn_name: str = 'attn',
        proj_drop: float = 0.
    ):
        super().__init__()
        self.norm = nn.GroupNorm(1, input_channels)
        self.conv1 = nn.Conv2d(input_channels, intermed_channels, kernel_size=3, padding=1)
        self.conv_pool = nn.Conv2d(intermed_channels, intermed_channels, kernel_size=4, stride=4, padding=(1, 1))

        self.norm2 = nn.LayerNorm(intermed_channels)

        if attn_name == 'local_attn':
            self.attn_temp = LocalAttentionLayer(
                dim=intermed_channels,
                num_heads=num_heads,
                window_sizes=window_sizes,
                proj_drop=proj_drop,
            )
        elif attn_name == 'attn': 
            self.attn_temp = CrossAttention(
                query_dim=intermed_channels,
                num_heads=num_heads,
                proj_drop=proj_drop,
            )
        elif attn_name == 'topk_attn':
            self.attn_temp = TopKCrossAttention(
                query_dim=intermed_channels,
                num_heads=num_heads,
                prob_topk=prob_topk,
                proj_drop=proj_drop,
            )
        elif attn_name == 'causal_attn':
            self.attn_temp = CausalAttentionLayer(
                dim=intermed_channels,
                num_heads=num_heads,
                proj_drop=proj_drop,
            )
        else:
            raise NotImplementedError

        self.conv_upsample = nn.ConvTranspose2d(
            intermed_channels, intermed_channels, kernel_size=4, stride=4,
            padding=(1, 1), output_padding=(2, 1)
        )
        self.conv2 = nn.Conv2d(intermed_channels, input_channels, kernel_size=3, padding=1)

        self.use_time_emb = use_time_emb
        self.time_emb_type = time_emb_type  # 'sinusoidal' or 'learnable'
        if use_time_emb and time_emb_type == 'learnable':
            self.time_embedding = nn.Embedding(max_time, intermed_channels)

    def init_net(self):
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, time=None) -> torch.Tensor:
        """
        x: [B, C, T, H, W]
        time: [B, T] => 每一帧在原始视频中的位置
        """

        B, C, T, H, W = x.shape

        x = self.norm(x)
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv_pool(x)
        x = F.relu(x)
        H2, W2 = x.shape[-2], x.shape[-1]

        x = rearrange(x, '(b t) c h w -> (b h w) t c', b=B, t=T)  # [B*H*W, T, C]
        x_res = x
        x = self.norm2(x)

        # ==== 添加时间嵌入 ====
        if self.use_time_emb and time is not None:
            if self.time_emb_type == 'learnable':
                time_emb = self.time_embedding(time)  # [B, T, C]
            elif self.time_emb_type == 'sinusoidal':
                time_emb = get_sinusoid_encoding_table(time, x.shape[-1])  # [B, T, C]
            else:
                raise ValueError(f"Unsupported time_emb_type: {self.time_emb_type}")
            # repeat 到所有空间位置 => [B*H2*W2, T, C]
            time_emb = time_emb.repeat_interleave(H2 * W2, dim=0)
            x = x + time_emb


        x = self.attn_temp(x)
        x = x + x_res

        x = rearrange(x, '(b h w) t c -> (b t) c h w', b=B, h=H2, w=W2, t=T)
        x = self.conv_upsample(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', b=B, t=T)
        return x


class TemporalHorizontalPyramid(nn.Module):
    def __init__(self, dim, bin_num=None, num_heads=8):
        super().__init__()
        if bin_num is None:
            bin_num = [10, 5, 2]
        self.bin_num = bin_num
        self.dim = dim
        self.attn_layers = nn.ModuleList([CausalAttentionLayer(dim, num_heads=num_heads) for _ in bin_num])
        total_bins = sum(bin_num)
        # self.final_attn = CausalAttentionLayer(dim, num_heads=num_heads)
        self.conv1d = nn.Conv1d(dim * total_bins, dim, kernel_size=1)

    def forward(self, x):
        """
        x: [n, c, t, p]
        output: [n, c, k, p]
        """
        n, c, t, p = x.size()
        features = []

        for b, attn_layer in zip(self.bin_num, self.attn_layers):
            # Dynamically handle varying sequence lengths
            s = t // b
            z = x[:, :, :s*b, :]
            z = rearrange(z, 'n c (b s) p -> (n p b) s c', b=b)
            # Attention on temporal dimension within each bin
            z = attn_layer(z)  # [n*p*b, s, c]
            # Max pooling on temporal dimension
            z = torch.max(z, dim=1, keepdim=True)[0]  # [n*p*b, 1, c]
            z = rearrange(z, '(n p b) 1 c -> n p b c', n=n, p=p, b=b)
            features.append(z)

        # concatenate bins
        features = torch.cat(features, dim=2)  # [n, p, k, c]
        k = features.size(2)

        # # attention across bins (k dimension)
        # features = rearrange(features, 'n p k c -> (n p) k c')
        # features = self.final_attn(features)

        # # rearrange to original shape
        # features = rearrange(features, '(n p) k c -> n c k p', n=n, p=p)

        # # merge c and k, then conv1d
        # features = rearrange(features, 'n c k p -> n (c k) p')

        features = rearrange(features, 'n p k c -> n (c k) p')

        features = self.conv1d(features)  # [n, c, p]

        return features

class GMAdapter(nn.Module):
    def __init__(self, gmadapter_cfg):
        """
        :param in_channels: 输入通道数，对应特征中的 C
        :param num_heads: 时序自注意力头数
        :param dropout: dropout 概率
        """
        super(GMAdapter, self).__init__()
        # 这里的 embed_dim 就是 in_channels，因为我们对每个时间步的 C 维特征做 attention
        self.use_adative_scale = gmadapter_cfg['use_adative_scale']
        gmadapter_cfg.pop('use_adative_scale')
        self.temporal_attn = TemporalHorizontalPyramid(**gmadapter_cfg)
        if self.use_adative_scale:
            self.adapter_scale = nn.Parameter(torch.tensor(0.1))  # 可训练 or 固定

    def init_net(self):
        if self.use_adative_scale:
            nn.init.zeros_(self.adapter_scale)

    def forward(self, x_hpp, x_tp):
        """
        :param x_hpp: 输入张量，形状为 (N, C, S, P), x_tp: 形状为 (N, C, P)
        :return: 输出张量(N, C, P)
        """
        out = self.temporal_attn(x_hpp)
        if self.use_adative_scale:
            out = x_tp + self.adapter_scale * out
        else:
            out = x_tp + out
        return out

