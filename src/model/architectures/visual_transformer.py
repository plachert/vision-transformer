import torch.nn as nn
import torch
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    pass


class ScaledDotProductAttention(nn.Module):
    """Scaled dot product attention without mask."""
    def __init__(self):
        super().__init__()
        
    def forward(self, q, k, v):
        d_k = q.shape[-1]
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) # matmul Q, K
        attention_scores_scaled = attention_scores / torch.sqrt(torch.tensor(d_k)) # Scale
        attention_weights  = F.softmax(attention_scores_scaled, -1) # softmax
        attention = torch.matmul(attention_weights, v) # matmul weights and V
        return attention 


class MultiHeadAttention(nn.Module):
    def __init__(
        self, 
        emb_dim: int, 
        num_heads: int = 8,
        ):
        super().__init__()
        assert emb_dim % num_heads == 0
        self.head_dim = emb_dim // num_heads
        self.num_heads = num_heads
        # Q, K, V
        self.q_transform = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim, bias=False) for _ in range(num_heads)])
        self.k_transform = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim, bias=False) for _ in range(num_heads)])
        self.v_transform = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim, bias=False) for _ in range(num_heads)])
        # Scaled Dot-Product Attention
        self.scaled_attention = ScaledDotProductAttention()
        # Linear projection (last layer)
        self.linear_projection = nn.Linear(emb_dim, emb_dim)
        self._init_params()
        
    def _init_params(self):
        pass
    
    def _chunk_qkv(self, q, k, v):
        n, _, _ = q.size()
        q = q.view(n, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(n, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(n, -1, self.num_heads, self.head_dim).transpose(1, 2)
        return q, k, v
    
    def forward(self, q, k, v):
        # TODO: vectorize
        q, k, v = self._chunk_qkv(q, k, v)
        Q = [net(q[:, idx, ...]) for idx, net in enumerate(self.q_transform)]
        K = [net(k[:, idx, ...]) for idx, net in enumerate(self.k_transform)]
        V = [net(v[:, idx, ...]) for idx, net in enumerate(self.v_transform)]
        attentions = [self.scaled_attention(Q[idx], K[idx], V[idx]) for idx in range(self.num_heads)]
        attention = torch.cat(attentions, dim=-1)
        return attention


class TransformerEncoder(nn.Module):
    pass


class ViT(nn.Module):
    pass


if __name__ == "__main__":
    q, k = [torch.rand(1, 3, 256) for _ in range(2)]
    v = torch.rand(1, 3, 256)
    a = MultiHeadAttention(256)
    print(a(q, k , v).shape)
    # sequence = torch.rand(1, 10, 3, 3)
    # m = MultiHeadAttention(3, 3)
    # print(m(sequence).shape)