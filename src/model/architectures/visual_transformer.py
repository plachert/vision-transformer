"""This module provides implementation of Visual Transformer described in https://arxiv.org/pdf/2010.11929v2.pdf."""

import torch.nn as nn
import torch
import torch.nn.functional as F


class PatchProjection(nn.Module):
    def __init__(self, patch_shape=(3, 16, 16), emb_size=256):
        super().__init__()
        self.emb_size = emb_size
        self.patch_c = patch_shape[0]
        self.patch_h = patch_shape[1]
        self.patch_w = patch_shape[2]
        self.conv = nn.Conv2d(
                in_channels=patch_shape[0], 
                out_channels=emb_size, 
                kernel_size=patch_shape[1:], 
                stride=patch_shape[1:],
            )
        
    def get_num_patches(self, image):
        """Return number of patches across height and width along with batch size."""
        n, c, h, w = image.size()
        assert c == self.patch_c, f"{self.patch_c} channels expected"
        n_h = h // self.patch_h
        n_w = w // self.patch_w
        assert h % self.patch_h == 0, "Image height not divisible by patch height"
        assert w % self.patch_w == 0, "Image width not divisible by patch width"
        return n, n_h, n_w
        
    
    def forward(self, image):
        n, n_h, n_w = self.get_num_patches(image)
        projection = self.conv(image) # N, emb_size, n_h, n_w
        projection_flattened = projection.reshape(n, self.emb_size, n_h * n_w) # N, emb_size, seq_len
        out = projection_flattened.permute(0, 2, 1) # N, seq_len, emb_size
        return out


class ClassToken(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.token = nn.Parameter(torch.randn(1, 1, emb_dim), requires_grad=True)
        
    def forward(self, sequence):
        n, _, _ = sequence.size()
        batch_token = self.token.repeat(n, 1, 1)
        sequence = torch.cat([batch_token, sequence], dim=1)
        return sequence


class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim: int, sequence_len: int):
        super().__init__()
        self.positions = nn.Parameter(torch.randn(1, sequence_len, emb_dim), requires_grad=True)
        
    def forward(self, sequence):
        embeddings = sequence + self.positions
        return embeddings


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
        for q_l, k_l, v_l in zip(self.q_transform, self.k_transform, self.v_transform):
            nn.init.xavier_uniform_(q_l.weight)
            nn.init.xavier_uniform_(k_l.weight)
            nn.init.xavier_uniform_(v_l.weight)
    
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
        projeted_attention = self.linear_projection(attention)
        return projeted_attention


class EncoderBlock(nn.Module):
    def __init__(
        self, 
        emb_dim, 
        num_heads, 
        dim_feedforward,
        dropout=0.0,
        ):
        super().__init__()
        self.attn = MultiHeadAttention(emb_dim, num_heads)
        self.feedforward = nn.Sequential(
            nn.Linear(emb_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(dim_feedforward, emb_dim), 
            nn.Dropout(dropout),
        )
        self.norm_1 = nn.LayerNorm(emb_dim)
        self.norm_2 = nn.LayerNorm(emb_dim)
        
    def forward(self, sequence):
        sequence = self.norm_1(sequence) # pre-norm
        sequence = sequence + self.attn(q=sequence, k=sequence, v=sequence) # 1st skip connection
        sequence = self.norm_2(sequence) # pre-norm
        sequence = sequence + self.feedforward(sequence) # 2nd skip connection
        return sequence
        

class TransformerEncoder(nn.Module):
    def __init__(self, num_blocks, emb_dim, num_heads, dim_feedforward, dropout):
        super().__init__()
        block_params = {"emb_dim": emb_dim, "num_heads": num_heads, "dim_feedforward": dim_feedforward, "dropout": dropout}
        self.encode = nn.Sequential(*[EncoderBlock(**block_params) for _ in range(num_blocks)])
    
    def forward(self, sequence):
        encoded = self.encode(sequence)
        return encoded
        

class VisionTransformer(nn.Module):
    def __init__(self, emb_dim, patch_shape, num_blocks, num_heads, num_classes, num_patches, dim_feedforward, dropout):
        super().__init__()
        self.patch_projection = PatchProjection(patch_shape, emb_dim)
        self.pos_encoding = PositionalEncoding(emb_dim, num_patches)
        self.class_token = ClassToken(emb_dim)
        self.transformer_encoder = TransformerEncoder(
            num_blocks=num_blocks, 
            emb_dim=emb_dim, 
            num_heads=num_heads, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, num_classes)
        )
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, image):
        sequence = self.patch_projection(image)
        sequence = self.pos_encoding(sequence)
        sequence = self.class_token(sequence)
        encoded = self.transformer_encoder(sequence)
        class_token = encoded[:, 0, :]
        out = self.mlp_head(class_token)
        return out
    
if __name__ == "__main__":
    import torch
    image = torch.rand(1, 3, 224, 224)
    patch_prj = PatchProjection()
    print(patch_prj(image).shape)

    m = VisionTransformer(256, (3, 16, 16), 6, 8, 10, 14*14, 256, 0)
    print(m(torch.rand(1, 3, 224, 224)).shape)