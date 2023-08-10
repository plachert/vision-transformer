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
    pass


class TransformerEncoder(nn.Module):
    pass


class ViT(nn.Module):
    pass


if __name__ == "__main__":
    a = ScaledDotProductAttention()
    q, k, v = [torch.rand(1, 3, 1) for _ in range(3)]
    print(q, k, v)
    print(a(q, k, v))