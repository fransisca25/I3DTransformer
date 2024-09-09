import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled += mask
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MultiheadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, max_seq_len, d_model = x.size()
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, max_seq_len, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.reshape(batch_size, max_seq_len, self.num_heads * self.head_dim)
        out = self.linear_layer(values)
        return out


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, num_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiheadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        residual_x = x
        x = self.attention(x, mask=mask)
        x = self.dropout1(x)
        x = self.norm1(x + residual_x)

        residual_x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual_x)
        return x


class TransformerEncoder(nn.Module):

    def __init__(self, d_model, d_ff, num_heads, dropout, num_layers):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderLayer(d_model, d_ff, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, x):
        x = self.layers(x)
        return x


class MultiheadCrossAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_layer = nn.Linear(d_model, d_model)
        self.kv_layer = nn.Linear(d_model, 2 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, y, mask=None):
        batch_size, max_seq_len, d_model = x.size()
        kv = self.kv_layer(y)
        q = self.q_layer(x)
        kv = kv.reshape(batch_size, max_seq_len, self.num_heads, 2 * self.head_dim)
        q = q.reshape(batch_size, max_seq_len, self.num_heads, self.head_dim)
        kv = kv.permute(0, 2, 1, 3)
        q = q.permute(0, 2, 1, 3)
        k, v = kv.chunk(2, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.reshape(batch_size, max_seq_len, self.num_heads * self.head_dim)
        out = self.linear_layer(values)
        return out


class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, num_heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.cross_attn = MultiheadCrossAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, qry, tgt, mask=None):
        tgt = tgt + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        tgt2 = self.cross_attn(qry, tgt, mask=mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.ffn(tgt)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        x, y = inputs
        for module in self._modules.values():
            y = module(x, y) 
        return y

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers=1):
        super().__init__()
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                          for _ in range(num_layers)])

    def forward(self, x, y):
        y = self.layers(x, y)
        return y


class Transformer(nn.Module):

    def __init__(self, d_model, d_ff, num_heads, dropout, num_layers):
        super().__init__()
        self.encoder = TransformerEncoder(d_model, d_ff, num_heads, dropout, num_layers)
        self.decoder = TransformerDecoder(d_model, d_ff, num_heads, dropout, num_layers)

    def forward(self, x, y):
        enc = self.encoder(x)
        out = self.decoder(enc, y)
        return out


