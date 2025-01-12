import torch.nn as nn
import numpy as np
import torch
import math

class SelfAttention(nn.Module):
    def __init__(self, num_heads=8, d_model=512):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = self.d_model // self.num_heads
        self.qkv_layer = nn.Linear(self.d_model, 3*self.d_head)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        qkv = self.qkv_layer(x)
        q, k, v = np.split(qkv, 3, axis=-1)
        attn_out = self.softmax(q @ k.T / np.sqrt(self.d_head)) @ v
        return attn_out
    
class MultiHeadAttn(nn.Module):
    def __init__(self, num_heads=8, d_model=512):
        super(MultiHeadAttn, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.attn_heads = nn.Sequential(*[SelfAttention(num_heads=8, d_model=512) for _ in range(self.num_heads)])
    
    def forward(self, x):
        multi_attn_out = []
        for _ in range(self.num_heads):
            attn_out = self.attn(x)
            multi_attn_out.append(attn_out)
        multi_attn_out = np.vstack(multi_attn_out)
        return multi_attn_out
        
class EncoderBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super(EncoderBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.multi_head_attn = MultiHeadAttn(num_heads=num_heads, d_model=d_model)
        self.ff = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(normalized_shape=self.d_model)
    
    def forward(self, x):
        pass

class DecoderBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super(DecoderBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.multi_head_self_attn = MultiHeadAttn(num_heads=num_heads, d_model=d_model)
        self.enc_dec_attn = MultiHeadAttn(num_heads, d_model)
        self.ff = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        pass

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        coeff = torch.exp(-2 * torch.arange(embed_dim) * math.log(10000) / embed_dim)
        even_pos = torch.arange(0, max_len, 2)
        odd_pos = torch.arange(1, max_len, 2)
        self.pos_embedding = torch.zeros((max_len, embed_dim))
        self.pos_embedding[even_pos, :] = torch.sin(even_pos[:, np.newaxis] * coeff[np.newaxis, :])
        self.pos_embedding[odd_pos, :] = torch.cos(odd_pos[:, np.newaxis] * coeff[np.newaxis, :])
        self.register_buffer('positional_encoding', self.pos_embedding)
    
    def forward(self, x):
        return x + self.pos_embedding
        
class Transformer(nn.Module):
    def __init__(self, num_encoder=6, num_decoder=6, d_model=512, num_heads=8, src_vocab_size=1000, tgt_vocab_size=1000):
        super(Transformer, self).__init__()
        
        self.num_encoder = num_encoder
        self.num_decoder = num_decoder
        self.d_model = d_model
        self.num_heads = num_heads
        self.encoder = nn.Sequential(*[
            EncoderBlock(d_model=d_model, num_heads=num_heads)
            for _ in range(num_encoder)
        ])
        self.decoder = nn.Sequential(*[
            DecoderBlock(d_model=d_model, num_heads=num_heads)
            for _ in range(num_decoder)
        ])
        self.out = nn.Linear(d_model, tgt_vocab_size)
        self.src_embed = nn.Embedding(src_vocab_size, embedding_dim=d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, embedding_dim=d_model)
        self.positional_encoding = PositionalEncoding(embed_dim=d_model)
        
    def forward(self, x, y):
        print(x.shape, y.shape)
        
        
         
        