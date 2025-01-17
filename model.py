import torch.nn as nn
import torch
import math

class FeedForward(nn.Module):
    def __init__(self, d_model : int, ff_dim : int, dropout: float = 0.3):
        super().__init__()
        self.l1 = nn.Linear(d_model, ff_dim)
        self.l2 = nn.Linear(ff_dim, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.dropout(x)
        x = self.l2(x)
        return x

class ResidualConnection(nn.Module):
    def __init__(self, d_model : int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.d_model)
    
    def forward(self, x, sublayer):
        sublayer_out = sublayer(self.norm(x))
        return x + self.dropout(sublayer_out)
        
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads : int, d_model : int, dropout : int):
        super().__init__()
        assert d_model % num_heads == 0, "d model should be divisible by num heads"
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.d_head = d_model // num_heads
        self.d_model = d_model
        self.num_heads = num_heads
    
    @staticmethod
    def attention(q, k, v, mask, dropout: nn.Dropout):
        dk = q.shape[-1]
        attention_scores = (q @ k.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        attention_out = attention_scores @ v
        return attention_out, attention_scores
    
    def forward(self, q, k, v, mask=None):
        qval = self.wq(q)  # [b, seq len, d model]
        kval = self.wk(k)
        vval = self.wv(v)
        
        qval = qval.view((qval.shape[0], qval.shape[1], self.num_heads, self.d_head)).transpose(1, 2)  # [b, num heads, seq len, d head]
        kval = kval.view((kval.shape[0], kval.shape[1], self.num_heads, self.d_head)).transpose(1, 2)
        vval = vval.view((vval.shape[0], vval.shape[1], self.num_heads, self.d_head)).transpose(1, 2)
        
        x, self.attention_scores = MultiHeadAttention.attention(qval, kval, vval, mask, self.dropout)
        
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)
        x = self.wo(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout = 0.1):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        num_tokens = x.shape[1]
        x = x + self.pe[:, :num_tokens, :].requires_grad_(False)
        return self.dropout(x)

class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
        
class EncoderBlock(nn.Module):
    def __init__(self, d_model : int, self_attention_block : MultiHeadAttention, ff_block: FeedForward, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.self_attention_block = self_attention_block
        self.ff_block = ff_block
        self.residualblock = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)])
    
    def forward(self, x, src_mask):
        x = self.residualblock[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residualblock[1](x, self.ff_block)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model : int, layers : nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, self_attention_block : MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForward, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residualblock = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(3)])
        self.dropout = dropout
    
    def forward(self, x, enc_out, src_mask, tgt_mask):
        x = self.residualblock[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residualblock[1](x, lambda x: self.cross_attention_block(x, enc_out, enc_out, src_mask))
        x = self.residualblock[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, d_model : int, layers : nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, enc_out, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.layer = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        return self.layer(x)

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection = projection
    
    def encode(self, x, src_mask):
        src = self.src_embed(x)
        src = self.src_pos(src)
        src = self.encoder(src, src_mask)
        return src

    def decode(self, enc_out, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, enc_out, src_mask, tgt_mask)

    def project(self, x):
        return self.projection(x)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len : int, tgt_seq_len: int, d_model : int = 512,  N : int = 6, H : int = 8, dropout : float = 0.2, ff_dim : int = 2048):
    src_embedding = InputEmbedding(d_model, src_vocab_size)
    tgt_embedding = InputEmbedding(d_model, tgt_vocab_size)
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(H, d_model, dropout)
        feed_forward_block = FeedForward(d_model, ff_dim, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    decoder_blocks = []
    for _ in range(N):
        dec_self_attn = MultiHeadAttention(H, d_model, dropout)
        dec_cross_attn = MultiHeadAttention(H, d_model, dropout)
        dec_ff_block = FeedForward(d_model, ff_dim, dropout)
        dec_block = DecoderBlock(d_model, dec_self_attn, dec_cross_attn, dec_ff_block, dropout)
        decoder_blocks.append(dec_block)
    
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    projection = ProjectionLayer(d_model, tgt_vocab_size)
    transformer = Transformer(encoder, decoder, src_embedding, tgt_embedding, src_pos, tgt_pos, projection)
    
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer        