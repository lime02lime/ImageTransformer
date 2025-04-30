import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        assert emb_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, query, key=None, value=None, mask=None):
        if key is None:
            key = value = query
        B, N, D = query.shape
        q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            # mask: (B, 1, N, M) or (1, 1, N, M)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(attn_scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, emb_dim, ff_hidden_dim):
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Linear(emb_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, emb_dim)
        )
    def forward(self, x):
        return self.MLP(x)


class PositionalEncoding(nn.Module):
    def __init__(self, num_positions, emb_dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_positions, emb_dim))
    def forward(self, x):
        return x + self.pos_embedding[:, :x.size(1), :]


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, ff_hidden_dim):
        super().__init__()
        self.attn = MultiHeadSelfAttention(emb_dim, num_heads)
        self.ff = FeedForward(emb_dim, ff_hidden_dim)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ff(x))
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, ff_hidden_dim):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(emb_dim, num_heads)
        self.cross_attn = MultiHeadSelfAttention(emb_dim, num_heads)
        self.ff = FeedForward(emb_dim, ff_hidden_dim)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.norm3 = nn.LayerNorm(emb_dim)
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        x = self.norm1(tgt + self.self_attn(tgt, mask=tgt_mask))
        x = self.norm2(x + self.cross_attn(x, key=memory, value=memory, mask=memory_mask))
        x = self.norm3(x + self.ff(x))
        return x


class CustomTransformerEncoderDecoder(nn.Module):
    def __init__(self, patch_dim, emb_dim, num_heads, ff_hidden_dim, num_encoder_layers, num_decoder_layers, num_classes, num_patches, max_seq_length):
        super().__init__()
        # Encoder
        self.patch_proj = nn.Linear(patch_dim, emb_dim)
        self.pos_enc = PositionalEncoding(num_patches, emb_dim)
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(emb_dim, num_heads, ff_hidden_dim) for _ in range(num_encoder_layers)
        ])
        # Decoder
        self.tgt_embedding = nn.Embedding(num_classes, emb_dim)
        self.tgt_pos_enc = PositionalEncoding(max_seq_length, emb_dim)
        self.decoder_blocks = nn.ModuleList([
            TransformerDecoderBlock(emb_dim, num_heads, ff_hidden_dim) for _ in range(num_decoder_layers)
        ])
        self.classifier = nn.Linear(emb_dim, num_classes)
        self.max_seq_length = max_seq_length

    def encode(self, x):
        x = self.patch_proj(x)
        x = self.pos_enc(x)
        for block in self.encoder_blocks:
            x = block(x)
        return x

    def decode(self, tgt, memory):
        tgt_emb = self.tgt_embedding(tgt)
        tgt_emb = self.tgt_pos_enc(tgt_emb)
        seq_len = tgt_emb.size(1)
        tgt_mask = self.generate_tgt_mask(seq_len).to(tgt_emb.device)
        x = tgt_emb
        for block in self.decoder_blocks:
            x = block(x, memory, tgt_mask=tgt_mask)
        return x

    def forward(self, x, tgt):
        memory = self.encode(x)
        dec_out = self.decode(tgt, memory)
        return self.classifier(dec_out)

    def generate_tgt_mask(self, seq_len):
        # Causal mask: (1, 1, seq_len, seq_len)
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        return mask

    def generate(self, x, start_token, stop_token, device, max_len=20):
        self.eval()
        memory = self.encode(x)
        generated = [start_token]
        for _ in range(max_len):
            tgt = torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(0)  # (1, cur_len)
            dec_out = self.decode(tgt, memory)
            logits = self.classifier(dec_out)  # (1, cur_len, num_classes)
            next_token = logits[:, -1, :].argmax(-1).item()
            if next_token == stop_token:
                break
            generated.append(next_token)
        return generated[1:]  # Exclude start token
