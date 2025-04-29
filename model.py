import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        assert emb_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        self.qkv_proj = nn.Linear(emb_dim, emb_dim * 3)
        self.out_proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        B, N, D = x.shape  # (batch_size, sequence_length, embedding_dim)

        # Project to queries, keys, and values
        qkv = self.qkv_proj(x)  # Shape: (B, N, 3*D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, num_heads, N, head_dim)

        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, num_heads, N, N)
        attn = torch.softmax(scores, dim=-1)  # Attention weights
        out = attn @ v  # (B, num_heads, N, head_dim)

        # Concatenate heads
        out = out.transpose(1, 2).reshape(B, N, D)  # (B, N, D)

        # Final linear projection
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

class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, ff_hidden_dim):
        super().__init__()
        self.attn = MultiHeadSelfAttention(emb_dim, num_heads)
        self.ff = FeedForward(emb_dim, ff_hidden_dim)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x):
        # Self-attention + residual connection + norm
        x = self.norm1(x + self.attn(x))
        # Feedforward + residual connection + norm
        x = self.norm2(x + self.ff(x))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, num_patches, emb_dim):
        super().__init__()
        # Learnable positional embedding for each patch
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, emb_dim))

    def forward(self, x):
        # x: (B, num_patches, emb_dim)
        return x + self.pos_embedding

class CustomTransformerEncoder(nn.Module):
    def __init__(self, patch_dim, emb_dim, num_heads, ff_hidden_dim, num_layers, num_classes, num_patches):
        super().__init__()

        # Project patch to embedding dimension
        self.patch_proj = nn.Linear(patch_dim, emb_dim)
        self.pos_enc = PositionalEncoding(num_patches, emb_dim)

        # Stacked transformer blocks
        self.encoder_blocks = nn.Sequential(
            *[TransformerEncoderBlock(emb_dim, num_heads, ff_hidden_dim) for _ in range(num_layers)]
        )

        # Classification head
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        # x: (B, num_patches, patch_dim)
        x = self.patch_proj(x)  # (B, num_patches, emb_dim)
        x = self.pos_enc(x)     # Add positional encodings
        x = self.encoder_blocks(x)  # (B, num_patches, emb_dim)

        x = x.mean(dim=1)  # Global average pooling across patches
        return self.classifier(x)
