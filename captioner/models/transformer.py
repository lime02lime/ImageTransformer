"""

Manually coding Transformer attention blocks

Paper:
    Dosovitskiy, Alexey, Lucas Beyer, Alexander Kolesnikov, Dirk
    Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, et al.
    ‘An Image Is Worth 16x16 Words: Transformers for Image Recognition at
    Scale’. arXiv, 3 June 2021. https://doi.org/10.48550/arXiv.2010.11929.

"""


import torch
from torch import nn


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AttentionBlock(nn.Module):
    # Aggregates information across the sequence independently for each feature
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        # D_hidden or K: hidden dim, usually K < D
        self.hidden_dim = torch.tensor(hidden_dim, dtype=torch.int32)
        self.U_q = nn.Linear(in_dim, hidden_dim, bias=False)
        self.U_v = nn.Linear(in_dim, hidden_dim, bias=False)
        self.U_k = nn.Linear(in_dim, hidden_dim, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """ Applies attention mechanism on the input.

        Shape parameters
            B: batch size
            N: number of tokens
            D: dimension of each token

        Args:
            X (torch.Tensor): shape B x N x D

        Returns:
            Y (torch.Tensor): shape B x N x D_hidden
        """
        # Shape B x N x K
        q = self.U_q(X)
        k = self.U_k(X)
        v = self.U_v(X)
        # Shape is B x N x N
        A = self.softmax(
            torch.einsum(
                'bij, bkj -> bik', q,
                k,
            ) / torch.sqrt(self.hidden_dim),
        )
        # Shape is B x N x D_hidden
        Y = A@v
        return Y


class MultiHeadedAttentionBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim_per_head: int, num_heads: int):
        super().__init__()
        self.H = num_heads
        self.heads = torch.nn.ModuleList(
            [
                AttentionBlock(in_dim, hidden_dim_per_head)
                for i in range(num_heads)
            ],
        )
        self.linear = nn.Linear(
            num_heads * hidden_dim_per_head, in_dim, bias=False,
        )

    def forward(self, X):
        # Input is B x N x D
        # select number of heads such that Output is B x N x D
        head_outputs = torch.concatenate([h(X) for h in self.heads], dim=-1)
        return self.linear(head_outputs)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, out_dim)
        self.activation = nn.GELU()
        return

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.linear_2(self.activation(self.linear_1(X)))
        return X


class TransformerBlock(torch.nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int, num_heads: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(in_dim)
        self.msa = MultiHeadedAttentionBlock(in_dim, hidden_dim, num_heads)
        self.ln_2 = nn.LayerNorm(in_dim)
        self.mlp = MLP(in_dim, in_dim, in_dim)

    def forward(self, X):
        # layer norm, msa and residual, Eq. (2) in ViT paper
        X = self.msa(self.ln_1(X)) + X
        # layer norm, mlp and residual, Eq. (3) in ViT paper
        X = self.mlp(self.ln_2(X)) + X
        return X


class TransformerEncoder(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_heads: int, seq_len: int):
        super().__init__()
        self.linear_proj = nn.Linear(in_dim, hidden_dim)
        self.block_1 = TransformerBlock(hidden_dim, hidden_dim, num_heads)
        self.block_2 = TransformerBlock(hidden_dim, hidden_dim, num_heads)
        self.block_3 = TransformerBlock(hidden_dim, hidden_dim, num_heads)

        # Fixed learnable position embeddings shape (1, N, hidden_dim)
        self.positional_embeddings = nn.Parameter(
            torch.empty(1, seq_len, hidden_dim),
        )
        # Xavier/ Glorot initialisation
        nn.init.xavier_uniform_(self.positional_embeddings)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, X: torch.Tensor):
        """ Applies the transformer encoder

        Shape parameters
            B: batch size
            N: number of tokens
            D: dimension of each token
            D_hidden: hidden dimension of the transformer

        Args:
            X (torch.Tensor): shape B x N x D

        Returns:
            Y (torch.Tensor): shape B x D_hidden
        """
        # Linear project and add position encoding (patch encoding)
        X = self.linear_proj(X) + self.positional_embeddings
        X = self.block_1(X)
        X = self.block_2(X)
        X = self.block_3(X)
        # Choose first token, Eq. (4) in ViT paper
        y = self.ln(X[:, 0, :])
        return y


class TransformerClassifier(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_heads: int, seq_len: int, num_classes):
        super().__init__()
        self.enc = TransformerEncoder(in_dim, hidden_dim, num_heads, seq_len)
        self.classification_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, X):
        """ Applies the transformer encoder and uses it to return scores for
            the classification problem.

        Shape parameters
            B: batch size
            N: number of tokens
            D: dimension of each token
            D_hidden: hidden dimension of the transformer

        Args:
            X (torch.Tensor): shape B x N x D

        Returns:
            Y (torch.Tensor): shape B x num_classes
        """
        X = self.enc.forward(X)
        y = self.classification_head(X)
        return y


if __name__ == '__main__':
    # To keep compute and number of parameters constant when changing the
    # number of heads k, the hidden Dh (Eq. 5) is typically set to D/k
    in_dim = 49
    num_heads = 7
    hidden_dim = int(in_dim/num_heads)
    batch_size = 2
    num_tokens = 16
    num_classes = 10

    x = torch.randn(batch_size, num_tokens, in_dim)

    attention_block = AttentionBlock(in_dim, hidden_dim)
    print(attention_block(x).shape)

    multi_attention_block = MultiHeadedAttentionBlock(
        in_dim, hidden_dim, num_heads,
    )
    print(multi_attention_block(x).shape)

    transformer_block = TransformerBlock(in_dim, hidden_dim, num_heads)
    print(transformer_block(x).shape)

    transformer_encoder = TransformerEncoder(
        in_dim, hidden_dim, num_heads, seq_len=num_tokens,
    )
    assert transformer_encoder(x).shape == (batch_size, hidden_dim)

    transformer_classifier = TransformerClassifier(
        in_dim, hidden_dim, num_heads, num_tokens, num_classes,
    )
    assert transformer_classifier(x).shape == (batch_size, num_classes)

    print("Trainable parameters:", 
          count_trainable_params(transformer_classifier))

