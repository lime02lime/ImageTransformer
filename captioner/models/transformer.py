import torch
from torch import nn


class AttentionBlock(nn.Module):
    # Aggregates information across the sequence independently for each feature
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        # K: hidden dim, usually K < D
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
        A = self.softmax(torch.einsum('bij, bkj -> bik', q,
                         k) / torch.sqrt(self.hidden_dim))
        # Shape is B x N x D_hidden
        Y = A@v
        return Y


class MultiHeadedAttentionBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_heads: int):
        super().__init__()
        self.H = num_heads
        self.heads = torch.nn.ModuleList(
            [AttentionBlock(in_dim, hidden_dim) for i in range(num_heads)])
        # assert hidden_dim%num_heads == 0
        self.linear = nn.Linear(hidden_dim, in_dim, bias=False)

    def forward(self, X):
        # Input is B x N x D
        # select number of heads such that Output is B x N x D
        head_outputs = torch.concatenate([h(X) for h in self.heads], dim=-1)
        return self.linear(head_outputs)

# class TransformerEncoder(torch.nn.module):
#     def __init__(self, num_heads):
#         self.H = num_heads
#         return

#     def single_head_attention(self, X: torch.tensor):
#         """ Applies attention mechanism on the input.

#         Shape parameters
#             B: batch size
#             N: number of tokens
#             D: dimension of each token

#         Args:
#             X (torch.tensor): shape B x N x D
#         """

#         return


if __name__ == '__main__':
    in_dim = 224
    num_heads = 12
    hidden_dim = int(in_dim/num_heads)
    batch_size = 1
    num_tokens = 16
    # To keep compute and number of parameters constant when changing the
    # number of heads k, the hidden Dh (Eq. 5) is typically set to D/k
    x = torch.randn(batch_size, num_tokens, in_dim)

    # attention_block = AttentionBlock(in_dim, hidden_dim)
    # A = attention_block(x)
    # print(A)

    multi_attention_block = MultiHeadedAttentionBlock(
        in_dim, hidden_dim, num_heads)
    print(multi_attention_block(x).shape)
