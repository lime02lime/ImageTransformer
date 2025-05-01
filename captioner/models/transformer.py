"""
Manually coding Transformer attention blocks for the lolz.

Paper:
    Attention is All You Need (Mainly relevant for the Decoder)
    Vaswani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. ‘Attention Is All You
    Need’. In Advances in Neural Information Processing Systems, Vol. 30.
    Curran Associates, Inc., 2017.



    Vision Transformer (ViT), mainly useful for the encoder
    Dosovitskiy, Alexey, Lucas Beyer, Alexander Kolesnikov, Dirk
    Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, et al.
    ‘An Image Is Worth 16x16 Words: Transformers for Image Recognition at
    Scale’. arXiv, 3 June 2021. https://doi.org/10.48550/arXiv.2010.11929.

"""

import math

import torch
from torch import nn


def make_causal_attn_bias(size: int, device=None, float_dtype=torch.float):
    """
    Returns a [size x size] float mask with 0.0 on and below the diagonal,
    and -inf above. This can be added directly to attention logits.
    """
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
    return mask.masked_fill(mask == 1, float('-inf'))


class AttentionBlock(nn.Module):
    # Aggregates information across the sequence independently for each feature
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        # D_hidden or K: hidden dim, usually K < D
        self.hidden_dim = torch.tensor(hidden_dim, dtype=torch.int32)
        self.U_k = nn.Linear(in_dim, hidden_dim, bias=False)
        self.U_q = nn.Linear(in_dim, hidden_dim, bias=False)
        self.U_v = nn.Linear(in_dim, hidden_dim, bias=False)
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


class MultiHeadedSelfAttentionBlockSlow(nn.Module):
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


class MultiHeadedSelfAttentionBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim_per_head: int, num_heads: int):
        super().__init__()
        self.H = num_heads
        self.D_h = hidden_dim_per_head
        self.in_dim = in_dim
        self.Uqkv = nn.Parameter(
            torch.empty(in_dim, 3 * num_heads * hidden_dim_per_head),
        )
        # Xavier/ Glorot initialisation
        nn.init.xavier_uniform_(self.Uqkv)
        self.linear = nn.Linear(
            num_heads * hidden_dim_per_head, in_dim, bias=False,
        )
        # Normalise along query token dimension
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, X, is_causal=False):
        # B is the batch size
        # N is the number of tokens
        B, N, in_dim = X.shape
        assert in_dim == self.in_dim

        # Shapes are B x N x D_h*num_heads
        q, k, v = (X @ self.Uqkv).chunk(3, dim=-1)
        q = q.view(B, N, self.H, self.D_h).transpose(1, 2)  # (B, H, N, D_h)
        k = k.view(B, N, self.H, self.D_h).transpose(1, 2)  # (B, H, N, D_h)
        v = v.view(B, N, self.H, self.D_h).transpose(1, 2)  # (B, H, N, D_h)
        # Shape is B x H x N x N
        attn_scores = torch.matmul(
            q, k.transpose(-2, -1),
        ) / math.sqrt(self.D_h)

        if is_causal:
            # Create upper triangular mask of -infs
            causal_bias = make_causal_attn_bias(N, device=attn_scores.device)
            # This broadcasts across the batch and heads
            attn_scores = attn_scores + causal_bias

        A = self.softmax(attn_scores)
        # Shape is B x H x N x D_h
        head_outputs = A @ v
        # Final shape is back to B x N x D_in
        return self.linear(
            head_outputs.transpose(1, 2).contiguous()
            .view(B, N, -1),
        )


class MultiHeadedCrossAttentionBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim_per_head: int, num_heads: int):
        super().__init__()
        self.H = num_heads
        self.D_h = hidden_dim_per_head
        self.in_dim = in_dim
        # Learnable projections
        self.Uq = nn.Linear(
            in_dim, num_heads *
            hidden_dim_per_head, bias=False,
        )
        self.Uk = nn.Linear(
            in_dim, num_heads *
            hidden_dim_per_head, bias=False,
        )
        self.Uv = nn.Linear(
            in_dim, num_heads *
            hidden_dim_per_head, bias=False,
        )
        # Final projection
        self.linear = nn.Linear(
            num_heads * hidden_dim_per_head, in_dim, bias=False,
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X_dec, X_enc):
        """
        X shape (B x N_dec x D_in)
        X_enc shape (B x N_enc x D_in)

        N_enc: number of encoder tokens
        N_dec: number of decoder tokens (e.g. number of patches for a ViT)
        """
        B, N_dec, in_dim = X_dec.shape
        assert in_dim == self.in_dim
        _, N_enc, in_dim = X_enc.shape
        assert in_dim == self.in_dim

        # The tokens attend to the patches
        # Reshape into multiple heads
        q = self.Uq(X_dec).view(B, N_dec, self.H, self.D_h).transpose(
            1, 2,
        )  # (B, H, N_dec, D_h)
        k = self.Uk(X_enc).view(B, N_enc, self.H, self.D_h).transpose(
            1, 2,
        )  # (B, H, N_enc, D_h)
        v = self.Uv(X_enc).view(B, N_enc, self.H, self.D_h).transpose(
            1, 2,
        )  # (B, H, N_enc, D_h)
        # Shape is B x H x N_dec x N_enc
        A = self.softmax(
            torch.matmul(q, k.transpose(-2, -1)) /
            math.sqrt(self.D_h),
        )
        # Shape is B x H x N_dec x D_h
        head_outputs = torch.matmul(A, v)
        # Final shape is back to B x N_dec x D_in
        return self.linear(
            head_outputs.transpose(1, 2).contiguous()
            .view(B, N_dec, -1),
        )


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, activation):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, out_dim)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(
                f'Activation function {activation} not supported. '
                "Use 'relu' or 'gelu'.",
            )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Equation (2) in the Attention paper
        X = self.linear_2(self.activation(self.linear_1(X)))
        return X

# ------------------------------- Encoder blocks ------------------------------


class TransformerBlock(torch.nn.Module):
    """
    Layernorm (LN) is applied before every block,
    and residual connections after every block
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_heads: int,
        dim_feedforward: int,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(in_dim)
        self.msa = MultiHeadedSelfAttentionBlock(
            in_dim=in_dim,
            hidden_dim_per_head=hidden_dim//num_heads,
            num_heads=num_heads,
        )
        self.ln_2 = nn.LayerNorm(in_dim)
        self.mlp = MLP(in_dim, dim_feedforward, in_dim, 'gelu')

    def forward(self, X):
        # layer norm, msa and residual, Eq. (2) in ViT paper
        X = self.msa(self.ln_1(X)) + X
        # layer norm, mlp and residual, Eq. (3) in ViT paper
        X = self.mlp(self.ln_2(X)) + X
        return X


class TransformerEncoder(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_transformer_blocks: int,
        dim_feedforward: int,
    ):
        super().__init__()
        blocks = []
        for i in range(num_transformer_blocks):
            blocks.append(
                TransformerBlock(
                    hidden_dim, hidden_dim, num_heads, dim_feedforward,
                ),
            )
        self.blocks = nn.Sequential(*blocks)
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
            Y (torch.Tensor): shape B x N x D_hidden
        """
        X = self.blocks(X)
        return self.ln(X)

# ------------------------------- Decoder blocks ------------------------------


class TransformerDecoderBlock(torch.nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dim_feedforward: int):
        """
        Following the PyTorch interface
        https://pytorch.org/docs/stable/generated/
            torch.nn.TransformerDecoderLayer.html

        # Using notation from Vaswani et al. (2017) paper
        # in dim: d_model
        # set hidden_dim_per_head: d_model/h
        # out_dim: d_model
        # are recommended parameters s.t. num of parameters is constant w.r.t.
        # number of heads

        each of the layers in our encoder and decoder contains a fully connected
        feed-forward network, which is applied to each position separately and
        identically

        """
        super().__init__()

        self.msa = MultiHeadedSelfAttentionBlock(
            in_dim=hidden_dim,
            hidden_dim_per_head=hidden_dim//num_heads,
            num_heads=num_heads,
        )
        self.mca = MultiHeadedCrossAttentionBlock(
            in_dim=hidden_dim,
            hidden_dim_per_head=hidden_dim//num_heads,
            num_heads=num_heads,
        )

        # Normalises over last dimension (the feature dimension), for each token
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.ln_3 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, dim_feedforward, hidden_dim, 'gelu')

    def forward(self, X, X_enc):
        """
        X shape (B x N_tokens x D_in)
        X_enc shape (B x N_patches x D_enc)

        Shape parameters
            B: batch size
            N_outputs: number of tokens
            D: dimension of each token
            D_hidden: hidden dimension of the transformer

        # Implement masking for max-sequence length, a hyperparameter
        """
        # Masked multi-head (self) attention
        X = self.msa(self.ln_1(X), is_causal=True) + X
        # Multi-head (cross) attention with encoder outputs
        X = self.mca(self.ln_2(X), X_enc) + X
        # layer norm, mlp and residual, Eq. (3) in ViT paper
        X = self.mlp(self.ln_3(X)) + X
        return X


class TransformerDecoder(torch.nn.Module):
    """
    Vaswani et al. (2017) paper use
        num_layers N = 6
        hidden_dim d_model = 512
        dim_feedforward d_ff = 2048
        num_heads h = 8
        d_k = d_v = d_model/h = 64
        P_drop = 0.1
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(hidden_dim, num_heads, dim_feedforward)
            for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, X_dec: torch.Tensor, X_enc: torch.Tensor):
        """
        X_dec shape (B x M x D_h)
        X_enc shape (B x N x D_h)
        Y shape B x N x D_h
        """
        for block in self.blocks:
            X_dec = block(X_dec, X_enc)

        # TODO Should I put a LayerNorm here or no?
        return X_dec

# ------------------------------- Compositions -------------------------------


class TransformerCaptioner(torch.nn.Module):
    """
    API is here
    https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
    """

    def __init__(
        self,
        in_dim_enc: int,
        d_model: int,
        num_heads: int,
        seq_len_enc: int,
        seq_len_dec: int,
        num_layers: int,
        dim_feedforward: int,
        num_classes: int,
    ):

        # seq_len_dec = N_tokens is the (max) number of tokens in the decoder
        super().__init__()
        self.in_dim_enc = in_dim_enc

        self.linear_proj_enc = nn.Linear(in_dim_enc, d_model)
        self.linear_proj_dec = nn.Embedding(num_classes, d_model)

        # Fixed learnable position embeddings shape (1, N, d_model)
        self.positional_embeddings_enc = nn.Parameter(
            torch.empty(1, seq_len_enc, d_model),
        )
        self.positional_embeddings_dec = nn.Parameter(
            torch.empty(1, seq_len_dec, d_model),
        )
        # Xavier/ Glorot initialisation produced NaNs
        # now use normal
        nn.init.normal_(self.positional_embeddings_enc, std = 0.02)
        nn.init.normal_(self.positional_embeddings_dec, std = 0.02)

        self.enc = TransformerEncoder(
            d_model, num_heads, num_layers, dim_feedforward,
        )
        self.dec = TransformerDecoder(
            d_model, num_heads, num_layers, dim_feedforward,
        )

        self.output_proj = nn.Linear(d_model, num_classes)

    def forward(self, X, target=None):
        # X is the encoder input, shape B x N_patches x D_model
        # target is the decoder input, shape B x N_tokens 
        assert X.shape[2] == self.in_dim_enc, (
            f'input shape {X.shape} does not match in_dim_enc {self.in_dim_enc}'
        )

        if target is None:
            raise NotImplementedError(
                'target is None, not implemented yet. '
                'Use the decoder with teacher forcing.',
            )
        assert target.ndim == 2, (
            f'target shape {target.shape} is not 2'
        )
        # Encode image: out is B x N_patches x D_model
        X_enc = self.enc(
            self.linear_proj_enc(
                X,
            ) + self.positional_embeddings_enc,
        )
        # Generate captions using encoder output, out is B x N_tokens x D_model
        out = self.dec(
            self.linear_proj_dec(target) + self.positional_embeddings_dec,
            X_enc,
        )

        # Shape is B x N x num_classes
        scores = self.output_proj(out)
        return scores


class TransformerClassifier(torch.nn.Module):
    def __init__(
        self, in_dim: int, hidden_dim: int, num_heads: int,
        seq_len: int, num_classes, num_transformer_blocks: int,
    ):
        super().__init__()
        self.linear_proj = nn.Linear(in_dim, hidden_dim)
        # Fixed learnable position embeddings shape (1, N, hidden_dim)
        self.positional_embeddings = nn.Parameter(
            torch.empty(1, seq_len, hidden_dim),
        )
        # Xavier/ Glorot initialisation
        nn.init.xavier_uniform_(self.positional_embeddings)

        self.enc = TransformerEncoder(
            hidden_dim, num_heads, num_transformer_blocks,
        )
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
        # Linear project and add position encoding (patch encoding)
        X = self.linear_proj(X) + self.positional_embeddings
        X = self.enc.forward(X)
        # Choose first token for classification, assume is <CLS> token
        y = self.classification_head(X)[:, 0, :]
        return y


if __name__ == '__main__':
    from torch.profiler import ProfilerActivity, profile, record_function

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,        # record tensor shapes
        profile_memory=True,       # track allocation / memory usage
        with_stack=True,            # capture Python call stack (optional)
    ) as prof:
        with record_function('model_inference'):
            # To keep compute and number of parameters constant when changing the
            # number of heads k, the hidden Dh (Eq. 5) is typically set to D/k

            # in_dim = 49
            # num_heads = 7
            # hidden_dim = int(in_dim/num_heads)
            # batch_size = 2
            # num_tokens = 16
            # num_patches = 32

            # x = torch.randn(batch_size, num_patches, in_dim)

            # multi_attention_block = MultiHeadedSelfAttentionBlock(
            #     in_dim, hidden_dim, num_heads,
            # )
            # in_dim = 64
            # num_classes = 10

            # print(multi_attention_block(x, is_causal=True).shape)
            # x_enc = torch.randn(batch_size, num_patches, in_dim)
            # x_dec = torch.randn(batch_size, num_tokens, in_dim)
            # # mhca =  MultiHeadedCrossAttentionBlock(in_dim = in_dim, hidden_dim_per_head=16, num_heads=4)
            # # mhca(x_dec, x_enc)

            # Testing the decoder block
            def test_decoder_block():
                in_dim = 49
                num_heads = 7
                hidden_dim = int(in_dim/num_heads)
                batch_size = 2
                num_patches = 32
                num_tokens = 16
                x_enc = torch.randn(batch_size, num_patches, hidden_dim)
                x_dec = torch.randn(batch_size, num_tokens, hidden_dim)
                decoder_block = TransformerDecoderBlock(hidden_dim, num_heads)
                decoder_block(x_dec, x_enc)

            # test_decoder_block()

            def test_captioner():
                in_dim_enc = 49
                in_dim_dec = 1
                d_model = 256
                num_heads = 8
                # hidden_dim = d_model//num_heads
                batch_size = 2
                num_classes = 10
                seq_len_enc = 32
                seq_len_dec = 16
                num_layers = 6
                dim_feedforward = 512

                x_enc = torch.randn(batch_size, seq_len_enc, in_dim_enc)
                x_dec = torch.randint(
                    0, 12, (batch_size, seq_len_dec, in_dim_dec),
                )

                captioner = TransformerCaptioner(
                    in_dim_enc,
                    in_dim_dec,
                    d_model,
                    num_heads,
                    seq_len_enc,
                    seq_len_dec,
                    num_layers,
                    dim_feedforward,
                    num_classes,
                )
                out = captioner(x_enc, x_dec)
                print(out.shape)

            test_captioner()

    # Print a table of top 10 slowest ops
    print(
        prof.key_averages().table(
            sort_by='self_cpu_time_total',  # or self_cpu_time_total
            row_limit=10,
        ),
    )

    # attention_block = AttentionBlock(in_dim, hidden_dim)
    # print(attention_block(x).shape)

    # multi_attention_block = MultiHeadedSelfAttentionBlockSlow(
    #     in_dim, hidden_dim, num_heads,
    # )
    # print(multi_attention_block(x).shape)

    # transformer_block = TransformerBlock(in_dim, hidden_dim, num_heads)
    # print(transformer_block(x).shape)

    # transformer_encoder = TransformerEncoder(
    #     in_dim, hidden_dim, num_heads,
    # )
    # assert transformer_encoder(x).shape == (batch_size, hidden_dim)

    # transformer_classifier = TransformerClassifier(
    #     in_dim, hidden_dim, num_heads, num_tokens, num_classes,
    # )
    # assert transformer_classifier(x).shape == (batch_size, num_classes)

    # print("Trainable parameters:",
    #       count_trainable_params(transformer_classifier))
