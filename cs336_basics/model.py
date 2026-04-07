import torch
import torch.nn as nn
import numpy as np
import einops
import matplotlib.pyplot as plt


def count_params(module, trainable_only=False, verbose=True):
    """
    Count parameters in a PyTorch module and format in M or B.

    Args:
        module: nn.Module
        trainable_only: if True, count only parameters with requires_grad=True

    Returns:
        (raw_count, formatted_string)
    """
    if trainable_only:
        n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    else:
        n_params = sum(p.numel() for p in module.parameters())

    # Format nicely
    if n_params >= 1e9:
        formatted = f"{n_params / 1e9:.2f}B"
    elif n_params >= 1e6:
        formatted = f"{n_params / 1e6:.2f}M"
    elif n_params >= 1e3:
        formatted = f"{n_params / 1e3:.2f}K"
    else:
        formatted = str(n_params)

    if verbose:
        print(f"Number of params in {module.__class__.__name__}: {formatted}")
    else:
        return n_params, formatted


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None, sigma_scale=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        # Define W matrix (with careful initialization)
        sigma = sigma_scale * np.sqrt(2 / (out_features + in_features))
        data = nn.init.trunc_normal_(
            torch.empty(out_features, in_features, dtype=dtype, device=device), mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma,
        )
        self.W = nn.Parameter(data, requires_grad=True)

    def forward(self, x):
        """
        x: [..., d_in]
        y: [..., d_out]
        """
        return einops.einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        # Initialize embedding matrix
        data = nn.init.trunc_normal_(
            torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device), mean=0.0, std=1., a=-3, b=3,
        )
        self.E = nn.Parameter(data, requires_grad=True)

    def forward(self, token_ids):

        # # Basic sanity checks
        # assert token_ids.max() < self.num_embeddings
        # assert token_ids.min() > -1

        # Pick out the given indices
        x = self.E[token_ids]
        return x


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        # Initialise parameters g
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = (((x ** 2).mean(dim=-1) + self.eps) ** -0.5)
        rms = einops.repeat(rms, "b t -> b t d", d=self.d_model)
        result = rms * x * self.g 

        return result.to(in_dtype)


def silu(x):
    return x * torch.sigmoid(x)


def relu(x):
    return torch.threshold(x, 0, 0)


class FFNSwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        if d_ff is None:
            # Default value
            d_ff = int(8 * d_model / 3)
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        # Initialize parameters
        self.W1 = Linear(d_model, d_ff, device, dtype)
        self.W3 = Linear(d_model, d_ff, device, dtype)
        self.W2 = Linear(d_ff, d_model, device, dtype)

    def forward(self, x):
        return self.W2(silu(self.W1(x)) * (self.W3(x)))


class FFNSiLU(nn.Module):
    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        if d_ff is None:
            # Default value
            d_ff = int(8 * d_model / 3)
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        # Initialize parameters
        self.W1 = Linear(d_model, d_ff, device, dtype)
        self.W2 = Linear(d_ff, d_model, device, dtype)

    def forward(self, x):
        return self.W2(silu(self.W1(x)))


def rope_angles(theta, d_k, max_seq_len):
    a = torch.arange(max_seq_len)
    # b = 1. / (theta ** ((2 * torch.arange(d_k // 2) - 2) / d_k))
    b = 1. / (theta ** ((2 * torch.arange(1, d_k // 2 + 1) - 2) / d_k))
    return einops.einsum(a, b, "a, b -> a b")


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # Register angle values to be used later
        angles = rope_angles(theta, d_k, max_seq_len)
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        # [T d/2 2 2]: [i, k] is 2x2 rotation matrix
        values = einops.rearrange(
            torch.stack([cos, -sin, sin, cos], dim=-1),
            "t d_half (a b) -> t d_half a b", a=2, b=2,
        )
        self.register_buffer("values", values, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # Expand token_positions to match x's batch dims (minus d)
        for _ in range(x.ndim - 1 - token_positions.ndim):
            token_positions = token_positions.unsqueeze(-2)
        token_positions = token_positions.expand(x.shape[:-1])

        assert x.shape[:-1] == token_positions.shape

        # [... T d] -> [... T d/2 2]
        x_ = einops.rearrange(x, "... t (b c) -> ... t b c", b=self.d_k//2, c=2)

        # Take slice of the position embedding matrix according to token_positions
        rot = self.values[token_positions]

        # [... T d/2 2 2] x [... T d/2 2] -> [... T d/2 2]
        x_rot = einops.einsum(rot, x_, "... t d_half a_out a_in, ... t d_half a_in -> ... t d_half a_out")
        
        # [... T d/2 2] -> [... T d]
        x_rot = einops.rearrange(x_rot, "... t d_half a_out -> ... t (d_half a_out)")

        return x_rot


def softmax(x, dim):
    """Applies softmax to x along dim dimension."""
    z = torch.exp(x - x.max(dim=dim, keepdim=True).values)
    return z / z.sum(dim=dim, keepdim=True)



def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Args:
        q (torch.Tensor): [batch_size, ..., seq_len, d_k]
        k (torch.Tensor): [batch_size, ..., seq_len, d_k]
        v (torch.Tensor): [batch_size, ..., seq_len, d_v]
        mask (torch.Tensor): apply attention to only entries where mask=True
    """
    if mask is None:
        # By default, apply a causal mask
        mask = torch.triu(torch.ones(q.shape[-2], k.shape[-2]), diagonal=1).to(torch.bool)
        for _ in range(q.ndim - 1 - mask.ndim):
            mask = mask.unsqueeze(-2)
        mask = mask.expand(q.shape[:-1])


    x = ((q.shape[-1]) ** -0.5) * einops.einsum(q, k, "b ... t_q d_k, b ... t_k d_k -> b ... t_q t_k") 

    # Apply mask
    x = x.masked_fill(~mask, float('-inf'))

    # Softmax
    x = softmax(x, dim=-1)
    # print(x[0, 0, 0])
    # print(mask[0])
    
    x = einops.einsum(x, v, "b ... t_q t_k, b ... t_k d_v -> b ... t_q d_v")
    return x


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, use_rope=False, max_seq_len=None, theta=None):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads

        # Calculate d_k and d_v
        d_k = d_model // num_heads
        self.d_k = d_k
        d_v = d_model // num_heads
        self.d_v = d_v

        # Projections
        self.WQ = Linear(d_model, num_heads * d_k)
        self.WK = Linear(d_model, num_heads * d_k)
        self.WV = Linear(d_model, num_heads * d_v)
        self.WO = Linear(num_heads * d_v, d_model)

        # Define rope position embeddings
        self.use_rope = use_rope
        if use_rope:
            assert theta is not None and max_seq_len is not None
            self.rope = RotaryPositionalEmbedding(theta=theta, max_seq_len=max_seq_len, d_k=d_k)
        else:
            self.rope = None

    def forward(self, x, token_positions=None):
        # x: [B, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape

        # Project: [B, seq_len, d_model] -> [B, seq_len, num_heads x d_*]
        q, k, v = self.WQ(x), self.WK(x), self.WV(x)
        q = einops.rearrange(q, "b t (h d) -> b h t d", h=self.num_heads, d=self.d_k)
        k = einops.rearrange(k, "b t (h d) -> b h t d", h=self.num_heads, d=self.d_k)
        v = einops.rearrange(v, "b t (h d) -> b h t d", h=self.num_heads, d=self.d_v)

        # Apply RoPE
        if self.use_rope:
            # Repeat the token positions along heads dimension
            token_positions = einops.repeat(token_positions, "b t -> b h t", h=self.num_heads)

            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # Construct causal mask
        mask = torch.triu(torch.ones((seq_len, seq_len))).T.to(dtype=torch.bool, device=x.device)

        # Apply MHSA
        x = scaled_dot_product_attention(q, k, v, mask)

        # Output projection
        x = einops.rearrange(x, "b h t d_v -> b t (h d_v)")
        x = self.WO(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, theta, max_seq_len, layer_norm="pre", use_rope=True, ffn_act="swiglu"):
        super().__init__()

        self.cmhsa = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            use_rope=use_rope,
            theta=theta,
            max_seq_len=max_seq_len,
        )
        
        assert ffn_act in ["swiglu", "silu"]
        self.ffn_act = ffn_act
        if ffn_act == "swiglu":
            self.ffn = FFNSwiGLU(d_model, d_ff)
        elif ffn_act == "silu":
            # Hard code the feedforward dimension for SiLU
            d_ff = 4 * d_model
            self.ffn = FFNSiLU(d_model, d_ff)
        else:
            raise ValueError(f"Invalid FFN activation: {ffn_act}")
        
        assert layer_norm in ["pre", "post", "none"]
        self.layer_norm = layer_norm
        if layer_norm == "none":
            self.rmsnorm1 = torch.nn.Identity()
            self.rmsnorm2 = torch.nn.Identity()
        else:
            self.rmsnorm1 = RMSNorm(d_model)
            self.rmsnorm2 = RMSNorm(d_model)

    def forward(self, x, token_positions):
        
        if self.layer_norm == "pre":
            x = x + self.cmhsa(self.rmsnorm1(x), token_positions)
            x = x + self.ffn(self.rmsnorm2(x))
        elif self.layer_norm == "post":
            x = self.rmsnorm1(x + self.cmhsa(x, token_positions))
            x = self.rmsnorm2(x + self.ffn(x))
        elif self.layer_norm == "none":
            x = x + self.cmhsa(x, token_positions)
            x = x + self.ffn(x)
        else:
            raise ValueError(f"Invalid layer norm: {self.layer_norm}")
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, context_length, num_layers, d_model, d_ff, num_heads, theta, layer_norm="pre", ffn_act="swiglu", use_rope=True, tie_weights=False):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.layer_norm = layer_norm
        self.ffn_act = ffn_act
        self.use_rope = use_rope
        self.tie_weights = tie_weights
        
        # Embedding
        self.embedding = Embedding(vocab_size, d_model)

        # Layers
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    d_ff=d_ff,
                    num_heads=num_heads,
                    max_seq_len=context_length,
                    theta=theta,
                    layer_norm=layer_norm,
                    ffn_act=ffn_act,
                    use_rope=use_rope,
                ) for _ in range(num_layers)
            ]
        )

        # RMSNorm at the output
        self.rmsnorm_output = RMSNorm(d_model)
        
        # Linear output layer
        if self.tie_weights:
            # Share the weights of the embedding and the language model head
            self.lm_head = Linear(d_model, vocab_size, sigma_scale=0.5)
            self.embedding.E = self.lm_head.W
        else:
            self.lm_head = Linear(d_model, vocab_size, sigma_scale=1.0)
        
    def forward(self, token_ids, token_positions):
        x = self.embedding(token_ids)
        for b in self.blocks:
            x = b(x, token_positions)
        x = self.rmsnorm_output(x)
        x = self.lm_head(x)
        # x = softmax(x, dim=-1)
        return x