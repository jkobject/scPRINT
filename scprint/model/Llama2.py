from torch import nn
import torch
import torch.nn.functional as F


def precompute_theta_pos_frequencies(head_dim, seq_len, device, theta=10000.0):
    # theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
    # (head_dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)

    # (seq_len)
    m = torch.arange(seq_len, device=device)

    # (seq_len, head_dim / 2)
    freqs = torch.outer(m, theta).float()

    # complex numbers in polar, c = R * exp(m * theta), where R = 1:
    # (seq_len, head_dim/2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def apply_rotary_embeddings(x, freqs_complex, device):
    # last dimension pairs of two values represent real and imaginary
    # two consecutive values will become a single complex number

    # (m, seq_len, num_heads, head_dim/2, 2)
    x = x.float().reshape(*x.shape[:-1], -1, 2)

    # (m, seq_len, num_heads, head_dim/2)
    x_complex = torch.view_as_complex(x)

    # (seq_len, head_dim/2) --> (1, seq_len, 1, head_dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)

    # multiply each complex number
    # (m, seq_len, n_heads, head_dim/2)
    x_rotated = x_complex * freqs_complex

    # convert back to the real number
    # (m, seq_len, n_heads, head_dim/2, 2)
    x_out = torch.view_as_real(x_rotated)

    # (m, seq_len, n_heads, head_dim)
    x_out = x_out.reshape(*x.shape)

    return x_out.type_as(x).to(device)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (m, seq_len, dim) * (m, seq_len, 1) = (m, seq_len, dim)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # weight is a gain parameter used to re-scale the standardized summed inputs
        # (dim) * (m, seq_len, dim) = (m, seq_Len, dim)
        return self.weight * self._norm(x.float()).type_as(x)


class KVCache:
    def __init__(self, max_batch_size, max_seq_len, n_kv_heads, head_dim, device):
        self.cache_k = torch.zeros(
            (max_batch_size, max_seq_len, n_kv_heads, head_dim)
        ).to(device)
        self.cache_v = torch.zeros(
            (max_batch_size, max_seq_len, n_kv_heads, head_dim)
        ).to(device)

    def update(self, batch_size, start_pos, xk, xv):
        self.cache_k[:batch_size, start_pos : start_pos + xk.size(1)] = xk
        self.cache_v[:batch_size, start_pos : start_pos + xv.size(1)] = xv

    def get(self, batch_size, start_pos, seq_len):
        keys = self.cache_k[:batch_size, : start_pos + seq_len]
        values = self.cache_v[:batch_size, : start_pos + seq_len]
        return keys, values


def repeat_kv(x, n_rep):
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        # (m, seq_len, n_kv_heads, 1, head_dim)
        # --> (m, seq_len, n_kv_heads, n_rep, head_dim)
        # --> (m, seq_len, n_kv_heads * n_rep, head_dim)
        return (
            x[:, :, :, None, :]
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        )


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config["n_heads"]
        self.n_kv_heads = config["n_kv_heads"]
        self.dim = config["embed_dim"]
        self.n_kv_heads = self.n_heads if self.n_kv_heads is None else self.n_kv_heads
        self.n_heads_q = self.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = self.dim // self.n_heads

        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

        self.cache = KVCache(
            max_batch_size=config["max_batch_size"],
            max_seq_len=config["max_seq_len"],
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            device=config["device"],
        )

    def forward(self, x, start_pos, freqs_complex):
        # seq_len is always 1 during inference
        batch_size, seq_len, _ = x.shape

        # (m, seq_len, dim)
        xq = self.wq(x)

        # (m, seq_len, h_kv * head_dim)
        xk = self.wk(x)
        xv = self.wv(x)

        # (m, seq_len, n_heads, head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)

        # (m, seq_len, h_kv, head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # (m, seq_len, num_head, head_dim)
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)

        # (m, seq_len, h_kv, head_dim)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        # replace the entry in the cache
        self.cache.update(batch_size, start_pos, xk, xv)

        # (m, seq_len, h_kv, head_dim)
        keys, values = self.cache.get(batch_size, start_pos, seq_len)

        # (m, seq_len, h_kv, head_dim) --> (m, seq_len, n_heads, head_dim)
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # (m, n_heads, seq_len, head_dim)
        # seq_len is 1 for xq during inference
        xq = xq.transpose(1, 2)

        # (m, n_heads, seq_len, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # (m, n_heads, seq_len_q, head_dim) @ (m, n_heads, head_dim, seq_len) -> (m, n_heads, seq_len_q, seq_len)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        # (m, n_heads, seq_len_q, seq_len)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (m, n_heads, seq_len_q, seq_len) @ (m, n_heads, seq_len, head_dim) -> (m, n_heads, seq_len_q, head_dim)
        output = torch.matmul(scores, values)

        # ((m, n_heads, seq_len_q, head_dim) -> (m, seq_len_q, dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # (m, seq_len_q, dim)
        return self.wo(output)


def sigmoid(x, beta=1):
    return 1 / (1 + torch.exp(-x * beta))


def swiglu(x, beta=1):
    return x * sigmoid(x, beta)


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        hidden_dim = 4 * config["embed_dim"]
        hidden_dim = int(2 * hidden_dim / 3)

        if config["ffn_dim_multiplier"] is not None:
            hidden_dim = int(config["ffn_dim_multiplier"] * hidden_dim)

        # Round the hidden_dim to the nearest multiple of the multiple_of parameter
        hidden_dim = config["multiple_of"] * (
            (hidden_dim + config["multiple_of"] - 1) // config["multiple_of"]
        )

        self.w1 = nn.Linear(config["embed_dim"], hidden_dim, bias=False)
        self.w2 = nn.Linear(config["embed_dim"], hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, config["embed_dim"], bias=False)

    def forward(self, x: torch.Tensor):
        # (m, seq_len, dim) --> (m, seq_len, hidden_dim)
        swish = swiglu(self.w1(x))
        # (m, seq_len, dim) --> (m, seq_len, hidden_dim)
        x_V = self.w2(x)

        # (m, seq_len, hidden_dim)
        x = swish * x_V

        # (m, seq_len, hidden_dim) --> (m, seq_len, dim)
        return self.w3(x)


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config["n_heads"]
        self.dim = config["embed_dim"]
        self.head_dim = self.dim // self.n_heads

        self.attention = SelfAttention(config)
        self.feed_forward = FeedForward(config)

        # rms before attention block
        self.attention_norm = RMSNorm(self.dim, eps=config["norm_eps"])

        # rms before  feed forward block
        self.ffn_norm = RMSNorm(self.dim, eps=config["norm_eps"])

    def forward(self, x, start_pos, freqs_complex):
        # (m, seq_len, dim)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        # (m, seq_len, dim)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config["vocab_size"]
        self.n_layers = config["n_layers"]
        self.tok_embeddings = nn.Embedding(self.vocab_size, config["embed_dim"])
        self.head_dim = config["embed_dim"] // config["n_heads"]

        self.layers = nn.ModuleList()
        for layer_id in range(config["n_layers"]):
            self.layers.append(DecoderBlock(config))

        self.norm = RMSNorm(config["embed_dim"], eps=config["norm_eps"])
        self.output = nn.Linear(config["embed_dim"], self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(
            self.head_dim, config["max_seq_len"] * 2, device=(config["device"])
        )

    def forward(self, tokens, start_pos):
        # (m, seq_len)
        batch_size, seq_len = tokens.shape

        # (m, seq_len) -> (m, seq_len, embed_dim)
        h = self.tok_embeddings(tokens)

        # (seq_len, (embed_dim/n_heads)/2]
        freqs_complex = self.freqs_complex[start_pos : start_pos + seq_len]

        # Consecutively apply all the encoder layers
        # (m, seq_len, dim)
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)

        # (m, seq_len, vocab_size)
        output = self.output(h).float()
        return output


model = Transformer(config).to(config["device"])
res = model.forward(test_set["input_ids"].to(config["device"]), 0)
print(res.size())
