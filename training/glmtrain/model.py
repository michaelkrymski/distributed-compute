"""Bias-free decoder: dense attention, RoPE, RMSNorm, GELU FFN."""

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 8192
    width: int = 384
    layers: int = 12
    heads: int = 6
    expansion: int = 4
    context: int = 512
    tied: bool = True

    def __post_init__(self):
        for key in ("vocab_size", "width", "layers", "heads", "expansion", "context"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if not 257 <= self.vocab_size <= 65536:
            raise ValueError("Byte BPE requires vocabulary between 257 and 65536")
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("width / heads must be an even integer for RoPE")
        if type(self.tied) is not bool:
            raise ValueError("tied must be boolean")

    def parameter_count(self):
        d, f = self.width, self.width * self.expansion
        return (
            self.vocab_size * d * (1 if self.tied else 2)
            + self.layers * (4 * d * d + 2 * d * f + 2 * d)
            + d
        )


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        scaled = x.float() * torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + 1e-6
        )
        return scaled.to(x.dtype) * self.weight


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads, self.head_dim = c.heads, c.width // c.heads
        self.qkv = nn.Linear(c.width, 3 * c.width, bias=False)
        self.out = nn.Linear(c.width, c.width, bias=False)
        positions = torch.arange(c.context).float()
        frequency = 10000 ** (
            -torch.arange(0, self.head_dim, 2).float() / self.head_dim
        )
        angles = positions[:, None] * frequency[None, :]
        self.register_buffer("cos", angles.cos()[None, None], persistent=False)
        self.register_buffer("sin", angles.sin()[None, None], persistent=False)

    def rotate(self, x):
        cos, sin = (
            self.cos[:, :, : x.size(2)].to(x.dtype),
            self.sin[:, :, : x.size(2)].to(x.dtype),
        )
        a, b = x[..., ::2], x[..., 1::2]
        return torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (
            a.view(b, t, self.heads, self.head_dim).transpose(1, 2) for a in (q, k, v)
        )
        y = F.scaled_dot_product_attention(
            self.rotate(q), self.rotate(k), v, is_causal=True
        )
        return self.out(y.transpose(1, 2).contiguous().view(b, t, d))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(c.width), RMSNorm(c.width)
        self.attention = Attention(c)
        self.ffn = nn.Sequential(
            nn.Linear(c.width, c.width * c.expansion, bias=False),
            nn.GELU(),
            nn.Linear(c.width * c.expansion, c.width, bias=False),
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm = RMSNorm(config.width)
        self.output = (
            None
            if config.tied
            else nn.Linear(config.width, config.vocab_size, bias=False)
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, tokens, targets=None):
        if tokens.ndim != 2 or tokens.size(1) > self.config.context:
            raise ValueError("Expected [batch, sequence] within configured context")
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = (
            F.linear(x, self.embedding.weight)
            if self.output is None
            else self.output(x)
        )
        loss = (
            None
            if targets is None
            else F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
        )
        return logits, loss
