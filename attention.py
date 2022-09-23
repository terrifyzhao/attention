import torch
import torch.nn as nn
import math


class BaseAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, all_head_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.all_head_size = all_head_size

        self.head_size = self.all_head_size // num_attention_heads

        self.query = nn.Linear(hidden_size, all_head_size)
        self.key = nn.Linear(hidden_size, all_head_size)
        self.value = nn.Linear(hidden_size, all_head_size)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        return x, q, k, v


class SelfAttention(BaseAttention):
    def __init__(self, hidden_size, num_attention_heads, all_head_size):
        super().__init__(hidden_size, num_attention_heads, all_head_size)

    def forward(self, x):
        x, q, k, v = super().forward(x)
        out_shape = x.size()
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        out = torch.matmul(attention_probs, v)
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(out_shape)
        return out


class HydraAttention(BaseAttention):

    def __init__(self, hidden_size, num_attention_heads, all_head_size):
        super().__init__(hidden_size, num_attention_heads, all_head_size)

    def forward(self, x):
        x, q, k, v = super().forward(x)
        out_shape = x.size()
        # [batch, hidden, len, 1]
        q = q / q.norm(dim=-1, keepdim=True)
        # [batch, hidden, len, 1]
        k = k / k.norm(dim=-1, keepdim=True)
        # [batch, hidden, len, 1]
        kv = (k * v).sum(dim=-2, keepdim=True)
        out = q * kv
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(out_shape)
        return out


if __name__ == '__main__':
    import time

    s = time.time()
    attention = HydraAttention(768, 768, 768)
    # attention = SelfAttention(768, 12, 768)
    for _ in range(10):
        r = attention(torch.rand((2, 500, 768)))
    print(time.time() - s)
