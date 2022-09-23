import torch.nn as nn

from attention import HydraAttention, SelfAttention


class Model(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_labels):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.attention = SelfAttention(hidden_size, hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dense = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        embedding = self.embedding(x)
        out = self.attention(embedding)
        out = self.layer_norm(out + embedding)
        out = self.dense(out)
        return out
