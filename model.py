import torch.nn as nn

from attention import HydraAttention, SelfAttention


class Model(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_labels, layer_num):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        self.model_list = nn.ModuleList(
            [SelfAttention(hidden_size, hidden_size, hidden_size, relax=True) for _ in range(layer_num)])

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dense = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        out = self.embedding(x)
        for layer in self.model_list:
            inputs = out
            out = layer(out)
            out = self.layer_norm(out + inputs)
        out = self.dense(out)
        return out
