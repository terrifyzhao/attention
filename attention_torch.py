import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchtext.datasets import SogouNews


class SelfAttention(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 attention_probs_dropout_prob=0.1):
        super().__init__()
        assert hidden_size % num_attention_heads == 0
        self.d_k = hidden_size // num_attention_heads
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.drop_out = nn.Dropout(attention_probs_dropout_prob)
        self.q_matrix = nn.Linear(768, self.hidden_size)
        self.k_matrix = nn.Linear(768, self.hidden_size)
        self.v_matrix = nn.Linear(768, self.hidden_size)
        self.out_linear = nn.Linear(768, 2)

    def forward(self, x):
        query, key, value = x

        # mask = tf.cast(tf.expand_dims(mask, axis=1), dtype=tf.float32)
        # ones = tf.expand_dims(tf.ones(shape=tf.shape(query)[:2], dtype=tf.float32), axis=-1)
        # attention_mask = ones * mask

        query = self.q_matrix(query)
        key = self.k_matrix(key)
        value = self.v_matrix(value)

        # [batch_size, seq_len, n_heads, head_size]
        query = torch.reshape(query, [-1, query.shape[1], self.num_attention_heads, self.d_k])
        key = torch.reshape(key, [-1, key.shape[1], self.num_attention_heads, self.d_k])
        value = torch.reshape(value, [-1, value.shape[1], self.num_attention_heads, self.d_k])

        query = torch.transpose(query, 1, 2)
        key = torch.transpose(key, 1, 2)
        key = torch.transpose(key, 2, 3)
        value = torch.transpose(value, 1, 2)

        # [batch_size, n_heads, seq_len, seq_len]
        out = torch.matmul(query, key) / (self.d_k ** 0.5)

        # if attention_mask is not None:
        #     attention_mask = tf.expand_dims(attention_mask, axis=1)
        #     # {1: position, 0: mask} -> {0: position, -10000: mask}
        #     adder = (1.0 - tf.cast(attention_mask, dtype=tf.float32)) * -1e8
        #     out += adder

        out = F.softmax(out, dim=-1)
        out = self.drop_out(out)
        #  [batch_size, n_heads, seq_len, head_size]
        out = torch.matmul(out, value)
        out = torch.transpose(out, 1, 2)
        out = torch.reshape(out, [-1, out.shape[1], self.hidden_size])

        out = torch.mean(out, dim=1)
        # out = self.out_linear(out)
        out = self.out_linear(out)
        # out = F.sigmoid(out)

        return out


if __name__ == '__main__':

    train_dataset, test_dataset = SogouNews(root='sougounews', ngrams=3)

    net = SelfAttention(768, 12)

    x = np.random.random([1000, 50, 768])
    y = np.random.randint(0, 2, 1000)
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()

    dataset = TensorDataset(x, y)
    trainloader = DataLoader(dataset, batch_size=10)

    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net([inputs, inputs, inputs])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 50 == 0:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
