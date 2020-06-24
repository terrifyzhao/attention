import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model
from utils import create_initializer, pad_sequences
from tensorflow import keras

tf.config.experimental_run_functions_eagerly(False)


class SelfAttention(Layer):
    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 attention_probs_dropout_prob=0.1,
                 initializer_range=0.02,
                 **kwargs):
        super().__init__(**kwargs)
        assert hidden_size % num_attention_heads == 0
        self.d_k = hidden_size // num_attention_heads
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.drop_out = Dropout(attention_probs_dropout_prob)
        self.q_matrix = Dense(self.hidden_size,
                              kernel_initializer=create_initializer(initializer_range),
                              name='query')
        self.k_matrix = Dense(self.hidden_size,
                              kernel_initializer=create_initializer(initializer_range),
                              name='key')
        self.v_matrix = Dense(self.hidden_size,
                              kernel_initializer=create_initializer(initializer_range),
                              name='value')

    def call(self, inputs, training=False, mask=None):
        query, key, value = inputs

        mask = tf.cast(tf.expand_dims(mask, axis=1), dtype=tf.float32)
        ones = tf.expand_dims(tf.ones(shape=tf.shape(query)[:2], dtype=tf.float32), axis=-1)
        attention_mask = ones * mask

        query = self.q_matrix(query)
        key = self.k_matrix(key)
        value = self.v_matrix(value)

        # [batch_size, seq_len, n_heads, head_size]
        query = tf.reshape(query, [-1, tf.shape(query)[1], self.num_attention_heads, self.d_k])
        key = tf.reshape(key, [-1, tf.shape(key)[1], self.num_attention_heads, self.d_k])
        value = tf.reshape(value, [-1, tf.shape(value)[1], self.num_attention_heads, self.d_k])

        query = tf.transpose(query, [0, 2, 1, 3])
        key = tf.transpose(key, [0, 2, 1, 3])
        value = tf.transpose(value, [0, 2, 1, 3])

        # [batch_size, n_heads, seq_len, seq_len]
        out = tf.matmul(query, key, transpose_b=True) / (self.d_k ** 0.5)

        if attention_mask is not None:
            attention_mask = tf.expand_dims(attention_mask, axis=1)
            # {1: position, 0: mask} -> {0: position, -10000: mask}
            adder = (1.0 - tf.cast(attention_mask, dtype=tf.float32)) * -1e8
            out += adder

        out = tf.nn.softmax(out, axis=-1)
        out = self.drop_out(out, training=training)
        #  [batch_size, n_heads, seq_len, head_size]
        out = tf.matmul(out, value)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [-1, tf.shape(out)[1], self.hidden_size])

        return out


class DilationAttention(Layer):
    """
    思路是把间隔dilation_rate的元素组成一个张量，
    例如[1,2,3,4,5]->[1,2,3,4,5,0]->[[1,2],[3,4],[5,0]]->[[1,3,5],[2,4,0]]
    此时再做attention就是dilation attention
    """

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 dilation_rate=1,
                 attention_probs_dropout_prob=0.1,
                 initializer_range=0.02,
                 **kwargs):
        super().__init__(**kwargs)
        self.dilation_rate = dilation_rate
        self.attention = SelfAttention(hidden_size,
                                       num_attention_heads,
                                       attention_probs_dropout_prob,
                                       initializer_range)

    def call(self, inputs, **kwargs):
        if isinstance(inputs, list):
            x, mask = inputs
        else:
            x, mask = inputs, None

        # 序列长度、编码维度
        seq_len, seq_dim = x.get_shape()[1:]
        # 计算需要补长多少，余数是指有几个数没办法组成一组，和dilation_rate相减就是补0的个数
        pad_len = self.dilation_rate - seq_len % self.dilation_rate
        # 在序列长度的维度补0
        x = tf.pad(x, paddings=[[0, 0], [0, pad_len], [0, 0]])
        new_seq_len = tf.shape(x)[1]
        # 把间隔dilation_rate的元素放在一组
        x = tf.reshape(x, [-1, new_seq_len // self.dilation_rate, self.dilation_rate, seq_dim])
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [-1, new_seq_len, seq_dim])
        # attention
        x = self.attention([x, x, x], mask=mask)
        #  转变回原来的维度
        x = tf.reshape(x, [-1, self.dilation_rate, new_seq_len // self.dilation_rate, seq_dim])
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [-1, new_seq_len, seq_dim])
        # 把补0的元素去除
        x = x[:, : -pad_len]

        return x


if __name__ == '__main__':
    imdb = tf.keras.datasets.imdb
    (x_train, y_train), (x_test, y_test) = imdb.load_data()
    x_train = pad_sequences(x_train, 200)

    x = Input([200, ])
    mask = Input([7, ])

    tf.keras.models.Sequential([
        Embedding(88586, 256),
        DilationAttention(64, 4)
    ])

    embedding = Embedding(88586, 256)
    attention = DilationAttention(64, 4)
    x = embedding(x)
    y = attention([x, mask])
    model = Model([x, mask], y)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(1e-5),
        metrics=['accuracy']
    )
    model.fit(x_train, y_train, batch_size=32, epochs=2)
