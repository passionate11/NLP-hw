# 导入包 
import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import spacy
import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


# Set to False to skip notebook execution (e.g. for debugging)
warnings.filterwarnings("ignore")
RUN_EXAMPLES = True

# Some convenience helper functions used throughout the notebook


def is_interactive_notebook():
    return __name__ == "__main__"


def show_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        return fn(*args)


def execute_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        fn(*args)


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    # 参数 : encoder decoder -> 模型结构
    # src_embed, tgt_embed -> int 源语以及目标语embedding维度，一般为512 
    # generator -> 生成器 将当前向量转为词表大小的概率数据
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

# 语言模型生成时将输入向量转化为词表大小的向量，再计算每个单词的概率
class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        # 线性层，将d_model维度转化为vocab（词表大小）维度
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # log softmax操作
        # 举例
        # input logit = t.tensor([0.1,0.1,0.1,0.7])
        # print(torch.nn.functional.log_softmax(logit)) 
        # output tensor([-1.5732, -1.5732, -1.5732, -0.9732])
        # 即 将词表维度大小的向量调整为和为1的向量
        return log_softmax(self.proj(x), dim=-1)

# 复制多层
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    # 平滑
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    # 每个transformer的encoder有两个sublayer(多头注意力和FFN)，该部分做的子层连接，
    # 包括3步 layernorm dropout 以及 residual连接
    # pre norm 和 last norm应该是在这里设置，之后实验的时候可以试试
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
      # size:512 self_attn、feed_foward:自注意力层和前馈层
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # 两个连接子层
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        # 两个子层 multi_head-attention 以及 ffn
        # self.self_attn(x, x, x, mask)) 三个x是开始的kqv都是原输入
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
        
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    # decoder包含3个子层 masked-multi_head-attention cross-multi_head-attention ffn
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        # 来自源语言序列的Encoder之后的输出，作为memory
        # 供目标语言的序列检索匹配：（类似于alignment in SMT)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    # torch.triu -> 返回一个张量的上三角矩阵 diagonal=1表示不包含对角线,0代表包含对角线
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    # 返回上三角（不包括对角线）为false 其他为true的一个矩阵
    return subsequent_mask == 0

# attention计算,这里的mask就是前面的subsequent_mask的结果作为输入（上三角不包括对角线为true其余为false）
def attention(query, key, value, mask=None, dropout=None):
# query, key, value的形状类似于(30, 8, 10, 64)-q  kv形状相同(30, 8, 11, 64), 
# (30, 8, 11, 64)
# 其中30是batchsize 8是head.num 10是目标语中词的个数 11是源语言传过来的memory中当前序列词的个数 64是每个词的向量维度
# 类似于，这里假定query来自target language sequence；
# key和value都来自source language sequence.
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1) # d_k=64
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 先是(30,8,10,64)和(30, 8, 64, 11)在最后两个维度相乘，得到(30,8,10,11)，
    # 代表10个目标语言序列中每个词和11个源语言序列的分别的“亲密度”，除以sqrt(d_k)=8，防止过大的亲密度。
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        # 使用mask，对已经计算好的scores，按照mask矩阵，填-1e9，
        # 然后在下一步计算softmax的时候，被设置成-1e9的数对应的值~0,被忽视
    p_attn = scores.softmax(dim=-1)
    # 对scores的最后一个维度执行softmax，得到的还是一个tensor, (30, 8, 10, 11)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
#返回的第一项，是(30,8,10, 11)乘以（最后两个维度相乘）value=(30,8,11,64)，得到的tensor是(30,8,10,64)，这样最后得到的(10,64)就是10个混合着11个不同权重的64维向量
#和query的最初的形状一样。另外，返回p_attn，形状为(30,8,10,11). 注意，这里返回p_attn主要是用来可视化显示多头注意力机制。

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
      # h=8头数 d_model=512
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0 # 设定d_v等于d_k，且均为 512 / 8 =64
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        # 4个线性层
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
      # 这里的query形状为(30,10,512) 30:batchsize 10:目标语长度 512:每个token维度
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]
        # 过3个线性层
        # 对query (30,10,512) -> Linear network -> (30,10,512) -> view -> (30,10,8,64) -> transpose(1,2) -> (30,8,10,64)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )
        # 用刚刚定义好的attention对每个投影的x做注意力计算返回(30,8,10,64)
        # attn为（30,8,10,11）

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        # x(30,8,10,64) -> transpose(1,2) -> (30,10,8,64) -> contiguous and view -> (30,10,512)
        del query
        del key
        del value
        return self.linears[-1](x)
        # 第四个线性层，将(30,10,512)的x再过一次线性层

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    # FNN-两个线性层 512->中间维度->512
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
      # d_model:512 vocab:当前词表大小
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        # one-hot转为词嵌入 一个可以训练的embedding矩阵 大小为vocab*d_model
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

# 一种不用学习参数的位置编码
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


def make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model