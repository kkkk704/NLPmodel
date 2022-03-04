import re
import math
import torch
import numpy as np
from random import *
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

text = (
    'Hello, how are you? I am Romeo.\n'  # R
    'Hello, Romeo My name is Juliet. Nice to meet you.\n'  # J
    'Nice meet you too. How are you today?\n'  # R
    'Great. My baseball team won the competition.\n'  # J
    'Oh Congratulations, Juliet\n'  # R
    'Thank you Romeo\n'  # J
    'Where are you going today?\n'  # R
    'I am going shopping. What about you?\n'  # J
    'I am going to visit my grandmother. she is not very well'  # R
)

# 正则匹配，空格替换文本中特殊符号 text.lower() 大写转换为小写
sentences = re.sub('[.,!?\\-]', '', text.lower()).split('\n')
# word_list:句子中所以的token组成的不重复列表
word_list = list(set(' '.join(sentences).split()))
# word_list：token:id 为所有的token添加一个id
word2idx = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
for i, w in enumerate(word_list):
    word2idx[w] = i + 4
# idx2word:转置word2idx的key和value id:token
idx2word = {i: w for i, w in enumerate(word2idx)}
# 词典大小
vocab_size = len(word2idx)

# token_list:以句子为单位存储token_id [[idx1,idx2,idx3,...],[...],[...],...]
token_list = list()
for sentence in sentences:
    arr = [word2idx[s] for s in sentence.split()]
    token_list.append(arr)

# 模型参数
maxlen = 30  # 句子最大长度，不够补PAD
batch_size = 6
max_pred = 5  # 句子中最大mask数
n_layers = 6  # encoder个数
n_heads = 12  # attention 头数
d_model = 768  # embedding维度
d_ff = d_model * 4  # linear维度
d_k = d_v = 64
n_segments = 2  # input中最大合并句子数
epochs = 180
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# 数据预处理
# input的MASK操作和上下句子合并

def mask_data():
    batch = []
    positive = negative = 0
    while positive != batch_size / 2 or negative != batch_size / 2:
        # 随机选择上下句索引并获取其下标 tokens_a/b:[idx1_1,idx1_2,...]/[idx2_1,idx2_2,...]
        tokens_a_index, tokens_b_index = randrange(len(sentences)), randrange(len(sentences))
        tokens_a, tokens_b = token_list[tokens_a_index], token_list[tokens_b_index]
        # 构造input:[CLS]+sentence_a+[SEP]+sentence_b+[SEP] [0,idx1_1,idx1_2,...,1,idx2_1,idx2_2,...,1]
        input_ids = [word2idx['[CLS]']] + tokens_a + [word2idx['[SEP]']] + tokens_b + [word2idx['[SEP]']]
        # 构造input段信息：[0 0 0 ... 0 1 1 1 ... 1]
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

        # MASK LM
        # mask token的个数
        n_pred = min(max_pred, max(1, int(len(input_ids) * 0.15)))
        # 候选mask的token_position mask的token不能是CLS和SEP  [idx1_1,idx1_2,....,idx2_1,idx2_2,...]
        cand_masked_pos = [i for i, token in enumerate(input_ids)
                           if token != word2idx['[CLS]'] and token != word2idx['[SEP]']]
        # 随机取前n_pred个token进行mask 打乱候选列表顺序
        shuffle(cand_masked_pos)
        # 存mask的
        masked_tokens, masked_pos = [], []
        for pos in cand_masked_pos[:n_pred]:
            # pos: rand_idx
            masked_tokens.append(input_ids[pos])
            masked_pos.append(pos)

            if random() < 0.8:
                input_ids[pos] = word2idx['[MASK]']
            elif random() > 0.5:
                index = randint(4, vocab_size - 1)  # randint双闭区间
                input_ids[pos] = index

        # PAD
        n_pad = maxlen - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        # 覆盖mask不足的部分，同一个batch里mask的个数必须一致
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)

        if tokens_a_index + 1 == tokens_b_index and positive < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True])
            positive += 1
        elif tokens_a_index + 1 != tokens_b_index and negative < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False])
            negative += 1

    return batch


batch = mask_data()
input_ids, segment_ids, masked_tokens, masked_pos, isNext = zip(*batch)
input_ids, segment_ids, masked_tokens, masked_pos, isNext = \
    torch.LongTensor(input_ids), torch.LongTensor(segment_ids), \
    torch.LongTensor(masked_tokens), torch.LongTensor(masked_pos), torch.LongTensor(isNext)


class MyDataSet(Data.Dataset):
    def __init__(self, input_ids, segment_ids, masked_tokens, masked_pos, isNext):
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.masked_tokens = masked_tokens
        self.mask_pos = masked_pos
        self.isNext = isNext

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.segment_ids[idx], self.masked_tokens[idx], \
               self.mask_pos[idx], self.isNext[idx]


loader = Data.DataLoader(MyDataSet(input_ids, segment_ids, masked_tokens, masked_pos, isNext), batch_size, True)


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, seq_len = seq_q.size()  # [batch_size,seq_len]
    # PAD=0
    pad_attn_mask = seq_q.data.eq(0).unsqueeze(1)  # [batch_size,1,seq_len]的True or Fasle矩阵
    return pad_attn_mask.expand(batch_size, seq_len, seq_len)


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.segment_embed = nn.Embedding(n_segments, d_model)
        self.position_embed = nn.Embedding(maxlen, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        '''
        :param x: [batch_size,seq_len]
        :param seg: [batch_size,seq_len]
        :return: [batch_size,seq_len,d_model]
        '''
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long).to(device)  # 创建长度为seq_len的顺序序列
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len]->[batch_size,seq_len]
        embedding = self.token_embed(x) + self.segment_embed(seg) + self.position_embed(pos)

        return self.norm(embedding)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)

    def forward(self, Q, K, V, attn_mask):
        '''
        :param Q: [batch_size,seq_len,d_model]
        :param attn_mask: [batch_size,max_len,max_len]
        :return:
        '''
        residual, batch_size = Q, Q.size(0)

        # [batch_size,seq_len,d_k*n_head]->[batch_size,seq_len,n_head,d_k]->[batch_size,n_head_seq_len,d_k]
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
        # [batch_size,n_heads,max_len,max_len]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        context = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = nn.Linear(n_heads * d_v, d_model).to(device)(context)
        return nn.LayerNorm(d_model).to(device)(output + residual)


class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        residual = x
        x = self.fc2(gelu(self.fc1(x)))
        return nn.LayerNorm(d_model).to(device)(x + residual)


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention()
        self.feedward = FeedForward()

    def forward(self, input, self_attn_mask):
        outputs = self.self_attn(input, input, input, self_attn_mask)
        outputs = self.feedward(outputs)

        return outputs


# bert：两个步骤：预测mask的值；预测句子是否连续
'''
Input:
[CLS] calculus is a branch of math [SEP] panda is native to [MASK] central china [SEP]

Targets: false, south
----------------------------------
Input:
[CLS] calculus is a [MASK] of math [SEP] it [MASK] developed by newton and leibniz [SEP]

Targets: true, branch, was
'''


class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(),
            nn.Tanh(),
        )
        # 根据CLS对应的输出给出两个句子之间的关系
        self.classifier = nn.Linear(d_model, 2)
        self.linear = nn.Linear(d_model, d_model)
        self.active = gelu

        embed_weight = self.embedding.token_embed.weight
        self.fc2 = nn.Linear(d_model, vocab_size, bias=False)
        self.fc2.weight = embed_weight

    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.embedding(input_ids, segment_ids)  # [batch_size,seq_len,d_model]
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
        for layer in self.layers:
            output = layer(output, enc_self_attn_mask)  # [batch_size,seq_len,d_model]

        h_pooled = self.fc(output[:, 0])  # 句子是否连续的预测 CLS位置的输出
        logits_clsf = self.classifier(h_pooled)

        # output.shape:batch_size,seq_len,d_model 其masked_pos.shape:batch_size,max_pred
        # 扩展masked_pos的原因：gather函数需要为矩阵中每一个元素指定位置，所以embedding中每一个元素都要给位置
        masked_pos = masked_pos[:, :, None].expand(-1, -1, d_model)
        # h_masked.shape as masked_pos:即 [batch_size,max_pred,d_model]
        h_masked = torch.gather(output, 1, masked_pos)
        # batch_size,max_pred,d_model->d_model->vocab_size
        h_masked = self.active(self.linear(h_masked))
        logits_lm = self.fc2(h_masked)

        return logits_lm, logits_clsf


model = BERT().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adadelta(model.parameters(), lr=0.001)

for epoch in range(epochs):
    for input_ids, segment_ids, masked_tokens, masked_pos, isNext in loader:
        input_ids, segment_ids, masked_tokens, masked_pos, isNext = \
            input_ids.to(device), segment_ids.to(device), masked_tokens.to(device), masked_pos.to(device), isNext.to(
                device)
        logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
        loss_lm = criterion(logits_lm.view(-1, vocab_size), masked_tokens.view(-1))
        loss_lm = (loss_lm.float()).mean()

        loss_clsf = criterion(logits_clsf, isNext)
        loss = loss_lm + loss_clsf

        if (epoch + 1) % 10 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

input_ids, segment_ids, masked_tokens, masked_pos, isNext = batch[0]
print(text)
print([idx2word[w] for w in input_ids if idx2word[w] != '[PAD]'])

input_ids, segment_ids, masked_pos = \
    torch.LongTensor([input_ids]).to(device),torch.LongTensor([segment_ids]).to(device),torch.LongTensor([masked_pos]).to(device)

logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
logits_lm = logits_lm.data.max(2)[1][0].data.numpy()
print('mask tokens lost:', [pos for pos in masked_tokens if pos != 0])
print('predict mask tokens list:', [pos for pos in logits_lm if pos != 0])

# output.data(数据).max(1)(dim=1的最大值:values,indices)[1](取最大值的下标)
logits_clsf = logits_clsf.data.max(1)[1].data.numpy()[0]
print('isNext:', True if isNext else False)
print('predict isNext:', True if logits_clsf else False)
