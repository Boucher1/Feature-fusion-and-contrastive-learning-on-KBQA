import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, BertConfig, AutoModel, AutoConfig, T5EncoderModel
from transformers import BertTokenizer
from torchsummary import summary

class SimcseModel(nn.Module):
    """Simcse有监督模型定义"""

    def __init__(self, pretrained_model, pooling, dropout=0):
        super(SimcseModel, self).__init__()
        # config = SimBertConfig.from_pretrained(pretrained_model)
        # 有监督也可以加入dropout
        config = AutoConfig.from_pretrained(pretrained_model)
        config.attention_probs_dropout_prob = dropout  # 修改config的dropout系数
        config.hidden_dropout_prob = dropout
        self.bert = AutoModel.from_pretrained(pretrained_model, config=config)
        self.pooling = pooling


        # self.bert = SimBertModel.from_pretrained(pretrained_model, config=config)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True, return_dict=True)
        # return out[1]
        if self.pooling == 'cls':
            return out.last_hidden_state[:, 0]  # [batch, 768]
        if self.pooling == 'pooler':
            return out.pooler_output  # [batch, 768]
        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen] 
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]



class SimcseModel_T5(nn.Module):
    """Simcse有监督模型定义"""

    def __init__(self, pretrained_model, pooling, dropout=0):
        super(SimcseModel_T5, self).__init__()
        # config = SimBertConfig.from_pretrained(pretrained_model)
        # 有监督也可以加入dropout
        config = AutoConfig.from_pretrained(pretrained_model)
        config.attention_probs_dropout_prob = dropout  # 修改config的dropout系数
        config.hidden_dropout_prob = dropout
        # tokenizer对句子进行编码[batch, max_len]，Encoder再对词进行编码[batch, max_len, embedding_size]
        self.T5 = T5EncoderModel.from_pretrained(pretrained_model, config=config)
        self.pooling = pooling


        # self.bert = SimBertModel.from_pretrained(pretrained_model, config=config)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 这里输入进来的input_ids, attention_mask, token_type_ids的维度经过转换为了[192,64]，原来[64,3,64]     [batch, sent_per_batch, max_len]
        out = self.T5(input_ids, attention_mask, token_type_ids, output_hidden_states=True, return_dict=True)
        print(out.last_hidden_state.size())   # [192,64,768]
        # return out.last_hidden_state[:, 0]
        if self.pooling == 'cls':
            print(out.last_hidden_state[:, 0])
            print(out.last_hidden_state[:, 0].size())
            return out.last_hidden_state[:, 0]  # [192, 768]
        if self.pooling == 'pooler':
            return out.pooler_output  # [192, 768]
        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]



class SimcseModel_T5_CNN(nn.Module):
    """Simcse有监督模型定义"""

    def __init__(self, pretrained_model, pooling, dropout=0):
        super(SimcseModel_T5_CNN, self).__init__()
        # config = SimBertConfig.from_pretrained(pretrained_model)
        # 有监督也可以加入dropout
        config = AutoConfig.from_pretrained(pretrained_model)
        config.attention_probs_dropout_prob = dropout  # 修改config的dropout系数
        config.hidden_dropout_prob = dropout
        # tokenizer对句子进行编码[batch, max_len]，Encoder再对词进行编码[batch, max_len, embedding_size]
        self.T5 = T5EncoderModel.from_pretrained(pretrained_model, config=config)
        # self.T5 = AutoModel.from_pretrained(pretrained_model, config=config)
        self.pooling = pooling

        # TextCNN参数
        # 卷积核个数
        self.num_filters = 100
        # 每个卷积核的长宽深
        self.filter_sizes = [3, 4, 5]
        # 构建了三个卷积层
        self.conv_layers = nn.ModuleList([
            # in_channels：代表输入特征矩阵的深度即channel，比如输入一张RGB彩色图像，那in_channels=3；
            # out_channels：代表卷积核的个数，使用C个卷积核输出的特征矩阵深度即channel就是C；
            # kernel_size：代表卷积核的尺寸，输入可以是int类型，例如：3 ，代表卷积核的height = width = 3，也可以是tuple类型，例如(3, 5)，代表卷积核的height = 3，width = 5；
            # stride：代表卷积核的步距，默认为1，和kernel_size一样，输入可以是int类型，也可以是tuple类型。注意：若为tuple类型第一个代表高度，第二个代表宽度；
            # padding：代表在输入特征矩阵四周补零的情况默认为0，同样输入可以为int型，例如：1 ，代表补一圈0，如果输入为tuple型如(2, 1) 代表在上下补两行，左右补一列。
            # dilation：定义了卷积核处理数据时各值的间距。换句话说，相比原来的标准卷积，扩张卷积多了一个超参数称之为dilation rate（扩张率），指的是kernel各点之间的间隔数量，正常的卷积核的dilation为1。
            nn.Conv2d(1, self.num_filters, (k, config.hidden_size)) for k in self.filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        # 全连接层（线性层）
        self.fc = nn.Linear(len(self.filter_sizes) * self.num_filters, config.hidden_size)


        # self.bert = SimBertModel.from_pretrained(pretrained_model, config=config)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 这里输入进来的input_ids, attention_mask, token_type_ids的维度经过转换为了[192,128]，原来[64,3,128]
        out = self.T5(input_ids, attention_mask, token_type_ids, output_hidden_states=True, return_dict=True)
        # print(out.last_hidden_state.size())   # [192,128,768]

        # TextCNN操作
        x = out.last_hidden_state.unsqueeze(1)  # [batch, 1, 128, 768]
        x = [F.relu(conv(x)).squeeze(3) for conv in
             self.conv_layers]  # [batch, num_filters, seqlen - filter_size + 1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [batch, num_filters]
        x = torch.cat(x, 1)  # [batch, num_filters * len(filter_sizes)] 在第二个维度上进行拼接
        x = self.dropout(x)
        # print("线性层之前的维度：", x.size())
        textcnn_output = self.fc(x)  # [batch, hidden_size]

        if self.pooling == 'cls':
            # print(textcnn_output.size())
            return textcnn_output    # [192, 768]
        if self.pooling == 'pooler':
            return textcnn_output.pooler_output  # [192, 768]
        if self.pooling == 'last-avg':
            last = textcnn_output.transpose(1, 2)  # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        if self.pooling == 'first-last-avg':
            first = textcnn_output[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = textcnn_output[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]
        # # T5池化操作
        # pooled_output = ''
        # if self.pooling == 'cls':
        #     pooled_output = out.last_hidden_state[:, 0]  # [batch, 768]
        # elif self.pooling == 'pooler':
        #     pooled_output = out.pooler_output  # [batch, 768]
        # elif self.pooling == 'last-avg':
        #     last_hidden_state = out.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
        #     pooled_output = torch.avg_pool1d(last_hidden_state, kernel_size=last_hidden_state.shape[-1]).squeeze(
        #         -1)  # [batch, 768]
        # elif self.pooling == 'first-last-avg':
        #     first_hidden_state = out.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
        #     last_hidden_state = out.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
        #     first_avg = torch.avg_pool1d(first_hidden_state, kernel_size=last_hidden_state.shape[-1]).squeeze(
        #         -1)  # [batch, 768]
        #     last_avg = torch.avg_pool1d(last_hidden_state, kernel_size=last_hidden_state.shape[-1]).squeeze(
        #         -1)  # [batch, 768]
        #     avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
        #     pooled_output = torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]




class SimcseModel_T5_GRU_CNN(nn.Module):
    """Simcse有监督模型定义"""

    def __init__(self, pretrained_model, pooling, dropout=0):
        super(SimcseModel_T5_GRU_CNN, self).__init__()
        # config = SimBertConfig.from_pretrained(pretrained_model)
        # 有监督也可以加入dropout
        config = AutoConfig.from_pretrained(pretrained_model)
        config.attention_probs_dropout_prob = dropout  # 修改config的dropout系数
        config.hidden_dropout_prob = dropout
        # tokenizer对句子进行编码[batch, max_len]，Encoder再对词进行编码[batch, max_len, embedding_size]
        self.T5 = T5EncoderModel.from_pretrained(pretrained_model, config=config)
        # self.T5 = AutoModel.from_pretrained(pretrained_model, config=config)
        self.pooling = pooling

        # 双向GRU参数
        self.gru_units = 256
        self.gru = nn.GRU(input_size=config.hidden_size, hidden_size=self.gru_units, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # TextCNN参数
        # 卷积核个数
        self.num_filters = 100
        # 每个卷积核的长宽深
        self.filter_sizes = [3, 4, 5]
        # 构建了三个卷积层
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1, self.num_filters, (k, config.hidden_size)) for k in self.filter_sizes
        ])

        # 全连接层（线性层）
        self.fc = nn.Linear(self.gru_units * 2 + len(self.filter_sizes) * self.num_filters, config.hidden_size)


    def forward(self, input_ids, attention_mask, token_type_ids):
        # 这里输入进来的input_ids, attention_mask, token_type_ids的维度经过转换为了[192,128]，原来[64,3,128]
        out = self.T5(input_ids, attention_mask, token_type_ids, output_hidden_states=True, return_dict=True)
        print(out.last_hidden_state.size())   # [192,128,768]

        # 双向GRU操作
        gru_output, _ = self.gru(out.last_hidden_state)
        gru_output = self.dropout(gru_output)
        print("GRU层之后的维度：", gru_output.size())

        # TextCNN操作
        cnn_output = out.last_hidden_state.unsqueeze(1)  # [batch, 1, 128, 768]
        cnn_output = [F.relu(conv(cnn_output)).squeeze(3) for conv in self.conv_layers]  # [batch, num_filters, seqlen - filter_size + 1]
        cnn_output = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in cnn_output]  # [batch, num_filters]
        cnn_output = torch.cat(cnn_output, 1)  # [batch, num_filters * len(filter_sizes)]
        cnn_output = self.dropout(cnn_output)
        print("TextCNN层之后的维度：", cnn_output.size())

        # 将GRU层和CNN层的输出拼接起来
        merged_output = torch.cat([gru_output[:, -1, :], cnn_output], dim=1)
        print("合并之后的维度：", merged_output.size())

        # 定义输出层
        outputs = self.fc(merged_output)
        print("输出层之后的维度：", outputs.size())

        return outputs

# word attention
class SimcseModel_T5_GRU_CNN_Attention(nn.Module):
    """Simcse有监督模型定义"""

    def __init__(self, pretrained_model, pooling, kesai, dropout=0):
        super(SimcseModel_T5_GRU_CNN_Attention, self).__init__()
        # config = SimBertConfig.from_pretrained(pretrained_model)
        # 有监督也可以加入dropout
        config = AutoConfig.from_pretrained(pretrained_model)
        config.attention_probs_dropout_prob = dropout  # 修改config的dropout系数
        config.hidden_dropout_prob = dropout
        # tokenizer对句子进行编码[batch, max_len]，Encoder再对词进行编码[batch, max_len, embedding_size]
        self.T5 = T5EncoderModel.from_pretrained(pretrained_model, config=config)
        # self.T5 = AutoModel.from_pretrained(pretrained_model, config=config)
        self.pooling = pooling
        # kesai超参数
        self.kesai = kesai
        self.attention_size = 128
        # 双向GRU参数
        self.gru_units = 150
        self.gru = nn.GRU(input_size=config.hidden_size, hidden_size=self.gru_units, num_layers=1, bidirectional=True, batch_first=True)
        self.gru_attention = nn.Sequential(
            nn.Linear(self.gru_units*2, self.attention_size),
            nn.Tanh(),
            nn.Linear(self.attention_size, 1)
        )
        self.dropout = nn.Dropout(dropout)

        # TextCNN参数
        # 卷积核个数
        self.num_filters = 100
        # 每个卷积核的长，这里有三个卷积层，100个卷积核
        self.filter_sizes = [3, 4, 5]
        # 构建了三个卷积层
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1, self.num_filters, (k, config.hidden_size)) for k in self.filter_sizes
        ])
        self.cnn_attention = nn.Sequential(
            nn.Linear(self.num_filters*3, self.attention_size),
            nn.Tanh(),
            nn.Linear(self.attention_size, 1)
        )
        # 全连接层（线性层）
        # self.fc = nn.Linear(self.gru_units * 2 + len(self.filter_sizes) * self.num_filters, config.hidden_size)
        self.fc = nn.Linear(self.gru_units * 2, config.hidden_size)


    def forward(self, input_ids, attention_mask, token_type_ids):
        # 这里输入进来的input_ids, attention_mask, token_type_ids的维度经过转换为了[192,128]，原来[64,3,128]
        out = self.T5(input_ids, attention_mask, token_type_ids, output_hidden_states=True, return_dict=True)
        # print(out.last_hidden_state.size())   # [192,32,1024]

        # 双向GRU操作
        gru_output, _ = self.gru(out.last_hidden_state)
        gru_output = self.dropout(gru_output)
        # [96, 32, 150*2]
        # print("GRU层之后的维度：", gru_output.size())

        # GRU层attention操作
        gru_attention_weights = self.gru_attention(gru_output).squeeze(2)
        # print("gru_attention_weights维度：", gru_attention_weights.size())#96,32
        gru_attention_weights = F.softmax(gru_attention_weights, dim=1)
        gru_attention_output = torch.bmm(gru_attention_weights.unsqueeze(1), gru_output).squeeze(1)
        # [96, 300]
        # print("GRU层attention之后的维度：", gru_attention_output.size())

        # TextCNN操作
        cnn_output = out.last_hidden_state.unsqueeze(1)  # [batch, 1, 32, 1024]
        cnn_output = [F.relu(conv(cnn_output)).squeeze(3) for conv in self.conv_layers]  # [batch, num_filters, seqlen - filter_size + 1]
        cnn_output = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in cnn_output]  # [batch, num_filters]
        cnn_output = torch.cat(cnn_output, 1)  # [batch, num_filters * len(filter_sizes)]
        cnn_output = self.dropout(cnn_output)
        # print("TextCNN层之后的维度：", cnn_output.size())

        # TextCNN层attention操作
        cnn_attention_weights = self.cnn_attention(cnn_output.unsqueeze(1)).squeeze(2)
        # print("cnn_attention_weights1:", cnn_attention_weights.size())
        cnn_attention_weights = F.softmax(cnn_attention_weights, dim=1) #归一化
        # print("cnn_attention_weights2:", cnn_attention_weights.size())
        cnn_attention_output = torch.bmm(cnn_attention_weights.unsqueeze(1), cnn_output.unsqueeze(1)).squeeze(1)
        # print("TextCNN层attention之后的维度：", cnn_attention_output.size())

        # 将GRU和TextCNN的输出拼接起来，然后加上一个线性层
        # concat_output = torch.cat([gru_attention_output, cnn_attention_output], dim=1)
        # 相加
        # concat_output = gru_attention_output + cnn_attention_output
        # Hadamard点乘
        # gru_attention_output = (1 - self.kesai) * gru_attention_output
        # cnn_attention_output = self.kesai * cnn_attention_output
        # concat_output = gru_attention_output * cnn_attention_output
        # 消融实验
        concat_output = cnn_attention_output
        concat_output = self.fc(concat_output)
        # print("拼接之后的维度：", concat_output.size())

        # 返回拼接后的张量
        return concat_output

# sentence attention
class SimcseModel_T5_GRU_CNN_Attention_new(nn.Module):
    """Simcse有监督模型定义"""

    def __init__(self, pretrained_model, pooling, dropout=0):
        super(SimcseModel_T5_GRU_CNN_Attention_new, self).__init__()
        # config = SimBertConfig.from_pretrained(pretrained_model)
        # 有监督也可以加入dropout
        config = AutoConfig.from_pretrained(pretrained_model)
        config.attention_probs_dropout_prob = dropout  # 修改config的dropout系数
        config.hidden_dropout_prob = dropout
        # tokenizer对句子进行编码[batch, max_len]，Encoder再对词进行编码[batch, max_len, embedding_size]
        self.T5 = T5EncoderModel.from_pretrained(pretrained_model, config=config)
        # self.T5 = AutoModel.from_pretrained(pretrained_model, config=config)
        self.pooling = pooling
        self.attention_size = 128
        # 双向GRU参数
        self.gru_units = 150
        self.gru = nn.GRU(input_size=config.hidden_size, hidden_size=self.gru_units, num_layers=1, bidirectional=True, batch_first=True)
        # 这里就是算注意力的权重值，然后softmax归一化后加权求和，一行就代表一个句子的表示了
        self.gru_attention = nn.Sequential(
            # 线性层的作用就类似于权重矩阵变换向量维度
            nn.Linear(self.gru_units*2, self.attention_size),
            nn.Tanh(),
            nn.Linear(self.attention_size, 1)
        )
        self.dropout = nn.Dropout(dropout)

        # TextCNN参数
        # 卷积核个数
        self.num_filters = 100
        # 每个卷积核的长，这里有三个卷积层，100个卷积核
        self.filter_sizes = [3, 4, 5]
        # 构建了三个卷积层
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1, self.num_filters, (k, config.hidden_size)) for k in self.filter_sizes
        ])
        self.cnn_attention = nn.Sequential(
            nn.Linear(self.num_filters*3, self.attention_size),
            nn.Tanh(),
            nn.Linear(self.attention_size, 1)
        )
        # 全连接层（线性层）
        # self.fc = nn.Linear(self.gru_units * 2 + len(self.filter_sizes) * self.num_filters, config.hidden_size)
        self.fc = nn.Linear(self.gru_units * 2, config.hidden_size)


    def forward(self, input_ids, attention_mask, token_type_ids):
        # 这里输入进来的input_ids, attention_mask, token_type_ids的维度经过转换为了[192,128]，原来[64,3,128]
        # 这里返回encoder所有隐藏层的输出，方便后面进行特征提取
        out = self.T5(input_ids, attention_mask, token_type_ids, output_hidden_states=True, return_dict=True)
        # print(out.last_hidden_state.size())   # [192,32,1024]

        # 双向GRU操作
        gru_output, _ = self.gru(out.last_hidden_state)
        # 这里取所有时间步的输出，能够更好地捕捉序列中的相关性和重要性
        gru_output = self.dropout(gru_output)
        # gru_output = gru_output[:, -1, :]
        # [96, 32, 150*2]
        # print("GRU层之后的维度：", gru_output.size())

        # GRU层attention操作
        # [96, 32]
        gru_attention_weights = self.gru_attention(gru_output).squeeze(2)
        # print("gru_attention_weights维度：", gru_attention_weights.size())#
        # [96, 32]
        gru_attention_weights = F.softmax(gru_attention_weights, dim=1)
        # [96, 300]
        # (batch_size, n, m) 和 (batch_size, m, p)，其中 n、m 和 p 分别表示矩阵的行数、列数和深度。bmm 的输出是一个形状为 (batch_size, n, p) 的三维张量，表示两个输入张量的矩阵乘积。
        gru_attention_output = torch.bmm(gru_attention_weights.unsqueeze(1), gru_output).squeeze(1)
        # print("GRU层attention之后的维度：", gru_attention_output.size())

        # TextCNN操作
        cnn_output = out.last_hidden_state.unsqueeze(1)  # [batch, 1, 32, 1024]
        #  这里的seqlen - filter_size + 1代表num_filters个卷积核按照不同filter_size提取的同一个句子的特征（关注的点不一样）
        cnn_output = [F.relu(conv(cnn_output)).squeeze(3) for conv in self.conv_layers]  # [batch, num_filters, seqlen - filter_size + 1]
        # 池化之后每个句子的特征的长度就是num_filters
        cnn_output = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in cnn_output]  # [batch, num_filters]
        # 这里再将三个不同size的filter给拼接起来
        cnn_output = torch.cat(cnn_output, 1)  # [batch, num_filters * len(filter_sizes)]
        # [96, 300]
        cnn_output = self.dropout(cnn_output)
        # print("TextCNN层之后的维度：", cnn_output.size())

        # TextCNN层attention操作
        # [96,1,300]
        cnn_attention_weights = self.cnn_attention(cnn_output.unsqueeze(1)).squeeze(2)
        print("cnn_attention_weights1:", cnn_attention_weights.size())
        cnn_attention_weights = F.softmax(cnn_attention_weights, dim=1) #归一化
        print("cnn_attention_weights2:", cnn_attention_weights.size())
        cnn_attention_output = torch.bmm(cnn_attention_weights.unsqueeze(1), cnn_output.unsqueeze(1)).squeeze(1)
        print("TextCNN层attention之后的维度：", cnn_attention_output.size())

        # 将GRU和TextCNN的输出拼接起来，然后加上一个线性层
        # concat_output = torch.cat([gru_attention_output, cnn_attention_output], dim=1)
        # 相加
        # concat_output = gru_attention_output + cnn_attention_output
        # Hadamard点乘
        concat_output = gru_attention_output * cnn_attention_output
        concat_output = self.fc(concat_output)
        # print("拼接之后的维度：", concat_output.size())

        # 返回拼接后的张量
        return concat_output



class Matching_network(nn.Module):
    """matching network"""

    def __init__(self, pretrained_model, pooling, dropout=0, hidden_size=384, num_layers=1):
        super(Matching_network, self).__init__()
        config = AutoConfig.from_pretrained(pretrained_model)
        config.attention_probs_dropout_prob = dropout
        config.hidden_dropout_prob = dropout
        self.jina = T5EncoderModel.from_pretrained(pretrained_model, config=config)
        self.bilstm = nn.LSTM(input_size=768, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.jina(input_ids, attention_mask, token_type_ids, output_hidden_states=True, return_dict=True)
        # print(f"jina输出向量维度：{out.last_hidden_state.size()} )     # [192,64,768]

        lstm_out, _ = self.bilstm(out.last_hidden_state)
        # if self.training:
        #     lstm_out = lstm_out.detach()  # 将BiLSTM的输出断开计算图
        # print(f"lstm输出向量维度：{lstm_out[:, -1, :].size()}")

        return lstm_out[:, -1, :]

class Matching_network_new(nn.Module):
    def __init__(self, pretrained_model, pooling, dropout=0):
        super(Matching_network_new, self).__init__()
        # config = SimBertConfig.from_pretrained(pretrained_model)
        # 有监督也可以加入dropout
        config = AutoConfig.from_pretrained(pretrained_model)
        config.attention_probs_dropout_prob = dropout  # 修改config的dropout系数
        config.hidden_dropout_prob = dropout
        # tokenizer对句子进行编码[batch, max_len]，Encoder再对词进行编码[batch, max_len, embedding_size]
        self.jina = T5EncoderModel.from_pretrained(pretrained_model, config=config)
        # self.T5 = AutoModel.from_pretrained(pretrained_model, config=config)
        self.pooling = pooling
        self.attention_size = 128
        # 双向GRU参数
        self.gru_units = 150
        self.gru = nn.GRU(input_size=config.hidden_size, hidden_size=self.gru_units, num_layers=1, bidirectional=True, batch_first=True)
        self.gru_attention = nn.Sequential(
            nn.Linear(self.gru_units*2, self.attention_size),
            nn.Tanh(),
            nn.Linear(self.attention_size, 1)
        )
        self.dropout = nn.Dropout(dropout)

        # TextCNN参数
        # 卷积核个数
        self.num_filters = 100
        # 每个卷积核的长，这里有三个卷积层，100个卷积核
        self.filter_sizes = [3, 4, 5]
        # 构建了三个卷积层
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1, self.num_filters, (k, config.hidden_size)) for k in self.filter_sizes
        ])
        self.cnn_attention = nn.Sequential(
            nn.Linear(self.num_filters*3, self.attention_size),
            nn.Tanh(),
            nn.Linear(self.attention_size, 1)
        )
        # 全连接层（线性层）
        # self.fc = nn.Linear(self.gru_units * 2 + len(self.filter_sizes) * self.num_filters, config.hidden_size)
        self.fc = nn.Linear(self.gru_units * 2, config.hidden_size)


    def forward(self, input_ids, attention_mask, token_type_ids):
        # 这里输入进来的input_ids, attention_mask, token_type_ids的维度经过转换为了[192,128]，原来[64,3,128]
        out = self.jina(input_ids, attention_mask, token_type_ids, output_hidden_states=True, return_dict=True)
        # print(out.last_hidden_state.size())   # [192,32,1024]

        # 双向GRU操作
        gru_output, _ = self.gru(out.last_hidden_state)
        # 这里取所有时间步的输出，能够更好地捕捉序列中的相关性和重要性
        gru_output = self.dropout(gru_output)
        # gru_output = gru_output[:, -1, :]
        # [96, 32, 256*2]
        # print("GRU层之后的维度：", gru_output.size())

        # GRU层attention操作
        # [96, 32]
        gru_attention_weights = self.gru_attention(gru_output).squeeze(2)
        # print("gru_attention_weights维度：", gru_attention_weights.size())#
        gru_attention_weights = F.softmax(gru_attention_weights, dim=1)
        gru_attention_output = torch.bmm(gru_attention_weights.unsqueeze(1), gru_output).squeeze(1)
        # [96, 300]
        # print("GRU层attention之后的维度：", gru_attention_output.size())

        # TextCNN操作
        cnn_output = out.last_hidden_state.unsqueeze(1)  # [batch, 1, 32, 1024]
        #  这里的seqlen - filter_size + 1代表num_filters个卷积核按照不同filter_size提取的同一个句子的特征（关注的点不一样）
        cnn_output = [F.relu(conv(cnn_output)).squeeze(3) for conv in self.conv_layers]  # [batch, num_filters, seqlen - filter_size + 1]
        # 池化之后每个句子的特征的长度就是num_filters
        cnn_output = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in cnn_output]  # [batch, num_filters]
        # 这里再将三个不同size的filter给拼接起来
        cnn_output = torch.cat(cnn_output, 1)  # [batch, num_filters * len(filter_sizes)]
        # [96, 300]
        cnn_output = self.dropout(cnn_output)
        # print("TextCNN层之后的维度：", cnn_output.size())

        # TextCNN层attention操作
        # [96,1]
        cnn_attention_weights = self.cnn_attention(cnn_output.unsqueeze(1)).squeeze(2)
        # print("cnn_attention_weights1:", cnn_attention_weights.size())
        # [96,1]
        cnn_attention_weights = F.softmax(cnn_attention_weights, dim=1) #归一化
        # print("cnn_attention_weights2:", cnn_attention_weights.size())
        # [96,300]
        cnn_attention_output = torch.bmm(cnn_attention_weights.unsqueeze(1), cnn_output.unsqueeze(1)).squeeze(1)
        # print("TextCNN层attention之后的维度：", cnn_attention_output.size())

        # 将GRU和TextCNN的输出拼接起来，然后加上一个线性层
        # concat_output = torch.cat([gru_attention_output, cnn_attention_output], dim=1)
        # 相加
        # concat_output = gru_attention_output + cnn_attention_output
        # Hadamard点乘
        concat_output = gru_attention_output * cnn_attention_output
        concat_output = self.fc(concat_output)
        # print("拼接之后的维度：", concat_output.size())

        # 返回拼接后的张量
        return concat_output


def simcse_unsup_loss(y_pred, device, temp=0.05):
    """无监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 2, 768]

    """
    # 得到y_pred对应的label, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
    y_true = torch.arange(y_pred.shape[0], device=device)
    y_true = (y_true - y_true % 2 * 2) + 1
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响

    sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12
    # 相似度矩阵除以温度系数
    sim = sim / temp
    # 计算相似度矩阵与y_true的交叉熵损失
    # 计算交叉熵，每个case都会计算与其他case的相似度得分，得到一个得分向量，目的是使得该得分向量中正样本的得分最高，负样本的得分最低
    loss = F.cross_entropy(sim, y_true)
    return torch.mean(loss)


def simcse_sup_loss(y_pred, device, lamda=0.05):
    """
    有监督损失函数
    """
    # torch.unsqueeze(input, dim)指在张量的某个维度前插入一个大小为1的维度
    # y_pred [192, 768] 这里每一行的768维是输出出来的句子向量，一共batch_size*3个句子（三种类型句子都在）
    # y_pred [1, 192, 768] 和 y_pred [192, 1, 768]进行没对行之间的计算
    # similarities [192, 192]对应行之间的相似度
    similarities = F.cosine_similarity(y_pred.unsqueeze(0), y_pred.unsqueeze(1), dim=2)
    # row tensor[0,3,6,...., 191]
    row = torch.arange(0, y_pred.shape[0], 3)
    # col tensor[0,1,2,...., 191]
    col = torch.arange(0, y_pred.shape[0])
    # 一个元素能够被 3 整除，则对应的布尔值为 False，否则为 True
    # [false, true, true, false, ...]
    col = col[col % 3 != 0]
    # [64, 192]
    similarities = similarities[row, :]
    # [64, 128]
    similarities = similarities[:, col]
    similarities = similarities / lamda
    # tensor[0,2,4,...190] 这里指的是列索引，similarity矩阵每一行从0,2,4开始，这些位置的值应尽量接近1
    y_true = torch.arange(0, len(col), 2, device=device)
    loss = F.cross_entropy(similarities, y_true)
    return loss


if __name__ == '__main__':
    y_pred = torch.rand((30, 16))
    print(y_pred.unsqueeze(0), y_pred.unsqueeze(1))
    loss = simcse_sup_loss(y_pred, 'cpu', lamda=0.05)
    print(loss)


