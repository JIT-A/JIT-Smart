import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiFocalLoss(nn.Module):
    """
    Focal_Loss= -1*alpha*((1-pt)**gamma)*log(pt)
    Args:
        num_class: number of classes
        alpha: class balance factor shape=[num_class, ]
        gamma: hyper-parameter
        reduction: reduction type
    """

    def __init__(self, num_class, alpha=None, gamma=2, reduction='mean'):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = 1e-4
        self.alpha = alpha
        if alpha is None:
            self.alpha = torch.ones(num_class, ) - 0.5
        elif isinstance(alpha, (int, float)):
            self.alpha = torch.as_tensor([alpha] * num_class)
        elif isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.as_tensor(alpha)
        if self.alpha.shape[0] != num_class:
            raise RuntimeError('the length not equal to number of class')

    def forward(self, logit, target):
        # assert isinstance(self.alpha,torch.Tensor)\
        alpha = self.alpha.to(logit.device)
        prob = F.softmax(logit, dim=1)

        if prob.dim() > 2:
            # used for 3d-conv:  N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            N, C = logit.shape[:2]
            prob = prob.view(N, C, -1)
            prob = prob.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            prob = prob.view(-1, prob.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]

        ori_shp = target.shape
        target = target.view(-1, 1)

        prob = prob.gather(1, target).view(-1) + self.smooth  # avoid nan
        logpt = torch.log(prob)
        # alpha_class = alpha.gather(0, target.squeeze(-1))
        alpha_weight = alpha[target.squeeze().long()]
        loss = -alpha_weight * torch.pow(torch.sub(1.0, prob), self.gamma) * logpt

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'none':
            loss = loss.view(ori_shp)

        return loss


class Attention(nn.Module):  # x:[batch, seq_len, hidden_dim*2]
    """
        此注意力的计算步骤：
        1.将输入（包含lstm的所有时刻的状态输出）和w矩阵进行矩阵相乘，然后用tanh压缩到(-1, 1)之间
        2.然后再和矩阵u进行矩阵相乘后，矩阵变为1维，然后进行softmax变化即得到注意力得分。
        3.将输入和此注意力得分线性加权，即相当于将所有时刻的状态进行了一个聚合。
    """

    def __init__(self, hidden_size, need_aggregation=True):
        super().__init__()
        self.need_aggregation = need_aggregation
        # 不双向的话就不用乘2
        self.w = nn.Parameter(torch.Tensor(hidden_size * 2, hidden_size * 2))
        self.u = nn.Parameter(torch.Tensor(hidden_size * 2, 1))
        nn.init.uniform_(self.w, -0.1, 0.1)
        nn.init.uniform_(self.u, -0.1, 0.1)

    def forward(self, x):
        device = x.device
        self.w = self.w.to(device)
        self.u = self.u.to(device)

        u = torch.tanh(torch.matmul(x, self.w))  # [batch, seq_len, hidden_size*2]
        score = torch.matmul(u, self.u)  # [batch, seq_len, 1]
        att = F.softmax(score, dim=1)
        # 下面操作即线性加权
        scored_x = x * att  # [batch, seq_len, hidden_size*2]

        # 因为词encoder和句encoder后均带有attention机制，而我需要做的是代码行级缺陷检测，
        # 所以句encoder后我不做聚合，相当于将每个代码行看做一个样本来传入全连接分类。
        if self.need_aggregation:
            context = torch.sum(scored_x, dim=1)  # [batch, hidden_size*2]
            return context
        else:
            return scored_x


class HAN_MODEL(nn.Module):
    def __init__(self, embedding_layer):
        super().__init__()

        self.hidden_size = 256
        self.num_layers = 1
        self.bidirectional = True

        self.embedding = embedding_layer

        self.lstm1 = nn.LSTM(input_size=self.embedding.embedding_dim,
                             hidden_size=self.hidden_size,
                             num_layers=self.num_layers,
                             bidirectional=self.bidirectional,
                             batch_first=True)
        self.att1 = Attention(self.hidden_size, need_aggregation=True)
        self.lstm2 = nn.LSTM(input_size=self.hidden_size * 2,
                             hidden_size=self.hidden_size,
                             num_layers=self.num_layers,
                             bidirectional=self.bidirectional,
                             batch_first=True)
        self.att2 = Attention(self.hidden_size, need_aggregation=False)

        # 代码行级分类输出层，代码有多少行，输出就有多少个神经元
        # self.fc1 = nn.Linear(512, 2)
        self.fc1 = nn.Linear(512, 128)
        self.relu = nn.Tanh()
        self.fc2 = nn.Linear(128, 2)
        # self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # input x : (bs, nus_sentences, nums_words)
        device = x.device
        x = self.embedding(x)  # out x : (bs, nus_sentences, nums_words, embedding_dim)
        x = self.dropout(x)
        batch_size, num_sentences, num_words, emb_dim = x.shape

        # 初始化：双向就乘2
        h0_1 = torch.randn(self.num_layers * 2, batch_size * num_sentences, self.hidden_size).to(device)
        c0_1 = torch.randn(self.num_layers * 2, batch_size * num_sentences, self.hidden_size).to(device)
        h0_2 = torch.randn(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        c0_2 = torch.randn(self.num_layers * 2, batch_size, self.hidden_size).to(device)

        # 先进入word的处理，将每句话的所有单词的表示通过attention聚合成一个表示
        # 将batch_size, num_sentences两个维度乘起来看成“batch_size”。使用.contiguous()方法首先拷贝了一份张量在内存中的地址，然后将地址按照形状改变后的张量的语义进行排列。
        # touch.view()方法对张量改变“形状”其实并没有改变张量在内存中真正的形状
        x = x.view(batch_size * num_sentences, num_words, emb_dim).contiguous()
        x, (_, _) = self.lstm1(x, (h0_1, c0_1))  # out：batch_size*num_sentences, num_words，hidden_size*2
        x = self.att1(x)  # 线性加权注意力后的输出：batch_size*num_sentences, hidden_size*2

        x = x.view(x.size(0) // num_sentences, num_sentences, self.hidden_size * 2).contiguous()
        x, (_, _) = self.lstm2(x, (h0_2, c0_2))  # out：batch_size, num_sentences，hidden_size*2

        x = self.att2(x)  # 线性加权注意力后的输出：batch_size, hidden_size*2
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


#class RobertaClassificationHead(nn.Module):
 #   """Head for sentence-level classification tasks."""

  #  def __init__(self, config):
   #     super().__init__()
    #    self.manual_dense = nn.Linear(config.feature_size, config.hidden_size)
     #   self.dropout = nn.Dropout(config.hidden_dropout_prob)
      #  self.out_proj_new = nn.Linear(config.hidden_size + config.hidden_size, 2)

    #def forward(self, features, manual_features=None, **kwargs):
     #   x = features[:, 0, :]  # take <s> token (equiv. to [CLS])  [bs,hidden_size]
      #  y = manual_features.float()  # [bs, feature_size]

       # y = self.manual_dense(y)
        #y = torch.tanh(y)

        #x = torch.cat((x, y), dim=-1)
        #x = self.dropout(x)
        #x = self.out_proj_new(x)
        #return x


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        #self.manual_dense = nn.Linear(config.feature_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj_new = nn.Linear(config.hidden_size, 2)

    def forward(self, features, manual_features=None, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])  [bs,hidden_size]
        #y = manual_features.float()  # [bs, feature_size]

        #y = self.manual_dense(y)
        #y = torch.tanh(y)

        #x = torch.cat((x, y), dim=-1)
        x = self.dropout(x)
        x = self.out_proj_new(x)
        return x








class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args

        # ----------------------HAN-------------------------------
        self.han_word_embedding_layer = self.encoder.embeddings.word_embeddings
        self.han_locator = HAN_MODEL(embedding_layer=self.han_word_embedding_layer)

        # --------------------------------------------------------

        # self.fusion_fc = nn.Linear(4,2)

    def forward(self, inputs_ids, attn_masks, manual_features,
                labels, line_ids, line_label, output_attentions=None):

        outputs = self.encoder(input_ids=inputs_ids, attention_mask=attn_masks, output_attentions=output_attentions)

        last_layer_attn_weights = outputs.attentions[self.config.num_hidden_layers - 1][:, :,
                                  0].detach() if output_attentions else None

        logits = self.classifier(outputs[0], manual_features)
        han_logits = self.han_locator(line_ids)

        # logits = self.fusion_fc(torch.cat((logits, han_outputs), dim=-1))
        logits = (logits + han_logits.mean(dim=1)) / 2

        if labels is not None:

            loss_dp = MultiFocalLoss(alpha=0.25, gamma=2, reduction='mean', num_class=2)
            loss1 = loss_dp(logits, labels)

            loss_dl = MultiFocalLoss(alpha=0.25, gamma=2, reduction='mean', num_class=2)
            loss2 = loss_dl(han_logits.reshape((-1, 2)), line_label.reshape((-1,)))
            # loss = (loss1 + loss2) / 2
            loss = loss1*self.args.dp_loss_weight + loss2*self.args.dl_loss_weight

            return loss, torch.softmax(logits, dim=1)[:, 1].unsqueeze(1), last_layer_attn_weights, torch.softmax(
                han_logits, dim=-1)[:, :, 1]  # shape: (bs, line_nums:256)
        else:
            # return torch.sigmoid(logits)[:, 1].unsqueeze(1)
            return torch.softmax(logits, dim=1)[:, 1].unsqueeze(1)

