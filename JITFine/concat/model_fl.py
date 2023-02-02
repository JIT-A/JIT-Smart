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


class BinaryFocalLoss(nn.Module):
    """
    Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param reduction: `none`|`mean`|`sum`
    """

    def __init__(self, alpha=1, gamma=2, reduction='mean', **kwargs):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6  # set '1e-4' when train with FP16
        self.reduction = reduction

        assert self.reduction in ['none', 'mean', 'sum']

    def forward(self, output, target):
        prob = torch.sigmoid(output)
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        target = target.unsqueeze(dim=1)
        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()

        pos_weight = (pos_mask * torch.pow(1 - prob, self.gamma)).detach()
        pos_loss = -pos_weight * torch.log(prob)  # / (torch.sum(pos_weight) + 1e-4)

        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        neg_loss = -self.alpha * neg_weight * F.logsigmoid(-output)  # / (torch.sum(neg_weight) + 1e-4)

        loss = pos_loss + neg_loss
        loss = loss.mean()
        return loss






class BinaryDSCLoss(torch.nn.Module):
    r"""
    Creates a criterion that optimizes a multi-class Self-adjusting Dice Loss
    ("Dice Loss for Data-imbalanced NLP Tasks" paper)

    Args:
        alpha (float): a factor to push down the weight of easy examples
        gamma (float): a factor added to both the nominator and the denominator for smoothing purposes
        reduction (string): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.

    Shape:
        - logits: `(N, C)` where `N` is the batch size and `C` is the number of classes.
        - targets: `(N)` where each value is in [0, C - 1]
    """

    def __init__(self, alpha: float = 1.0, smooth: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = torch.gather(probs, dim=1, index=targets.unsqueeze(1).to(dtype=torch.int64))

        targets = targets.unsqueeze(dim=1)
        pos_mask = (targets == 1).float()
        neg_mask = (targets == 0).float()

        pos_weight = pos_mask * ((1 - probs) ** self.alpha) * probs
        pos_loss = 1 - (2 * pos_weight + self.smooth) / (pos_weight + 1 + self.smooth)

        neg_weight = neg_mask * ((1 - probs) ** self.alpha) * probs
        neg_loss = 1 - (2 * neg_weight + self.smooth) / (neg_weight + self.smooth)

        loss = pos_loss + neg_loss
        loss = loss.mean()
        return loss





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


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.manual_dense = nn.Linear(config.feature_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj_new = nn.Linear(config.hidden_size + config.hidden_size, 2)

    def forward(self, features, manual_features=None, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])  [bs,hidden_size]
        y = manual_features.float()  # [bs, feature_size]
        y = self.manual_dense(y)
        y = torch.tanh(y)

        x = torch.cat((x, y), dim=-1)
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

    def forward(self, inputs_ids, attn_masks, manual_features=None,
                labels=None, output_attentions=None):
        outputs = \
            self.encoder(input_ids=inputs_ids, attention_mask=attn_masks, output_attentions=output_attentions)

        last_layer_attn_weights = outputs.attentions[self.config.num_hidden_layers - 1][:, :,
                                  0].detach() if output_attentions else None

        logits = self.classifier(outputs[0], manual_features)

        #prob = torch.sigmoid(logits)

        #prob = torch.softmax(logits, dim=1)  # 按列SoftMax,列和为1

        if labels is not None:

            # loss_fct = BCELoss()
            # print('prob.shape: ', prob.shape)
            # print('labels.unsqueeze(1).shape: ', labels.unsqueeze(1).shape)
            # loss_fct = BinaryFocalLoss(alpha=0.25, gamma=1, reduction='mean')
            loss_fct = MultiFocalLoss(alpha=0.25, gamma=2, reduction='mean', num_class=2)
            #loss_fct = BinaryDSCLoss()
            loss = loss_fct(logits, labels)
            #return loss, prob[:, 1].unsqueeze(1), last_layer_attn_weights
            # return loss, torch.sigmoid(logits)[:, 1].unsqueeze(1), last_layer_attn_weights
            return loss, torch.softmax(logits,dim=1)[:, 1].unsqueeze(1), last_layer_attn_weights
        else:
            # return torch.sigmoid(logits)[:, 1].unsqueeze(1)
            return torch.softmax(logits, dim=1)[:, 1].unsqueeze(1)

