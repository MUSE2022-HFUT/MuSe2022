import torch
import torch.nn as nn
import torch.nn.functional as F

class CCCLoss(nn.Module):
    def __init__(self):
        super(CCCLoss, self).__init__()

    def forward(self, y_pred, y_true, seq_lens=None):
        if seq_lens is not None:
            mask = torch.ones_like(y_true, device=y_true.device)
            for i, seq_len in enumerate(seq_lens):
                mask[i, seq_len:] = 0
        else:
            mask = torch.ones_like(y_true, device=y_true.device)
        y_true_mean = torch.sum(y_true * mask, dim=1, keepdim=True) / torch.sum(mask, dim=1, keepdim=True)
        y_pred_mean = torch.sum(y_pred * mask, dim=1, keepdim=True) / torch.sum(mask, dim=1, keepdim=True)

        y_true_var = torch.sum(mask * (y_true - y_true_mean) ** 2, dim=1, keepdim=True) / torch.sum(mask, dim=1,
                                                                                                    keepdim=True)
        y_pred_var = torch.sum(mask * (y_pred - y_pred_mean) ** 2, dim=1, keepdim=True) / torch.sum(mask, dim=1,
                                                                                                    keepdim=True)

        cov = torch.sum(mask * (y_true - y_true_mean) * (y_pred - y_pred_mean), dim=1, keepdim=True) \
              / torch.sum(mask, dim=1, keepdim=True)

        ccc = torch.mean(2.0 * cov / (y_true_var + y_pred_var + (y_true_mean - y_pred_mean) ** 2), dim=0)
        ccc = ccc.squeeze(0)
        ccc_loss = 1.0 - ccc

        return ccc_loss

# wraps BCEWithLogitsLoss, but constructor accepts (and ignores) argument seq_lens
class WrappedBCEWithLogitsLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true, seq_lens=None):
        return self.loss_fn(y_pred, y_true)


class WrappedMSELoss(nn.Module):

    def __init__(self, reduction):
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction=reduction)

    def forward(self, y_pred, y_true, seq_lens=None):
        return self.loss_fn(y_pred, y_true)


# wraps BCELoss, but constructor accepts (and ignores) argument seq_lens
class WrappedBCELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCELoss()

    def forward(self, y_pred, y_true, seq_lens=None):
        return self.loss_fn(y_pred, y_true)


# 针对二分类任务的 Focal Loss
class WrappedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, size_average=True):
        super(WrappedFocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha).cuda()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, pred, target, seq_lens=None):
        # 如果模型最后没有 nn.Sigmoid()，那么这里就需要对预测结果计算一次 Sigmoid 操作
        # pred = nn.Sigmoid()(pred)

        # 展开 pred 和 target,此时 pred.size = target.size = (BatchSize,1) 
        pred = pred.view(-1,1)
        target = target.view(-1,1)

        # 此处将预测样本为正负的概率都计算出来，此时 pred.size = (BatchSize,2)
        pred = torch.cat((1-pred,pred),dim=1)

        # 根据 target 生成 mask，即根据 ground truth 选择所需概率
        # 用大白话讲就是：
        # 当标签为 1 时，我们就将模型预测该样本为正类的概率代入公式中进行计算
        # 当标签为 0 时，我们就将模型预测该样本为负类的概率代入公式中进行计算
        class_mask = torch.zeros(pred.shape[0],pred.shape[1]).cuda()
        # 这里的 scatter_ 操作不常用，其函数原型为:
        # scatter_(dim,index,src)->Tensor
        # Writes all values from the tensor src into self at the indices specified in the index tensor. 
        # For each value in src, its output index is specified by its index in src for dimension != dim and by the corresponding value in index for dimension = dim.
        class_mask.scatter_(1, target.view(-1, 1).long(), 1.)

        # 利用 mask 将所需概率值挑选出来
        probs = (pred * class_mask).sum(dim=1).view(-1,1)
        probs = probs.clamp(min=0.0001,max=1.0)

        # 计算概率的 log 值
        log_p = probs.log()

        # 根据论文中所述，对 alpha　进行设置（该参数用于调整正负样本数量不均衡带来的问题）
        alpha = torch.ones(pred.shape[0],pred.shape[1]).cuda()
        alpha[:,0] = alpha[:,0] * (1-self.alpha)
        alpha[:,1] = alpha[:,1] * self.alpha
        alpha = (alpha * class_mask).sum(dim=1).view(-1,1)

        # 根据 Focal Loss 的公式计算 Loss
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p

         # Loss Function的常规操作，mean 与 sum 的区别不大，相当于学习率设置不一样而已
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss


def get_segment_wise_labels(labels):
    # collapse labels to one label per segment (as originally for MuSe-Sent)
    segment_labels = []
    for i in range(labels.size(0)):
        segment_labels.append(labels[i, 0, :])
    labels = torch.stack(segment_labels).long()
    return labels


def get_segment_wise_logits(logits, feature_lens):
    # determines exactly one output for each segment (by taking the last timestamp of each segment)
    segment_logits = []
    for i in range(logits.size(0)):
        segment_logits.append(logits[i, feature_lens[i] - 1, :])  # (batch-size, frames, classes)
    logits = torch.stack(segment_logits, dim=0)
    return logits
