import torch
import torch.nn as nn
import torch.nn.functional as F


class TAMI(nn.Module):
    '''
        InfoCTM: A Mutual Information Maximization Perspective of Cross-lingual Topic Modeling. AAAI 2023

        Xiaobao Wu, Xinshuai Dong, Thong Nguyen, Chaoqun Liu, Liangming Pan, Anh Tuan Luu
    '''
    def __init__(self, temperature, weight_MI, pos_threshold, trans_e2c, pretrain_word_embeddings_en, pretrain_word_embeddings_cn):
        super().__init__()
        self.temperature = temperature
        self.weight_MI = weight_MI
        self.pos_threshold = pos_threshold
        self.pretrain_word_embeddings_en = pretrain_word_embeddings_en
        self.pretrain_word_embeddings_cn = pretrain_word_embeddings_cn

        self.trans_e2c = torch.as_tensor(trans_e2c).float()
        self.trans_e2c = nn.Parameter(self.trans_e2c, requires_grad=False)
        self.trans_c2e = self.trans_e2c.T

        pos_trans_mask_en, pos_trans_mask_cn, neg_trans_mask_en, neg_trans_mask_cn = self.compute_pos_neg(pretrain_word_embeddings_en, pretrain_word_embeddings_cn, self.trans_e2c, self.trans_c2e)
        self.pos_trans_mask_en = nn.Parameter(pos_trans_mask_en, requires_grad=False)
        self.pos_trans_mask_cn = nn.Parameter(pos_trans_mask_cn, requires_grad=False)
        self.neg_trans_mask_en = nn.Parameter(neg_trans_mask_en, requires_grad=False)
        self.neg_trans_mask_cn = nn.Parameter(neg_trans_mask_cn, requires_grad=False)

    def build_CVL_mask(self, embeddings):
        norm_embed = F.normalize(embeddings)
        cos_sim = torch.matmul(norm_embed, norm_embed.T)
        pos_mask = (cos_sim >= self.pos_threshold).float()
        return pos_mask

    def translation_mask(self, mask, trans_dict_matrix):
        # V1 x V2
        trans_mask = torch.matmul(mask, trans_dict_matrix)
        return trans_mask

    def compute_pos_neg(self, pretrain_word_embeddings_en, pretrain_word_embeddings_cn, trans_e2c, trans_c2e):
        # Ve x Ve
        pos_mono_mask_en = self.build_CVL_mask(torch.as_tensor(pretrain_word_embeddings_en))
        # Vc x Vc
        pos_mono_mask_cn = self.build_CVL_mask(torch.as_tensor(pretrain_word_embeddings_cn))

        # Ve x Vc
        pos_trans_mask_en = self.translation_mask(pos_mono_mask_en, trans_e2c)
        pos_trans_mask_cn = self.translation_mask(pos_mono_mask_cn, trans_c2e)

        neg_trans_mask_en = (pos_trans_mask_en <= 0).float()
        neg_trans_mask_cn = (pos_trans_mask_cn <= 0).float()

        return pos_trans_mask_en, pos_trans_mask_cn, neg_trans_mask_en, neg_trans_mask_cn

    def MutualInfo(self, anchor_feature, contrast_feature, mask, neg_mask):
        anchor_dot_contrast = torch.div(
            torch.matmul(F.normalize(anchor_feature, dim=1), F.normalize(contrast_feature, dim=1).T),
            self.temperature
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits) * neg_mask
        sum_exp_logits = exp_logits.sum(1, keepdim=True)

        log_prob = logits - torch.log(sum_exp_logits + torch.exp(logits) + 1e-10)
        mean_log_prob = -(mask * log_prob).sum()
        return mean_log_prob

    def forward(self, fea_en, fea_cn):
        loss_TAMI = self.MutualInfo(fea_en, fea_cn, self.pos_trans_mask_en, self.neg_trans_mask_en)
        loss_TAMI += self.MutualInfo(fea_cn, fea_en, self.pos_trans_mask_cn, self.neg_trans_mask_cn)

        loss_TAMI = loss_TAMI / (self.pos_trans_mask_en.sum() + self.pos_trans_mask_cn.sum())

        loss_TAMI = self.weight_MI * loss_TAMI
        return loss_TAMI
