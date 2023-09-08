import torch
import torch.nn as nn


# Topic-Semantic Contrastive Learning
class TSC(nn.Module):
    def __init__(self, temperature=0.07, weight_contrast=None, use_aug=False):
        super().__init__()
        self.use_aug = use_aug
        self.temperature = temperature
        self.weight_contrast = weight_contrast

    def forward(self, features, quant_idx=None, weight_same_quant=None):
        device = features.device

        batch_size = features.shape[0]
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases.
        # logits_mask is 1 - eye matrix
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask

        t_quant_idx = quant_idx.contiguous().view(-1, 1)

        # quant_idx_mask: 1 means same quantization; 0 means different quantization
        quant_idx_mask = torch.eq(t_quant_idx, t_quant_idx.T).float()
        quant_idx_mask = quant_idx_mask.repeat(anchor_count, contrast_count)

        exp_logits = torch.exp(logits) * (1 - quant_idx_mask)
        sum_exp_logits = exp_logits.sum(1, keepdim=True)

        if not self.use_aug:
            # quant_idx_mask includes self-contrast cases.
            # logits * logits_mask is to remove the positive pair but keep the negative pair in the self-contrast cases.
            # This is because some samples do not have positive pairs.
            log_prob = logits * logits_mask - torch.log(sum_exp_logits + 1e-10)
            mean_log_prob_pos = (quant_idx_mask * log_prob).sum(1) / quant_idx_mask.sum(1)
        else:
            log_prob = logits - torch.log(sum_exp_logits + 1e-10)
            # between original and augmented samples.
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

            # between original samples.
            same_quant_mask = quant_idx_mask * logits_mask
            same_quant_mean_log_prob_pos = (same_quant_mask * log_prob).sum(1) / (same_quant_mask.sum(1) + 1e-10)
            mean_log_prob_pos += weight_same_quant * same_quant_mean_log_prob_pos

        loss = - self.weight_contrast * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).sum(axis=0).mean()

        return loss
