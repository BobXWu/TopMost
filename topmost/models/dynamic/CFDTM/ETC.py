import torch
import torch.nn as nn
import torch.nn.functional as F


class ETC(nn.Module):
    def __init__(self, num_times, temperature, weight_neg, weight_pos):
        super().__init__()
        self.num_times = num_times
        self.weight_neg = weight_neg
        self.weight_pos = weight_pos
        self.temperature = temperature

    def forward(self, topic_embeddings):
        loss = 0.
        loss_neg = 0.
        loss_pos = 0.

        for t in range(self.num_times):
            loss_neg += self.compute_loss(topic_embeddings[t], topic_embeddings[t], self.temperature, self_contrast=True)

        for t in range(1, self.num_times):
            loss_pos += self.compute_loss(topic_embeddings[t], topic_embeddings[t - 1].detach(), self.temperature, self_contrast=False, only_pos=True)

        loss_neg *= (self.weight_neg / self.num_times)
        loss_pos *= (self.weight_pos / (self.num_times - 1))
        loss = loss_neg + loss_pos

        return loss

    def compute_loss(self, anchor_feature, contrast_feature, temperature, self_contrast=False, only_pos=False, all_neg=False):
        # KxK
        anchor_dot_contrast = torch.div(
            torch.matmul(F.normalize(anchor_feature, dim=1), F.normalize(contrast_feature, dim=1).T),
            temperature
        )

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        pos_mask = torch.eye(anchor_dot_contrast.shape[0]).to(anchor_dot_contrast.device)

        if self_contrast is False:
            if only_pos is False:
                if all_neg is True:
                    exp_logits = torch.exp(logits)
                    sum_exp_logits = exp_logits.sum(1)
                    log_prob = -torch.log(sum_exp_logits + 1e-12)

                    mean_log_prob = -log_prob.sum() / (logits.shape[0] * logits.shape[1])
            else:
                # only pos
                mean_log_prob = -(logits * pos_mask).sum() / pos_mask.sum()
        else:
            # self contrast: push away from each other in the same time slice.
            exp_logits = torch.exp(logits) * (1 - pos_mask)
            sum_exp_logits = exp_logits.sum(1)
            log_prob = -torch.log(sum_exp_logits + 1e-12)

            mean_log_prob = -log_prob.sum() / (1 - pos_mask).sum()

        return mean_log_prob
