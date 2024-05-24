import torch
import torch.nn as nn


class UWE(nn.Module):
    def __init__(self, ETC, num_times, temperature, weight_UWE, neg_topk):
        super().__init__()

        self.ETC = ETC
        self.weight_UWE = weight_UWE
        self.num_times = num_times
        self.temperature = temperature
        self.neg_topk = neg_topk

    def forward(self, time_wordcount, beta, topic_embeddings, word_embeddings):
        assert(self.num_times == time_wordcount.shape[0])

        topk_indices = self.get_topk_indices(beta)

        loss_UWE = 0.
        cnt_valid_times = 0.
        for t in range(self.num_times):
            neg_idx = torch.where(time_wordcount[t] == 0)[0]

            time_topk_indices = topk_indices[t]
            neg_idx = list(set(neg_idx.cpu().tolist()).intersection(set(time_topk_indices.cpu().tolist())))
            neg_idx = torch.tensor(neg_idx).long().to(time_wordcount.device)

            if len(neg_idx) == 0:
                continue

            time_neg_WE = word_embeddings[neg_idx]

            # topic_embeddings[t]: K x D
            # word_embeddings[neg_idx]: |V_{neg}| x D
            loss_UWE += self.ETC.compute_loss(topic_embeddings[t], time_neg_WE, temperature=self.temperature, all_neg=True)
            cnt_valid_times += 1

        if cnt_valid_times > 0:
            loss_UWE *= (self.weight_UWE / cnt_valid_times)

        return loss_UWE

    def get_topk_indices(self, beta):
        # topk_indices: T x K x neg_topk
        topk_indices = torch.topk(beta, k=self.neg_topk, dim=-1).indices
        topk_indices = torch.flatten(topk_indices, start_dim=1)
        return topk_indices
