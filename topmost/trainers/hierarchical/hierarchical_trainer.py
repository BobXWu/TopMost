import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from collections import defaultdict
from tqdm import tqdm
from topmost.utils import static_utils


# transform tensor list to numpy list
def to_nparray(tensor_list):
    return np.asarray([item.detach().cpu().numpy() for item in tensor_list], dtype=object)


class HierarchicalTrainer:
    def __init__(self, model, dataset_handler, epochs=200, learning_rate=0.002, batch_size=200, lr_scheduler=None, lr_step_size=125):
        self.model = model
        self.dataset_handler = dataset_handler
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lr_scheduler = lr_scheduler
        self.lr_step_size = lr_step_size

    def make_optimizer(self,):
        args_dict = {
            'params': self.model.parameters(),
            'lr': self.learning_rate,
        }

        optimizer = torch.optim.Adam(**args_dict)
        return optimizer

    def make_lr_scheduler(self, optimizer,):
        lr_scheduler = StepLR(optimizer, step_size=self.lr_step_size, gamma=0.5, verbose=True)
        return lr_scheduler

    def train(self):
        optimizer = self.make_optimizer()

        if self.lr_scheduler:
            print("===>Warning: use lr_scheduler")
            lr_scheduler = self.make_lr_scheduler(optimizer)

        data_size = len(self.dataset_handler.train_dataloader.dataset)

        for epoch in tqdm(range(1, self.epochs + 1), leave=False):
            self.model.train()
            loss_rst_dict = defaultdict(float)

            for batch_data in self.dataset_handler.train_dataloader:

                rst_dict = self.model(batch_data)
                batch_loss = rst_dict['loss']

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                for key in rst_dict:
                    loss_rst_dict[key] += rst_dict[key] * len(batch_data)

            if self.lr_scheduler:
                lr_scheduler.step()

            output_log = f'Epoch: {epoch:03d}'
            for key in loss_rst_dict:
                output_log += f' {key}: {loss_rst_dict[key] / data_size :.3f}'

            print(output_log)

    # def test(self, bow):
    #     data_size = bow.shape[0]

    #     num_topics_list = self.model.num_topics_list
    #     hierarchical_theta_list = np.empty(len(num_topics_list), object)
    #     for layer_id in range(len(num_topics_list)):
    #         hierarchical_theta_list[layer_id] = np.zeros((data_size, num_topics_list[layer_id]))

    #     all_idx = torch.split(torch.arange(data_size), self.batch_size)

    #     with torch.no_grad():
    #         self.model.eval()
    #         for idx in all_idx:
    #             batch_input = bow[idx]
    #             batch_theta_list = self.model.get_theta(batch_input)

    #             for layer_id in range(len(num_topics_list)):
    #                 hierarchical_theta_list[layer_id][idx] = batch_theta_list[layer_id].cpu().numpy()

    #     return hierarchical_theta_list

    def test(self, bow):
        data_size = bow.shape[0]

        num_topics_list = self.model.num_topics_list
        theta_list = [list() for _ in range(len(num_topics_list))]
        all_idx = torch.split(torch.arange(data_size), self.batch_size)

        with torch.no_grad():
            self.model.eval()
            for idx in all_idx:
                batch_input = bow[idx]
                batch_theta_list = self.model.get_theta(batch_input)

                for layer_id in range(len(num_topics_list)):
                    theta_list[layer_id].extend(batch_theta_list[layer_id].cpu().numpy().tolist())

        theta = np.empty(len(num_topics_list), object)
        theta[:] = [np.asarray(item) for item in theta_list]
        return theta

    def export_phi(self):
        phi = to_nparray(self.model.get_phi_list())
        return phi

    def export_beta(self):
        beta_list = to_nparray(self.model.get_beta())
        return beta_list

    def export_top_words(self, num_top_words=15):
        beta = self.export_beta()
        top_words_list = list()
        for layer in range(beta.shape[0]):
            print(f"======= Layer: {layer} number of topics: {beta[layer].shape[0]} =======")
            top_words = static_utils.print_topic_words(beta[layer], vocab=self.dataset_handler.vocab, num_top_words=num_top_words)
            top_words_list.append(top_words)
        return top_words_list

    def export_theta(self):
        train_theta = self.test(self.dataset_handler.train_bow)
        test_theta = self.test(self.dataset_handler.test_bow)
        return train_theta, test_theta
