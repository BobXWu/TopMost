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
    def __init__(self, model, epochs=200, learning_rate=0.002, batch_size=200, lr_scheduler=None, lr_step_size=125, log_interval=5):
        self.model = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lr_scheduler = lr_scheduler
        self.lr_step_size = lr_step_size
        self.log_interval = log_interval

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

    def fit_transform(self, dataset_handler, num_top_words=15, verbose=False):
        self.train(dataset_handler, verbose)
        top_words = self.export_top_words(dataset_handler.vocab, num_top_words)
        train_theta = self.test(dataset_handler.train_data)

        return top_words, train_theta

    def train(self, dataset_handler, verbose=False):
        optimizer = self.make_optimizer()

        if self.lr_scheduler:
            print("===>using lr_scheduler")
            lr_scheduler = self.make_lr_scheduler(optimizer)

        data_size = len(dataset_handler.train_dataloader.dataset)

        for epoch in tqdm(range(1, self.epochs + 1), leave=False):
            self.model.train()
            loss_rst_dict = defaultdict(float)

            for batch_data in dataset_handler.train_dataloader:

                rst_dict = self.model(batch_data)
                batch_loss = rst_dict['loss']

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                for key in rst_dict:
                    loss_rst_dict[key] += rst_dict[key] * len(batch_data)

            if self.lr_scheduler:
                lr_scheduler.step()

            if verbose and epoch % self.log_interval == 0:
                output_log = f'Epoch: {epoch:03d}'
                for key in loss_rst_dict:
                    output_log += f' {key}: {loss_rst_dict[key] / data_size :.3f}'

                print(output_log)

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

    def export_top_words(self, vocab, num_top_words=15, annotation=False):
        beta = self.export_beta()
        top_words_list = list()

        for layer in range(beta.shape[0]):
            print(f"======= Layer: {layer} number of topics: {beta[layer].shape[0]} =======")
            top_words = static_utils.print_topic_words(beta[layer], vocab, num_top_words=num_top_words)

            if not annotation:
                top_words_list.append(top_words)
            else:
                top_words_list.extend([f'L-{layer}_K-{k} {item}' for k, item in enumerate(top_words)])

        return top_words_list

    def export_theta(self, dataset_handler):
        train_theta = self.test(dataset_handler.train_data)
        test_theta = self.test(dataset_handler.test_data)

        return train_theta, test_theta
