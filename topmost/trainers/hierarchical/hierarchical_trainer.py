import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
from torch.optim.lr_scheduler import StepLR
from topmost.utils import _utils
from topmost.utils.logger import Logger


logger = Logger("WARNING")


# transform tensor list to numpy list
def to_nparray(tensor_list):
    return np.asarray([item.detach().cpu().numpy() for item in tensor_list], dtype=object)


class HierarchicalTrainer:
    def __init__(self,
                 model,
                 dataset,
                 num_top_words=15,
                 epochs=200,
                 learning_rate=0.002,
                 batch_size=200,
                 lr_scheduler=None,
                 lr_step_size=125,
                 log_interval=5,
                 verbose=False
                ):

        self.model = model
        self.dataset = dataset
        self.num_top_words = num_top_words
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lr_scheduler = lr_scheduler
        self.lr_step_size = lr_step_size
        self.log_interval = log_interval

        self.verbose = verbose
        if verbose:
            logger.set_level("DEBUG")
        else:
            logger.set_level("WARNING")

    def make_optimizer(self):
        args_dict = {
            'params': self.model.parameters(),
            'lr': self.learning_rate,
        }

        optimizer = torch.optim.Adam(**args_dict)
        return optimizer

    def make_lr_scheduler(self, optimizer):
        lr_scheduler = StepLR(optimizer, step_size=self.lr_step_size, gamma=0.5, verbose=self.verbose)
        return lr_scheduler

    def train(self):
        optimizer = self.make_optimizer()

        if self.lr_scheduler:
            logger.info("using lr_scheduler")
            lr_scheduler = self.make_lr_scheduler(optimizer)

        data_size = len(self.dataset.train_dataloader.dataset)

        for epoch in tqdm(range(1, self.epochs + 1), leave=False):
            self.model.train()
            loss_rst_dict = defaultdict(float)

            for batch_data in self.dataset.train_dataloader:

                rst_dict = self.model(batch_data)
                batch_loss = rst_dict['loss']

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                for key in rst_dict:
                    loss_rst_dict[key] += rst_dict[key] * len(batch_data)

            if self.lr_scheduler:
                lr_scheduler.step()

            if epoch % self.log_interval == 0:
                output_log = f'Epoch: {epoch:03d}'
                for key in loss_rst_dict:
                    output_log += f' {key}: {loss_rst_dict[key] / data_size :.3f}'

                logger.info(output_log)

        top_words = self.get_top_words()
        train_theta = self.test(self.dataset.train_data)

        return top_words, train_theta

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

    def get_phi(self):
        phi = to_nparray(self.model.get_phi_list())
        return phi

    def get_beta(self):
        beta_list = to_nparray(self.model.get_beta())
        return beta_list

    def get_top_words(self, num_top_words=None, annotation=False):
        if num_top_words is None:
            num_top_words = self.num_top_words

        beta = self.get_beta()
        top_words_list = list()

        for layer in range(beta.shape[0]):
            if self.verbose:
                print(f"======= Layer: {layer} number of topics: {beta[layer].shape[0]} =======")
            top_words = _utils.get_top_words(beta[layer], self.dataset.vocab, num_top_words, self.verbose)

            if not annotation:
                top_words_list.append(top_words)
            else:
                top_words_list.extend([f'L-{layer}_K-{k} {item}' for k, item in enumerate(top_words)])

        return top_words_list

    def export_theta(self):
        train_theta = self.test(self.dataset.train_data)
        test_theta = self.test(self.dataset.test_data)

        return train_theta, test_theta
