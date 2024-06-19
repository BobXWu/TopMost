import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
from torch.optim.lr_scheduler import StepLR
from topmost.utils import _utils
from topmost.utils.logger import Logger


logger = Logger("WARNING")


class DynamicTrainer:
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

    def make_optimizer(self,):
        args_dict = {
            'params': self.model.parameters(),
            'lr': self.learning_rate,
        }

        optimizer = torch.optim.Adam(**args_dict)
        return optimizer

    def make_lr_scheduler(self, optimizer):
        lr_scheduler = StepLR(optimizer, step_size=self.lr_step_size, gamma=0.5, verbose=False)
        return lr_scheduler

    def train(self):
        optimizer = self.make_optimizer()

        if self.lr_scheduler:
            logger.info("using lr_scheduler")
            lr_scheduler = self.make_lr_scheduler(optimizer)

        data_size = len(self.dataset.train_dataloader.dataset)

        for epoch in tqdm(range(1, self.epochs + 1)):
            self.model.train()
            loss_rst_dict = defaultdict(float)

            for batch_data in self.dataset.train_dataloader:

                rst_dict = self.model(batch_data['bow'], batch_data['times'])
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
        train_theta = self.test(self.dataset.train_bow, self.dataset.train_times)

        return top_words, train_theta

    def test(self, bow, times):
        data_size = bow.shape[0]
        theta = list()
        all_idx = torch.split(torch.arange(data_size), self.batch_size)

        with torch.no_grad():
            self.model.eval()
            for idx in all_idx:
                batch_theta = self.model.get_theta(bow[idx], times[idx])
                theta.extend(batch_theta.cpu().tolist())

        theta = np.asarray(theta)
        return theta

    def get_beta(self):
        self.model.eval()
        beta = self.model.get_beta().detach().cpu().numpy()
        return beta

    def get_top_words(self, num_top_words=None):
        if num_top_words is None:
            num_top_words = self.num_top_words

        beta = self.get_beta()
        top_words_list = list()
        for time in range(beta.shape[0]):
            if self.verbose:
                print(f"======= Time: {time} =======")
            top_words = _utils.get_top_words(beta[time], self.dataset.vocab, num_top_words, self.verbose)
            top_words_list.append(top_words)
        return top_words_list

    def export_theta(self):
        train_theta = self.test(self.dataset.train_bow, self.dataset.train_times)
        test_theta = self.test(self.dataset.test_bow, self.dataset.test_times)

        return train_theta, test_theta
