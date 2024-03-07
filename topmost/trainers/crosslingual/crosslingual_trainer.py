import torch
from torch.optim.lr_scheduler import StepLR
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from topmost.utils import static_utils


class CrosslingualTrainer:
    def __init__(self, model, epochs=500, learning_rate=0.002, batch_size=200, lr_scheduler=None, lr_step_size=125, log_interval=5):
        self.model = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lr_scheduler = lr_scheduler
        self.lr_step_size = lr_step_size
        self.log_interval = log_interval

    def make_optimizer(self):
        args_dict = {
            'params': self.model.parameters(),
            'lr': self.learning_rate
        }

        optimizer = torch.optim.Adam(**args_dict)
        return optimizer

    def make_lr_scheduler(self, optimizer):
        if self.lr_scheduler == 'StepLR':
            print("===>using lr_scheduler")
            lr_scheduler = StepLR(optimizer, step_size=self.lr_step_size, gamma=0.5, verbose=False)
        else:
            raise NotImplementedError(self.lr_scheduler)

        return lr_scheduler

    def fit_transform(self, dataset_handler, num_top_words=15, verbose=False):
        self.train(dataset_handler, verbose)
        top_words_en, top_words_cn = self.export_top_words(dataset_handler.vocab_en, dataset_handler.vocab_cn, num_top_words)
        train_theta_en, train_theta_cn = self.test(dataset_handler.train_bow_en, dataset_handler.train_bow_cn)

        return top_words_en, top_words_cn, train_theta_en, train_theta_cn


    def train(self, dataset_handler, verbose=False):
        data_size = len(dataset_handler.train_dataloader.dataset)
        optimizer = self.make_optimizer()

        if self.lr_scheduler:
            lr_scheduler = self.make_lr_scheduler(optimizer)

        for epoch in tqdm(range(1, self.epochs + 1)):


            loss_rst_dict = defaultdict(float)

            self.model.train()
            for batch_data in dataset_handler.train_dataloader:
                batch_bow_en = batch_data['bow_en']
                batch_bow_cn = batch_data['bow_cn']
                params_list = [batch_bow_en, batch_bow_cn]

                rst_dict = self.model(*params_list)

                batch_loss = rst_dict['loss']

                for key in rst_dict:
                    if 'loss' in key:
                        loss_rst_dict[key] += rst_dict[key]

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            if self.lr_scheduler:
                lr_scheduler.step()

            if verbose and epoch % self.log_interval == 0:
                output_log = f'Epoch: {epoch:03d}'
                for key in loss_rst_dict:
                    output_log += f' {key}: {loss_rst_dict[key] / data_size :.3f}'

                print(output_log)

    def get_theta(self, bow, lang):
        theta_list = list()
        data_size = bow.shape[0]
        all_idx = torch.split(torch.arange(data_size,), self.batch_size)
        with torch.no_grad():
            self.model.eval()
            for idx in all_idx:
                batch_bow = bow[idx]
                theta = self.model.get_theta(batch_bow, lang)
                theta_list.extend(theta.detach().cpu().numpy().tolist())

        return np.asarray(theta_list)

    def test(self, bow_en, bow_cn):
        theta_en = self.get_theta(bow_en, lang='en')
        theta_cn = self.get_theta(bow_cn, lang='cn')

        return theta_en, theta_cn

    def export_beta(self):
        beta_en, beta_cn = self.model.get_beta()
        beta_en = beta_en.detach().cpu().numpy()
        beta_cn = beta_cn.detach().cpu().numpy()

        return beta_en, beta_cn

    def export_top_words(self, vocab_en, vocab_cn, num_top_words=15):
        beta_en, beta_cn = self.export_beta()
        top_words_en = static_utils.print_topic_words(beta_en, vocab_en, num_top_words)
        top_words_cn = static_utils.print_topic_words(beta_cn, vocab_cn, num_top_words)

        return top_words_en, top_words_cn

    def export_theta(self, dataset_handler):
        train_theta_en, train_theta_cn = self.test(dataset_handler.train_bow_en, dataset_handler.train_bow_cn)
        test_theta_en, test_theta_cn = self.test(dataset_handler.test_bow_en, dataset_handler.test_bow_cn)

        return train_theta_en, train_theta_cn, test_theta_en, test_theta_cn
