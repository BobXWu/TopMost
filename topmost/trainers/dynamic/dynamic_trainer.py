import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from collections import defaultdict
from tqdm import tqdm
from topmost.utils import static_utils


class DynamicTrainer:
    def __init__(self, model, dataset_handler, epochs=200, learning_rate=0.002, batch_size=200, lr_scheduler=None, lr_step_size=125, log_interval=5):
        self.model = model
        self.dataset_handler = dataset_handler
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

    def make_lr_scheduler(self, optimizer):
        lr_scheduler = StepLR(optimizer, step_size=self.lr_step_size, gamma=0.5, verbose=False)
        return lr_scheduler

    def train(self):
        optimizer = self.make_optimizer()

        if self.lr_scheduler:
            print("===>Warning: use lr_scheduler")
            lr_scheduler = self.make_lr_scheduler(optimizer)

        data_size = len(self.dataset_handler.train_dataloader.dataset)

        for epoch in tqdm(range(1, self.epochs + 1)):
            self.model.train()
            loss_rst_dict = defaultdict(float)

            for batch_data in self.dataset_handler.train_dataloader:

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

                print(output_log)

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

    def export_beta(self):
        self.model.eval()
        beta = self.model.get_beta().detach().cpu().numpy()
        return beta

    def export_top_words(self, num_top_words=15):
        beta = self.export_beta()
        top_words_list = list()
        for time in range(beta.shape[0]):
            print(f"======= Time: {time} =======")
            top_words = static_utils.print_topic_words(beta[time], vocab=self.dataset_handler.vocab, num_top_words=num_top_words)
            top_words_list.append(top_words)
        return top_words_list

    def export_theta(self):
        train_theta = self.test(self.dataset_handler.train_bow, self.dataset_handler.train_times)
        test_theta = self.test(self.dataset_handler.test_bow, self.dataset_handler.test_times)
        return train_theta, test_theta
