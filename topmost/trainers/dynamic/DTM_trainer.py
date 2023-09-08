import gensim
import numpy as np
from gensim.models import ldaseqmodel
from tqdm import tqdm
import datetime
from multiprocessing.pool import Pool
from topmost.utils import static_utils


def work(arguments):
    model, docs = arguments
    theta_list = list()
    for doc in tqdm(docs):
        theta_list.append(model[doc])
    return theta_list


class DTMTrainer:
    def __init__(self, dataset_handler, num_topics=50, alphas=0.01, chain_variance=0.005, passes=10, lda_inference_max_iter=25, em_min_iter=6, em_max_iter=20):
        self.dataset_handler = dataset_handler
        self.vocab_size = dataset_handler.vocab_size
        self.num_topics = num_topics
        self.alphas = alphas
        self.chain_variance = chain_variance
        self.passes = passes
        self.lda_inference_max_iter = lda_inference_max_iter
        self.em_min_iter = em_min_iter
        self.em_max_iter = em_max_iter

    def train(self):
        id2word = dict(zip(range(self.vocab_size), self.dataset_handler.vocab))
        train_bow = self.dataset_handler.train_bow
        train_times = self.dataset_handler.train_times.astype('int32')

        # order documents by time slices
        self.doc_order_idx = np.argsort(train_times)
        train_bow = train_bow[self.doc_order_idx]
        time_slices = np.bincount(train_times)

        corpus = gensim.matutils.Dense2Corpus(train_bow, documents_columns=False)

        self.model = ldaseqmodel.LdaSeqModel(
            corpus=corpus,
            id2word=id2word,
            time_slice=time_slices,
            num_topics=self.num_topics,
            alphas=self.alphas,
            chain_variance=self.chain_variance,
            em_min_iter=self.em_min_iter,
            em_max_iter=self.em_max_iter,
            lda_inference_max_iter=self.lda_inference_max_iter,
            passes=self.passes
        )

    def test(self, bow):
        # bow = dataset.bow.cpu().numpy()
        # times = dataset.times.cpu().numpy()
        corpus = gensim.matutils.Dense2Corpus(bow, documents_columns=False)

        num_workers = 20
        split_idx_list = np.array_split(np.arange(len(bow)), num_workers)
        worker_size_list = [len(x) for x in split_idx_list]
        print("===>worker_size_list: ", worker_size_list)

        worker_id = 0
        docs_list = [list() for i in range(num_workers)]
        for i, doc in enumerate(corpus):
            docs_list[worker_id].append(doc)
            if len(docs_list[worker_id]) >= worker_size_list[worker_id]:
                worker_id += 1

        args_list = list()
        for docs in docs_list:
            args_list.append([self.model, docs])

        starttime = datetime.datetime.now()

        pool = Pool(processes=num_workers)
        results = pool.map(work, args_list)

        pool.close()
        pool.join()

        theta_list = list()
        for rst in results:
            theta_list.extend(rst)

        endtime = datetime.datetime.now()

        print("DTM test time: {}s".format((endtime - starttime).seconds))

        return np.asarray(theta_list)

    def get_theta(self):
        theta = self.model.gammas / self.model.gammas.sum(axis=1)[:, np.newaxis]
        # NOTE: MUST transform gamma to original order.
        return theta[np.argsort(self.doc_order_idx)]

    def export_beta(self):
        beta = list()
        # K x V x T
        for item in self.model.topic_chains:
            # V x T
            beta.append(item.e_log_prob)

        # T x K x V
        beta = np.transpose(np.asarray(beta), (2, 0, 1))
        # use softmax
        beta = np.exp(beta)
        beta = beta / beta.sum(-1, keepdims=True)
        return beta

    def export_top_words(self, num_top=15):
        beta = self.export_beta()
        top_words_list = list()
        for time in range(beta.shape[0]):
            top_words = static_utils.print_topic_words(beta[time], vocab=self.dataset_handler.vocab, num_top=num_top)
            top_words_list.append(top_words)
        return top_words_list

    def export_theta(self):
        train_theta = self.get_theta()
        test_theta = self.test(self.dataset_handler.test_bow)
        return train_theta, test_theta
