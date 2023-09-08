import topmost


device = "cuda" # or "cpu"

output_dir = "./datasets/NYT"
# topmost.preprocessing.download_20ng.download_save(output_dir=output_dir)

topmost.data.download_dataset('NYT', cache_path=output_dir)

preprocessing = topmost.preprocessing.Preprocessing(vocab_size=5000)
# preprocessing.parse_dataset(dataset_dir=output_dir, label_name="group")
# preprocessing.save(output_dir="./datasets/20NG")

########################### Dataset ####################################
# load a preprocessed dataset
dataset_dir = f'./datasets/preprocessed_data/20NG'
dataset_handler = topmost.data.BasicDatasetHandler(dataset_dir, device)


# ########################### NMF from gensim ####################################
# trainer = topmost.runners.NMFGensimRunner(dataset_handler, vocab_size=dataset_handler.vocab_size, max_iter=200)
# trainer.train()


########################### NMF from scikit-learn ####################################
# LDA scikit-learn
from sklearn.decomposition import NMF
model = NMF(n_components=50)

trainer = topmost.trainers.NMFSklearnTrainer(model, dataset_handler)
trainer.train()

train_theta, test_theta = trainer.export_theta()

top_words = trainer.export_top_words(num_top=15)

import IPython
IPython.embed()
exit()

########################### LDA ####################################
# LDA gensim
# trainer = topmost.runners.LDAGensimRunner(dataset_handler, vocab_size=dataset_handler.vocab_size)
# trainer.train()

# LDA scikit-learn
from sklearn.decomposition import LatentDirichletAllocation
model = LatentDirichletAllocation(n_components=50)

trainer = topmost.runners.LDASklearnRunner(model, dataset_handler)
trainer.train()

########################### Neural Topic Models ####################################
# for neural topic models
dataset_handler = topmost.data.BasicDatasetHandler(dataset_dir, device, as_tensor=True)
# create a model
# model = topmost.models.ProdLDA(vocab_size=dataset_handler.vocab_size)
# model = topmost.models.TSCTM(vocab_size=dataset_handler.vocab_size)
# model = topmost.models.ETM(vocab_size=dataset_handler.vocab_size, pretrained_WE=dataset_handler.pretrained_WE)
# model = topmost.models.NSTM(vocab_size=dataset_handler.vocab_size, pretrained_WE=dataset_handler.pretrained_WE)
model = topmost.models.ECRTM(vocab_size=dataset_handler.vocab_size, pretrained_WE=dataset_handler.pretrained_WE)
model = model.to(device)

# create a trainer
trainer = topmost.trainers.BasicTrainer(model, dataset_handler)

# train the model
trainer.train()

########################### Evaluate ####################################
# evaluate
# get theta (doc-topic distributions)
train_theta, test_theta = trainer.export_theta()

# get top words of topics
top_words = trainer.export_top_words(num_top=15)



# evaluate topic coherence


# evaluate topic diversity
TD = topmost.evaluations.compute_topic_diversity(top_words, _type="TD")
print(f"TD: {TD:.5f}")

# evaluate clustering
results = topmost.evaluations.evaluate_clustering(test_theta, dataset_handler.test_labels)
print(results)

# evaluate classification
results = topmost.evaluations.evaluate_classification(train_theta, test_theta, dataset_handler.train_labels, dataset_handler.test_labels)
print(results)

# Visualization

# test other documents
import torch

new_texts = [
    "This is a new document about space, including words like space, satellite, launch, orbit.",
    "This is a new document about Microsoft Windows, including words like windows, files, dos."
]

parsed_new_texts, new_bow = preprocessing.parse(new_texts, vocab=dataset_handler.vocab)
new_theta = trainer.test(torch.as_tensor(new_bow, device=device).float())
