import numpy as np
import topmost


device = "cuda" # or "cpu"

dataset_dir = './datasets/preprocessed_data/NYT'



# # ====================== HDP ==============================
dataset = topmost.data.BasicDatasetHandler(dataset_dir, read_labels=True)
trainer = topmost.trainers.HDPGensimTrainer(dataset)
trainer.train()
top_words = trainer.export_top_words()
train_theta, test_theta = trainer.export_theta()



# load a preprocessed dataset
dataset = topmost.data.BasicDatasetHandler(dataset_dir, read_labels=True, device=device, as_tensor=True)
# # ====================== SawETM ==============================
# model = topmost.models.SawETM(vocab_size=dataset.vocab_size, num_topics_list=[10, 50, 200], device=device)
# model = model.to(device)

# ====================== HyperMiner ==============================
model = topmost.models.HyperMiner(vocab_size=dataset.vocab_size, num_topics_list=[10, 50, 200], device=device)
model = model.to(device)

trainer = topmost.trainers.HierarchicalTrainer(model, dataset, epochs=2)

trainer.train()

# evaluate
# get top words of topics
top_words = trainer.export_top_words(num_top_words=15)
# get theta (doc-topic distributions)
train_theta, test_theta = trainer.export_theta()

# compute topic coherence


# compute topic diversity
TD_list = list()
for layer_top_words in top_words:
    TD = topmost.evaluations.compute_topic_diversity(layer_top_words)
    TD_list.append(TD)

TD = topmost.evaluations.compute_multiaspect_topic_diversity(top_words)
print(f"TD: {TD}")

# evaluate classification
results = topmost.evaluations.evaluate_hierarchical_classification(train_theta, test_theta, dataset.train_labels, dataset.test_labels)
print(results)
