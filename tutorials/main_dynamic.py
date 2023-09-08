import numpy as np
import topmost


device = "cuda" # or "cpu"

dataset_dir = './datasets/preprocessed_data/NYT'

# load a preprocessed dataset


# # ====================== DTM ==============================
# dataset = topmost.data.DynamicDatasetHandler(dataset_dir)
# # create a trainer
# trainer = topmost.trainers.DTMTrainer(dataset, passes=0, num_topics=1)

# # train the model
# trainer.train()

# trainer.export_top_words()

# ====================== DETM ==============================
dataset = topmost.data.DynamicDatasetHandler(dataset_dir, read_labels=True, device=device, as_tensor=True)

# create a model
model = topmost.models.DETM(
    vocab_size=dataset.vocab_size,
    num_times=dataset.num_times,
    train_size=dataset.train_size,
    train_time_wordfreq=dataset.train_time_wordfreq,
    device=device,
)
model = model.to(device)

trainer = topmost.trainers.DynamicTrainer(model, dataset, epochs=2)

trainer.train()

# evaluate
# get top words of topics
top_words_list = trainer.export_top_words(num_top_words=15)
# get theta (doc-topic distributions)
train_theta, test_theta = trainer.export_theta()

# compute topic coherence
dynamic_TC = topmost.evaluations.compute_dynamic_TC(dataset.train_texts, dataset.train_times.cpu().numpy(), dataset.vocab, top_words_list)

# compute topic diversity
TD_list = list()
for top_words in top_words_list:
    TD = topmost.evaluations.compute_topic_diversity(top_words)
    TD_list.append(TD)

print(f"TD: {np.mean(TD_list)}")

# evaluate classification
results = topmost.evaluations.evaluate_classification(train_theta, test_theta, dataset.train_labels, dataset.test_labels)
print(results)
