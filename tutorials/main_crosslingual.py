import topmost


device = "cuda" # or "cpu"

dataset_dir = './topmost/datasets/crosslingual_data/'
lang1 = 'en'
lang2 = 'cn'
dict_path = f'{dataset_dir}/dict/ch_en_dict.dat'

# load a preprocessed dataset
dataset_handler = topmost.data.CrosslingualDatasetHandler(dataset_dir, "ECNews", lang1, lang2, dict_path, device)

# create a model
model = topmost.models.InfoCTM(
    trans_e2c=dataset_handler.trans_matrix_en,
    pretrain_word_embeddings_en=dataset_handler.pretrain_word_embeddings_en,
    pretrain_word_embeddings_cn=dataset_handler.pretrain_word_embeddings_cn,
    vocab_size_en=dataset_handler.vocab_size_en,
    vocab_size_cn=dataset_handler.vocab_size_cn
)

model = model.to(device)

# create a trainer
trainer = topmost.trainers.CrosslingualTrainer(model, dataset_handler, epochs=2)

# train the model
trainer.train()

# evaluate
# get theta (doc-topic distributions)
train_theta_en, train_theta_cn, test_theta_en, test_theta_cn = trainer.export_theta()

# get top words of topics
top_words_en, top_words_cn = trainer.export_top_words(num_top=15)

# compute topic coherence (CNPMI)
# refer to https://github.com/BobXWu/CNPMI

# compute topic diversity
TD_en = topmost.evaluations.compute_topic_diversity(top_words_en)
TD_cn = topmost.evaluations.compute_topic_diversity(top_words_cn)
print(f"TD: {(TD_en + TD_cn) / 2:.5f}")

# evaluate classification
results = topmost.evaluations.evaluate_crosslingual_classification(
    train_theta_en,
    train_theta_cn,
    test_theta_en,
    test_theta_cn,
    dataset_handler.train_labels_en,
    dataset_handler.train_labels_cn,
    dataset_handler.test_labels_en,
    dataset_handler.test_labels_cn,
    classifier="SVM"
)

print(results)
