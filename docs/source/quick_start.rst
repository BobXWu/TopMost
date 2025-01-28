============
Quick Start
============

Install TopMost
-----------------

Install topmost with ``pip`` as 

.. code-block:: console

    $ pip install topmost

-------------------------------------------

We try FASTopic_ to get the top words of discovered topics, ``topic_top_words`` and the topic distributions of documents, ``doc_topic_dist``.
The preprocessing steps are configurable. See our documentations.

.. code-block:: python

    from topmost import RawDataset, Preprocess, FASTopicTrainer
    from sklearn.datasets import fetch_20newsgroups

    docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
    preprocess = Preprocess(vocab_size=10000)

    dataset = RawDataset(docs, preprocess, device="cuda")

    trainer = FASTopicTrainer(dataset, verbose=True)
    top_words, doc_topic_dist = trainer.train()

    new_docs = [
        "This is a document about space, including words like space, satellite, launch, orbit.",
        "This is a document about Microsoft Windows, including words like windows, files, dos."
    ]

    new_theta = trainer.test(new_docs)
    print(new_theta.argmax(1))


.. _FASTopic: https://arxiv.org/pdf/2405.17978


============
Usage
============

Download a preprocessed dataset
-----------------------------------

.. code-block:: python

    import topmost

    topmost.download_dataset('20NG', cache_path='./datasets')



Train a model
-----------------------------------

.. code-block:: python

    device = "cuda" # or "cpu"

    # load a preprocessed dataset
    dataset = topmost.BasicDataset("./datasets/20NG", device=device, read_labels=True)
    # create a model
    model = topmost.ProdLDA(dataset.vocab_size)
    model = model.to(device)

    # create a trainer
    trainer = topmost.BasicTrainer(model, dataset)

    # train the model
    top_words, train_theta = trainer.train()


Evaluate
-----------------------------------

.. code-block:: python

    from topmost import eva

    # topic diversity and coherence
    TD = eva._diversity(top_words)
    TC = eva._coherence(dataset.train_texts, dataset.vocab, top_words)

    # get doc-topic distributions of testing samples
    test_theta = trainer.test(dataset.test_data)
    # clustering
    clustering_results = eva._clustering(test_theta, dataset.test_labels)
    # classification
    cls_results = eva._cls(train_theta, test_theta, dataset.train_labels, dataset.test_labels)


Test new documents
-----------------------------------

.. code-block:: python

    import torch
    from topmost import Preprocess

    new_docs = [
        "This is a new document about space, including words like space, satellite, launch, orbit.",
        "This is a new document about Microsoft Windows, including words like windows, files, dos."
    ]

    preprocess = Preprocess()
    new_parsed_docs, new_bow = preprocess.parse(new_docs, vocab=dataset.vocab)
    new_theta = trainer.test(torch.as_tensor(new_bow.toarray(), device=device).float())
