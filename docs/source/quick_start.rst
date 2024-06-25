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

    import topmost
    from topmost.data import RawDataset
    from topmost.preprocessing import Preprocessing
    from sklearn.datasets import fetch_20newsgroups

    docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
    preprocessing = Preprocessing(vocab_size=10000, stopwords='English')

    device = 'cuda' # or 'cpu'
    dataset = RawDataset(docs, preprocessing, device=device)

    trainer = topmost.trainers.FASTopicTrainer(dataset, verbose=True)
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
    from topmost.data import download_dataset

    download_dataset('20NG', cache_path='./datasets')


Train a model
-----------------------------------

.. code-block:: python

    device = "cuda" # or "cpu"

    # load a preprocessed dataset
    dataset = topmost.data.BasicDataset("./datasets/20NG", device=device, read_labels=True)
    # create a model
    model = topmost.models.ProdLDA(dataset.vocab_size)
    model = model.to(device)

    # create a trainer
    trainer = topmost.trainers.BasicTrainer(model, dataset)

    # train the model
    top_words, train_theta = trainer.train()


Evaluate
-----------------------------------

.. code-block:: python

    # evaluate topic diversity
    TD = topmost.evaluations.compute_topic_diversity(top_words)

    # get doc-topic distributions of testing samples
    test_theta = trainer.test(dataset.test_data)
    # evaluate clustering
    clustering_results = topmost.evaluations.evaluate_clustering(test_theta, dataset.test_labels)
    # evaluate classification
    classification_results = topmost.evaluations.evaluate_classification(train_theta, test_theta, dataset.train_labels, dataset.test_labels)



Test new documents
-----------------------------------

.. code-block:: python

    import torch
    from topmost.preprocessing import Preprocessing

    new_docs = [
        "This is a new document about space, including words like space, satellite, launch, orbit.",
        "This is a new document about Microsoft Windows, including words like windows, files, dos."
    ]

    preprocessing = Preprocessing()
    new_parsed_docs, new_bow = preprocessing.parse(new_docs, vocab=dataset.vocab)
    new_theta = trainer.test(torch.as_tensor(new_bow, device=device).float())

