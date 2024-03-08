============
Quick Start
============

Install TopMost
-----------------

Install topmost with ``pip`` as 

.. code-block:: console

    $ pip install topmost


Discover topics from your own datasets
-----------------------------------

We can get the top words of discovered topics, ``topic_top_words``` and the topic distributions of documents, ``doc_topic_dist``.
The preprocessing steps are configurable. See our documentations.

.. code-block:: python

    import torch
    import topmost
    from topmost.preprocessing import Preprocessing

    # Your own documents
    docs = [
        "This is a document about space, including words like space, satellite, launch, orbit.",
        "This is a document about Microsoft Windows, including words like windows, files, dos.",
        # more documents...
    ]

    device = 'cuda' # or 'cpu'
    preprocessing = Preprocessing()
    dataset = topmost.data.RawDatasetHandler(docs, preprocessing, device=device, as_tensor=True)

    model = topmost.models.ProdLDA(dataset.vocab_size, num_topics=2)
    model = model.to(device)

    trainer = topmost.trainers.BasicTrainer(model)

    topic_top_words, doc_topic_dist = trainer.fit_transform(dataset, num_top_words=15, verbose=False)




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
    dataset = topmost.data.BasicDatasetHandler("./datasets/20NG", device=device, read_labels=True, as_tensor=True)
    # create a model
    model = topmost.models.ProdLDA(dataset.vocab_size)
    model = model.to(device)

    # create a trainer
    trainer = topmost.trainers.BasicTrainer(model)

    # train the model
    trainer.train(dataset)


Evaluate
-----------------------------------

.. code-block:: python

    # get theta (doc-topic distributions)
    train_theta, test_theta = trainer.export_theta(dataset)
    # get top words of topics
    topic_top_words = trainer.export_top_words(dataset.vocab)

    # evaluate topic diversity
    TD = topmost.evaluations.compute_topic_diversity(top_words)

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

    parsed_new_docs, new_bow = preprocessing.parse(new_docs, vocab=dataset.vocab)
    new_doc_topic_dist = trainer.test(torch.as_tensor(new_bow, device=device).float())
