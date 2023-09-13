
============
Quick Start
============

Install
-----------------

Install topmost with ``pip`` as 

.. code-block:: console

    $ pip install topmost


Download a preprocessed dataset
-----------------------------------
Download a preprocessed dataset from our github repo:

.. code-block:: python

    import topmost
    from topmost.data import download_dataset

    dataset_dir = "./datasets/20NG"
    download_dataset('20NG', cache_path='./datasets')


Train a model
-----------------------------------

.. code-block:: python

    device = "cuda" # or "cpu"

    # load a preprocessed dataset
    dataset = topmost.data.BasicDatasetHandler(dataset_dir, device=device, read_labels=True, as_tensor=True)
    # create a model
    model = topmost.models.ETM(vocab_size=dataset.vocab_size, pretrained_WE=dataset.pretrained_WE)
    model = model.to(device)

    # create a trainer
    trainer = topmost.trainers.BasicTrainer(model, dataset)

    # train the model
    trainer.train()


Evaluate
-----------------------------------

.. code-block:: python

    # evaluate
    # get theta (doc-topic distributions)
    train_theta, test_theta = trainer.export_theta()
    # get top words of topics
    top_words = trainer.export_top_words()

    # evaluate topic diversity
    TD = topmost.evaluations.compute_topic_diversity(top_words, _type="TD")
    print(f"TD: {TD:.5f}")

    # evaluate clustering
    results = topmost.evaluations.evaluate_clustering(test_theta, dataset.test_labels)
    print(results)

    # evaluate classification
    results = topmost.evaluations.evaluate_classification(train_theta, test_theta, dataset.train_labels, dataset.test_labels)
    print(results)


Test new documents (Optional)
-----------------------------------

.. code-block:: python

    # test new documents
    import torch

    new_docs = [
        "This is a new document about space, including words like space, satellite, launch, orbit.",
        "This is a new document about Microsoft Windows, including words like windows, files, dos."
    ]

    parsed_new_docs, new_bow = preprocessing.parse(new_docs, vocab=dataset.vocab)
    new_theta = runner.test(torch.as_tensor(new_bow, device=device).float())

