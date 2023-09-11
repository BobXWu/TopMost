*************
Quick Start
*************

.. code-block::python

  {"label": "rec.autos", "text": "WHAT car is this!?..."}
  {"label": "comp.sys.mac.hardware", "text": "A fair number of brave souls who upgraded their..."}


.. code-block::python

    import topmost

    preprocessing = topmost.preprocessing.Preprocessing(stopwords_dir="...")
    preprocessing.parse(dataset_dir="...", label_name="label")
    preprocessing.save(output_dir="...")




.. code-block::python

    import topmost

    device = "cuda" # or "cpu"

    # load a preprocessed dataset
    dataset_handler = topmost.data.StaticDatasetHandler("20NG", device)
    # create a model
    model = topmost.models.ETM(vocab_size=dataset_handler.vocab_size, pretrained_WE=dataset_handler.pretrained_WE)
    model = model.to(device)

    # create a runner
    runner = topmost.runners.StaticRunner(model, dataset_handler, epochs=2)
    # train the model
    runner.train()



.. code-block:: python

    # evaluate
    # get theta (doc-topic distributions)
    train_theta, test_theta = runner.export_theta()

    # get top words of topics
    top_words = runner.export_top_words(num_top=15)

    # evaluate clustering
    results = topmost.evaluations.evaluate_clustering(test_theta, dataset_handler.test_labels)
    print(results)

    # evaluate classification
    results = topmost.evaluations.evaluate_classification(train_theta, test_theta, dataset_handler.train_labels, dataset_handler.test_labels)
    print(results)

    # evaluate topic coherence

    # evaluate topic diversity
    TD = topmost.evaluations.compute_topic_diversity(top_words, _type="TD")
    print(f"TD: {TD:.5f}")
