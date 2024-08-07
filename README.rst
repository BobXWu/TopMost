|topmost-logo| TopMost
=================================

.. |topmost-logo| image:: docs/source/_static/topmost-logo.png
    :width: 38

.. image:: https://img.shields.io/github/stars/bobxwu/topmost?logo=github
        :target: https://github.com/bobxwu/topmost/stargazers
        :alt: Github Stars

.. image:: https://static.pepy.tech/badge/topmost
        :target: https://pepy.tech/project/topmost
        :alt: Downloads

.. image:: https://img.shields.io/pypi/v/topmost
        :target: https://pypi.org/project/topmost
        :alt: PyPi

.. image:: https://readthedocs.org/projects/topmost/badge/?version=latest
    :target: https://topmost.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/github/license/bobxwu/topmost
        :target: https://www.apache.org/licenses/LICENSE-2.0/
        :alt: License

.. image:: https://img.shields.io/github/contributors/bobxwu/topmost
        :target: https://github.com/bobxwu/topmost/graphs/contributors/
        :alt: Contributors

.. image:: https://img.shields.io/badge/arXiv-2309.06908-<COLOR>.svg
        :target: https://arxiv.org/pdf/2309.06908.pdf
        :alt: arXiv


TopMost provides complete lifecycles of topic modeling, including datasets, preprocessing, models, training, and evaluations. It covers the most popular topic modeling scenarios, like basic, dynamic, hierarchical, and cross-lingual topic modeling.


| Check our **ACL 2024 demo paper**: `Towards the TopMost: A Topic Modeling System Toolkit <https://arxiv.org/pdf/2309.06908.pdf>`_.
| Check our survey paper on neural topic models accepted to **Artificial Intelligence Review**: `A Survey on Neural Topic Models: Methods, Applications, and Challenges <https://arxiv.org/pdf/2401.15351.pdf>`_.


|
| If you want to use TopMost, please cite as

::

    @article{wu2023topmost,
        title={Towards the TopMost: A Topic Modeling System Toolkit},
        author={Wu, Xiaobao and Pan, Fengjun and Luu, Anh Tuan},
        journal={arXiv preprint arXiv:2309.06908},
        year={2023}
    }

    @article{wu2023survey,
        title={A Survey on Neural Topic Models: Methods, Applications, and Challenges},
        author={Wu, Xiaobao and Nguyen, Thong and Luu, Anh Tuan},
        journal={Artificial Intelligence Review},
        url={https://doi.org/10.1007/s10462-023-10661-7},
        year={2024},
        publisher={Springer}
    }



==================

.. contents:: **Table of Contents**
   :depth: 2



============
Overview
============

TopMost offers the following topic modeling scenarios with models, evaluation metrics, and datasets:

.. image:: https://github.com/BobXWu/TopMost/raw/main/docs/source/_static/architecture.svg
    :width: 390
    :align: center

+------------------------------+---------------+--------------------------------------------+-----------------+
|            Scenario          |     Model     |               Evaluation Metric            |  Datasets       |
+==============================+===============+============================================+=================+
|                              | | LDA_        |                                            |                 |
|                              | | NMF_        |                                            | | 20NG          |
|                              | | ProdLDA_    | | TC                                       | | IMDB          |
|                              | | DecTM_      | | TD                                       | | NeurIPS       |
| | Basic Topic Modeling       | | ETM_        | | Clustering                               | | ACL           |
|                              | | NSTM_       | | Classification                           | | NYT           |
|                              | | TSCTM_      |                                            | | Wikitext-103  |
|                              | | BERTopic_   |                                            |                 |
|                              | | ECRTM_      |                                            |                 |
|                              | | FASTopic_   |                                            |                 |
+------------------------------+---------------+--------------------------------------------+-----------------+
|                              |               |                                            | | 20NG          |
|                              | | HDP_        | | TC over levels                           | | IMDB          |
| | Hierarchical               | | SawETM_     | | TD over levels                           | | NeurIPS       |
| | Topic Modeling             | | HyperMiner_ | | Clustering over levels                   | | ACL           |
|                              | | ProGBN_     | | Classification over levels               | | NYT           |
|                              | | TraCo_      |                                            | | Wikitext-103  |
|                              |               |                                            |                 |
+------------------------------+---------------+--------------------------------------------+-----------------+
|                              |               | | TC over time slices                      |                 |
| | Dynamic                    | | DTM_        | | TD over time slices                      | | NeurIPS       |
| | Topic Modeling             | | DETM_       | | Clustering                               | | ACL           |
|                              | | CFDTM_      | | Classification                           | | NYT           |
+------------------------------+---------------+--------------------------------------------+-----------------+
|                              |               | | TC (CNPMI)                               | | ECNews        |
| | Cross-lingual              | | NMTM_       | | TD over languages                        | | Amazon        |
| | Topic Modeling             | | InfoCTM_    | | Classification (Intra and Cross-lingual) | | Review Rakuten|
|                              |               | |                                          | |               |
+------------------------------+---------------+--------------------------------------------+-----------------+

.. _LDA: https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf
.. _NMF: https://papers.nips.cc/paper_files/paper/2000/hash/f9d1152547c0bde01830b7e8bd60024c-Abstract.html
.. _ProdLDA: https://arxiv.org/pdf/1703.01488.pdf
.. _DecTM: https://aclanthology.org/2021.findings-acl.15.pdf
.. _ETM: https://aclanthology.org/2020.tacl-1.29.pdf
.. _NSTM: https://arxiv.org/abs/2008.13537
.. _BERTopic: https://arxiv.org/pdf/2203.05794.pdf
.. _CTM: https://aclanthology.org/2021.eacl-main.143/
.. _TSCTM: https://aclanthology.org/2022.emnlp-main.176/
.. _ECRTM: https://arxiv.org/pdf/2306.04217.pdf
.. _FASTopic: https://arxiv.org/pdf/2405.17978

.. _HDP: https://people.eecs.berkeley.edu/~jordan/papers/hdp.pdf
.. _SawETM: http://proceedings.mlr.press/v139/duan21b/duan21b.pdf
.. _HyperMiner: https://arxiv.org/pdf/2210.10625.pdf
.. _ProGBN: https://proceedings.mlr.press/v202/duan23c/duan23c.pdf
.. _TraCo: https://arxiv.org/pdf/2401.14113.pdf

.. _DTM: https://mimno.infosci.cornell.edu/info6150/readings/dynamic_topic_models.pdf
.. _DETM: https://arxiv.org/abs/1907.05545
.. _CFDTM: https://arxiv.org/pdf/2405.17957

.. _NMTM: https://bobxwu.github.io/files/pub/NLPCC2020_Neural_Multilingual_Topic_Model.pdf
.. _InfoCTM: https://arxiv.org/abs/2304.03544




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



============
Installation
============


Stable release
--------------

To install TopMost, run this command in the terminal:

.. code-block:: console

    $ pip install topmost

This is the preferred method to install TopMost, as it will always install the most recent stable release.

From sources
------------

The sources for TopMost can be downloaded from the Github repository.

.. code-block:: console

    $ pip install git+https://github.com/bobxwu/TopMost.git





============
Tutorials
============

.. |github0| image:: https://img.shields.io/badge/Open%20in%20Github-%20?logo=github&color=grey
    :target: https://github.com/BobXWu/TopMost/blob/master/tutorials/tutorial_quickstart.ipynb
    :alt: Open In GitHub

.. |github1| image:: https://img.shields.io/badge/Open%20in%20Github-%20?logo=github&color=grey
    :target: https://github.com/BobXWu/TopMost/blob/master/tutorials/tutorial_preprocessing_datasets.ipynb
    :alt: Open In GitHub

.. |github2| image:: https://img.shields.io/badge/Open%20in%20Github-%20?logo=github&color=grey
    :target: https://github.com/BobXWu/TopMost/blob/master/tutorials/tutorial_basic_topic_models.ipynb
    :alt: Open In GitHub

.. |github3| image:: https://img.shields.io/badge/Open%20in%20Github-%20?logo=github&color=grey
    :target: https://github.com/BobXWu/TopMost/blob/master/tutorials/tutorial_hierarchical_topic_models.ipynb
    :alt: Open In GitHub

.. |github4| image:: https://img.shields.io/badge/Open%20in%20Github-%20?logo=github&color=grey
    :target: https://github.com/BobXWu/TopMost/blob/master/tutorials/tutorial_dynamic_topic_models.ipynb
    :alt: Open In GitHub

.. |github5| image:: https://img.shields.io/badge/Open%20in%20Github-%20?logo=github&color=grey
    :target: https://github.com/BobXWu/TopMost/blob/master/tutorials/tutorial_crosslingual_topic_models.ipynb
    :alt: Open In GitHub



We provide tutorials for different usages:

+--------------------------------------------------------------------------------+-------------------+
| Name                                                                           | Link              |
+================================================================================+===================+
| Quickstart                                                                     | |github0|         |
+--------------------------------------------------------------------------------+-------------------+
| How to preprocess datasets                                                     | |github1|         |
+--------------------------------------------------------------------------------+-------------------+
| How to train and evaluate a basic topic model                                  | |github2|         |
+--------------------------------------------------------------------------------+-------------------+
| How to train and evaluate a hierarchical topic model                           | |github3|         |
+--------------------------------------------------------------------------------+-------------------+
| How to train and evaluate a dynamic topic model                                | |github4|         |
+--------------------------------------------------------------------------------+-------------------+
| How to train and evaluate a cross-lingual topic model                          | |github5|         |
+--------------------------------------------------------------------------------+-------------------+


============
Disclaimer
============

This library includes some datasets for demonstration. If you are a dataset owner who wants to exclude your dataset from this library, please contact `Xiaobao Wu <xiaobao002@e.ntu.edu.sg>`_.



============
Authors
============

+----------------------------------------------------------+
| |xiaobao-figure|                                         |
| `Xiaobao Wu <https://bobxwu.github.io>`__                |
+----------------------------------------------------------+
| |fengjun-figure|                                         |
| `Fengjun Pan <https://github.com/panFJCharlotte98>`__    |
+----------------------------------------------------------+

.. |xiaobao-figure| image:: https://bobxwu.github.io/assets/img/figure-1400.webp 
   :target: https://bobxwu.github.io
   :width: 50

.. |fengjun-figure| image:: https://avatars.githubusercontent.com/u/126648078?v=4
    :target: https://github.com/panFJCharlotte98
    :width: 50


==============
Contributors
==============


.. image:: https://contrib.rocks/image?repo=bobxwu/topmost
        :alt: Contributors



=================
Acknowledgments
=================

- Icon by `Flat-icons-com <https://www.freepik.com/icon/top_671169>`_.
- If you want to add any models to this package, we welcome your pull requests.
- If you encounter any problem, please either directly contact `Xiaobao Wu <xiaobao002@e.ntu.edu.sg>`_ or leave an issue in the GitHub repo.
