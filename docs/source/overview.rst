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

