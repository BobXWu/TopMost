# TopMost: A Topic Modeling System Toolkit

TopMost provides a complete lifecycle of topic models, including dataset preprocessing, model training and evaluation.
It covers the most popular topic modeling scenarios, like static, dynamic, hierarchical, and cross-lingual topic modeling.

This is our demo paper of TopMost: Towards the TopMost: A Topic Modeling System Toolkit.  
This is our survey on neural topic models: [A Survey on Neural Topic Models: Methods, Applications, and Challenges](https://www.researchsquare.com/article/rs-3049182/latest.pdf).


- [TopMost: A Topic Modeling System Toolkit](#topmost-a-topic-modeling-system-toolkit)
  - [Introduction](#introduction)
    - [Topic Models](#topic-models)
  - [Quick Start](#quick-start)
    - [1. Install](#1-install)
    - [2. Download a dataset](#2-download-a-dataset)
    - [3. Train a model](#3-train-a-model)
    - [4. Evaluate](#4-evaluate)
    - [Preprocessing Datasets (Optional)](#preprocessing-datasets-optional)
    - [Test New Documents (Optional)](#test-new-documents-optional)
  - [Tutorials](#tutorials)
  - [Notice](#notice)
    - [Differences from original implementations](#differences-from-original-implementations)
  - [Contributors](#contributors)
  - [Disclaimer](#disclaimer)
  - [Acknowledgements](#acknowledgements)


## Introduction

### Topic Models

We provide the following fundamental and recent models:

<table>
<tbody>
<tr>
<td>Scenario</td>
<td>Model</td>
<td>Evaluation Metric</td>
<td>Datasets</td>
</tr>
<tr>
<td>Static Topic Models</td>
<td>
    <a href="">LDA</a><br/>
    <a href="">NMF</a><br/>
    <a href="https://arxiv.org/pdf/1703.01488">ProdLDA</a><br/>
    <a href="https://aclanthology.org/2021.findings-acl.15.pdf">DecTM</a><br/>
    <a href="https://aclanthology.org/2020.tacl-1.29.pdf">ETM</a><br/>
    <a href="https://arxiv.org/abs/2008.13537">NSTM</a><br/>
    <a href="">CTM</a><br/>
    <a href="https://aclanthology.org/2022.emnlp-main.176">TSCTM</a><br/>
    <a href="https://arxiv.org/pdf/2306.04217">ECRTM</a>
</td>
<td>
    TC<br/>
    TD<br/>
    Clustering<br/>
    Classification
</td>
<td>
    20NG<br/>
    IMDB<br/>
    NeurIPS<br/>
    ACL<br/>
    NYT<br/>
    Wikitext-103<br/>
</td>
</tr>
<tr>
<td>Hierarchical Topic Models</td>
<td>
    <a href="https://arxiv.org/abs/2210.10625">HyperMiner</a><br/>
    <a href="https://proceedings.mlr.press/v202/duan23c/duan23c.pdf">ProGBN</a>
</td>
<td>
    TC over levels<br/>
    TD over levels<br/>
    Clustering over levels<br/>
    Classification over levels
</td>
<td>
    20NG<br/>
    IMDB<br/>
    NeurIPS<br/>
    ACL<br/>
    NYT<br/>
    Wikitext-103<br/>
</td>
</tr>
<tr>
<td>Dynamic Topic Models</td>
<td>
    <a href="https://arxiv.org/abs/2012.01524">DETM</a>
</td>
<td>
    TC over time slices<br/>
    TD over time slices<br/>
    Clustering<br/>Classification
</td>
<td>
    NeurIPS<br/>
    ACL<br/>
    NYT<br/>
</td>
</tr>
<tr>
<td>Cross-lingual Topic Models</td>
<td>
    <a href="https://bobxwu.github.io/files/pub/NLPCC2020_Neural_Multilingual_Topic_Model.pdf">NMTM</a><br/>
    <a href="https://arxiv.org/abs/2304.03544">InfoCTM</a>
</td>
<td>
    TC (CNPMI)<br/>
    TD over languages<br/>
    Classification (Intra-lingual and Cross-lingual)
</td>
<td>
    ECNews<br/>
    Amazon Review<br/>
    Rakuten Amazon<br/>
</td>
</tr>
</tbody>
</table>


## Quick Start

### 1. Install

Install topmost with `pip` as

```
pip install topmost
```


### 2. Download a dataset

```python
import topmost

dataset_dir = "./datasets/NYT"
topmost.data.download_dataset('NYT', cache_path=dataset_dir)

```

### 3. Train a model

```python
device = "cuda" # or "cpu"

# load a preprocessed dataset
dataset = topmost.data.BasicDatasetHandler(dataset_dir, device=device, read_labels=True, as_tensor=True)
# create a model
model = topmost.models.ETM(vocab_size=dataset.vocab_size, pretrained_WE=dataset.pretrained_WE)
model = model.to(device)

# create a trainer
trainer = topmost.trainers.BasicTrainer(model, dataset, epochs=2)

# train the model
trainer.train()
```

### 4. Evaluate

```python
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
```

### Preprocessing Datasets (Optional)

TopMost can preprocess datasets for topic modeling in a standard way.
Here are the steps:

1. Prepare datasets.

    A dataset must include two files: `train.jsonlist` and `test.jsonlist`. Each contains a list of json, like

```json
  {"label": "rec.autos", "text": "WHAT car is this!?..."}
  {"label": "comp.sys.mac.hardware", "text": "A fair number of brave souls who upgraded their..."}
```

2. Preprocess datasets.

    Here we download and preprocess 20newsgroup.

```python
import topmost
from topmost.preprocessing import Preprocessing
from topmost.preprocessing import download_20ng

dataset_dir = "./datasets/20NG"

download_20ng.download_save(output_dir=dataset_dir)
preprocessing = Preprocessing(vocab_size=5000)
preprocessing.parse_dataset(dataset_dir=dataset_dir, label_name="group")
```

### Test New Documents (Optional)

```python
# test new documents
import torch

new_texts = [
    "This is a new document about space, including words like space, satellite, launch, orbit.",
    "This is a new document about Microsoft Windows, including words like windows, files, dos."
]

parsed_new_texts, new_bow = preprocessing.parse(new_texts, vocab=dataset.vocab)
new_theta = runner.test(torch.as_tensor(new_bow, device=device).float())
```


## Tutorials

We provide tutorials in `./tutorials`.

## Notice

### Differences from original implementations

1. Oringal implementations may use different optmizer settings. For simplity and brevity, our package in default uses the same setting for different models.



## Contributors
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="120px">
          <a href="https://bobxwu.github.io/">
              <img src="https://bobxwu.github.io/img/figure.jpg" width="80px;" style="border-radius: 100%" alt="Xiaobao Wu"/>
              <br/>
              <sub><b>Xiaobao Wu</b></sub>
          </a>
      </td>
    </tr>
  </tbody>
</table>


<!-- ## Citation

If you are interested in our work and plan to use it, please cite as -->


## Disclaimer

This library includes some datasets for demostration.
If you are a dataset owner who wants to exclude your dataset from this library,
please contact [Xiaobao Wu](xiaobao002@e.ntu.edu.sg).

## Acknowledgements

- If you want to add any models to this package, we welcome your pull requests.
- If you encounter any problem, please either directly contact [Xiaobao Wu](xiaobao002@e.ntu.edu.sg) or leave an issue in the GitHub repo.
