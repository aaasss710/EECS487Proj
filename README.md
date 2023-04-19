# MoRA
MoRA: Momentum Contrastive Learning with RoBERTa

Course Project for EECS 487 NLP, Winter 2022

Team Members: Jiajun Xi, Yifu Lu, Zhenjie Sun

# Abstract
Embeddings from pretrained model provides easy access to large model training with lower resources. In this project, we propose MoRA (Momentum contrastive learning with Roberta), which could effectively provide semantic-rich sentence embeddings with the contrastive learning framework. We approach this problem by two phases: training the adaptor layers on the Wikipedia corpus using Uniformity and Alignment losses with momentum update for positive pairs generation and better stability; to evaluate our model, we fine-tune the model based on a downstream task, which is media bias detection with AllSides dataset. We set SimCSE, a contrastive learning paper accepted at EMNLP2021, as our baseline model and the MoRA model outperforms most of the STS tasks and has a better precision on media bias detection than SimCSE. 

# Data
- Wiki Dataset: run data/download_wiki.sh
- AllSides Dataset: download the dataset from https://github.com/ramybaly/Article-Bias-Prediction

# Evaluation
## STS score
First download the STS dataset from SentEval, then run eval.py.

## Downstream tasks
Run eval_downstream.py

# Training
## Unsupervised training on wiki dataset
Run RoBERTa.ipynb

## Finetuning with AllSides dataset
Run eval_downstream.py