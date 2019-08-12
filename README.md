# Automating Online Hate Speech Detection: A Survey of Deep Learning Approaches

## Motivation 
Hateful content on social media 
* Contributes to real-world violence
* Recruitment to and propaganda for terrorist individuals/groups
* Makes other users feel less safe and secure on social platforms
* Triggers increased levels of toxicity in the network

## Contributions 
* Survey of deep learning model architectures, embedding choices, and feature inputs
* Experiment with user behavior metrics in a multiple input model architectures
* We find that Google’s pretrained Bert embeddings provide enough semantic meaning. User behavior metrics do not improve upon Bert

## Research Question:
How can we improve the performance of automated systems on identifying hate speech when they must learn from very few hateful samples?

## Methodology
Our dataset: 
* 64,149 tweets total
* 4% hateful
* 20% abusive
* 62% normal
* 14% spam

### Dataset Analytics 
Source: [80k annotated tweets](http://www.aclweb.org/anthology/N16-2013)

### Embedding choices
* TF-IDF
* Pretrained Twitter
* Pretrained Bert

### Architectures 
* Logistic Regression baseline 
* Multilayer Perceptron 
* CNN
* LSTM
* DenseNet

### Experiment Design	
* <strong> Phase 1:</strong> Tweet Embeddings
* <strong>Phase 2:</strong> Tweet embeddings + Reply-pairing embeddings + reply network metrics as embedding coefficients (favorite count & retweet count)
* <strong>Phase 3:</strong> Tweet embeddings + Dominant LDA Topic words from user timeline tweets 

### To run
* Use Tweepy API to collect data or email me for 64k tweets dataset + embeddings 
* Install dependencies 
* Configure cloud computing environment 
* Sample run: `python deep_learning_experiments.py --num_epochs 100 --model CNN --name test --seed 28 --embedding_key twitter --embedding_level word --experiment_flag 2`

Full paper: Coming Soon
Contact: ashe.magalhaes@gmail.com
