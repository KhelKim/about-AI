# Massively Multilingual Word Embeddings

## Index

0. Abstract
1. Introduction
2. Estimation Multilingual Embeddings
   1. Multicluster
   2. MultiCCA
   3. MultiSkip
   4. Translation-invariance
3. Evaluating Multilingual Embeddings
   1. Word similarity
   2. Word translation
   3. Correlation-based evaluation
   4. Extrinsic tasks

## Abstract

이 논문에서는 단 하나의 embedding space에 50개가 넘는 언어의 단어들을 embedding하고 평가하는 방법에 대해서 소개하고 있다. MultiCluster과 multiCCA라고 부르는 새로운 embedding 방법은 단어 사전과 각 언어에 대한 데이터를 이용한다. 즉, 언어별 parallel data가 필요하지 않다. 또, multiQVEC-CCA라고 불리는 평가 방식은 여러 언어의 관계를 더 잘 파악할 수 있다.

## Introduction

단어들을 continuous space에 embedding하는 것은 NLP에서 자주 쓰이는 방법이다. 또, 여러 언어의 단어를 한 공간에 embedding하는 것은 추가적인 성증 향상을 할 수 있다.

1. 예를 들어, machine translation에서 한 언어의 단어에 대한 정보가 없을 때, 같은 공간에 제일 근접한 다른 언어의 단어를 출력할 수도 있다.
2. 또, transfer learning을 할 때 한 언어에 대해 학습된 모델이 있다면, 다른 언어에 그 모델을 적용하기 좋다(embedding space를 공유하기 때문에).

따라서 여러 언어의 단어를 한 공간에 embedding하는 것은 multilingual NLP를 할 때 여러 도움이 될 수 있고 이를 위해 본 논문은 다음과 같은 methods와 평가 방식을 제안한다.

1. MultiCluster & multiCCA: dictionary-based method로서 monolingual data와 pairwise parallel dictionary만 train할 때 필요하다. Parallel corpora는 사용가능하다면 사용할 수 있지만 필수적이지 않다.
2. MultiQVEC-CCA: 본 논문에서는 multilingual embeddings를 평가하기 위해 QVEC 평가 방식(multiQVEC)을 사용했고, 추가로 multiQVEC-CCA를 제안한다. 

## Estimation Multilingual Embeddings



### Multicluster



### MultiCCA



### MultiSkip



### Translation-invariance



## Evaluating Multilingual Embeddings



### Word similarity



### Word translation



### Correlation-based evaluation



### Extrinsic tasks

