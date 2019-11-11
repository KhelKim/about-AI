# Text Classification & Sentence Representation

## INDEX

1. Overview
2. How to represent sentence & token?
3. CBoW & RN & CNN
4. Self Attention & RNN
5. Summary

## Overview

### Text Classification

- input: a natural language sentence/paragraph
- Output: a category to which the input text belongs
  - There are a fixed number C of categories
- Examples
  - Sentiment analysis: is this review positive or negative?
  - Text categorization: which category does this blog post belong to?
  - Intent classification: is this a question about a Chinese restaurant?

## How to represent sentence & token?

- A sentence is a variable-length sequence of tokens: $X = (x_1, x_2, \cdots, x_T)$
- Each token could be any one from a vocabulary: $x_t \in V$
- Once the vocabulary is fixed and encoding is done, a sentence or text is just a sequence of "integer indices"

### How to represent a token

각 단어들의 상관관계가 aribitrary하다는 것을 반영한 one-hot encoding

- A token is an integer "index"
- How do should we represent a token so that it reflects its "meaning"?
- First, we assume nothing is known: use an one-hot encoding.
  - $x = [0, 0, \cdots, 0, 1, 0, \cdots, 0] \in {0, 1}^{|V|}$
  - |V|: the size of vocabulary
  - Only one of the elements is 1: $\sum^{|V|}_{i=1} x_i = 1$
  - Every token is equally distant away from all the others.
    - $||x - y|| = c \geq 0, if x \neq y$
- Second, the neural network capture the token's meaning as a vector.
  - This is done by a simple matrix multiplication:
    - $Wx = W[\hat{x}]$, if $x$ is one-hot, where $\hat{x} = argmax_j x_j$ is the token's index is the vocabulary. 
  - It is called by 'table-lookup layer'
- After the table-lookup operation, the input sentence is a sequence of continuous, high-dimensional vectors:
  - $X = (e_1, e_2, \cdots, e_T), \text{ where } e_t \in \mathbb{R}^d$
  - The sentence length $T$ differs from one sentence to another.
- The classifier needs to eventually compress it into a single vector.
  - classification: 문장에 대한 의미있는 벡터를 찾을 수 있는지를 고려하는 것

## CBoW & RN & CNN

문장에 대한 의미를 representation하는 방법에 대표적인 방법이 있다기보다는 풀고 싶은 문제에 따라 특별한 representation을 한다.

### CBoW

아주 간단하고 계산비용도 싸며 어느정도 정확도를 가지고 있다. 하지만 order를 전혀 고려하지 않는다.

- Continuous bag-of-words
  - Ignore the order of the tokens: $(x_1, x_2, \cdots, x_T) \rightarrow \{x_1, x_2, \cdots, x_T\}$
  - Simply average the token vectors:
    - Averaging is a differentiable operator. $\frac{1}{T}\sum^T_{t=1} e_t$
  - Generalizable to bag-of-n-grams
    - N-gram: a phrase of N tokens
  - Extremely effective in text classification
    - For instance, if there are many positive words, the review is likely positive.

### RN

계산 비용은 CBoW보다 비싸지만, 그래도 쌍을 이루는 단어에 대한 뜻은 잘 파악할 수 있다.

- Relation Network: Skip Bigrams
  - Consider all possible pairs of tokens: $(x_i, x_j) \forall i \neq j$
  - Combine two token vectors with a neural network for each pair
    - $f(x_i, x_j) = W \phi(U_{left}e_i + U_{right}e_j)$
    - $\phi$ is a element-wise nonlinear function, such as tanh or ReLU
  - Consider the "relation"ship between each pair of words
  - Averages all these relationship vectors
    - $RN(X) = \frac{1}{2N(N-1)}\sum^{T-1}_{i=1}\sum^{T}_{j=i+1}f(x_i, x_j)$
  - Could be generalizaed to triplets and so on at the expense of computational efficient.

### CNN

- Convolutional Networks
  - Captures k-grams hierarchically
  - One 1-D convolutional layer: considers all k-grams
    - $h_t = \phi(\sum^{k/2}_{\gamma = -k/2}W_{\gamma}e_{t+\gamma}), \text{ resulting in } H = (h_1, h_2, \cdots, h_T).$
  - Stack more than one convolutional layers: progressively-growing window
  - Fits our instuition of how sentence is understood: tokens -> mulit-word expressions -> phrases -> sentence
  - In practice, just another operation in a DAG:
    - Extremely efficient implementations are available in all of the major of frameworks.
  - Reccent advances
    - Multi-width convolutional layers [Kim, 2014; Lee et al., 2017]
    - Dilated convolutional layers [Kalchbrenner et al., 2016]
    - Gated convolutional layers [Gehring et al, 2017]

## Self Attention & RNN

### How to represent a sentence - Self-Attention

CNN의 단점: 아주 긴 sequence의 첫번째와 마지막이 dependency가 있다면 CNN이 캡쳐하기 어렵다.

RN의 단점: 모든 단어의 쌍을 보기 때문에 효율적이지 못하다.

RN는 단어의 순서쌍의 가중치가 모두 1인데, neural network가 가중치를 학습할 수 있을까?

- Can we compute those weights instead of fixing them to 0 or 1?
- That is, compute the weight of each pair $(x_t, {x_t'})$
  - $h_t = \sum^{T}_{t'=1}\alpha(x_t, x_{t'}) f(x_t, x_{t'})$
- The weighting function could be yet another neural network
  - Just another subgraph in a DAG: easy to use
    - $\alpha(x_t, x_{t'}) = \sigma(RN(x_t, x_{t'})) \in [0, 1]$
  - Perhaps, we want to normalize them so that the weights sum to one
    - $\alpha(x_t, x_{t'}) = \frac{exp(\beta(x_t, x_{t'}))}{\sum^T_{t''=1}exp(\beta(x_t, x_{t''}))}, \text{ where } \beta(x_t, x_{t'}) = RN(x_t, x_{t'})$
- Self-Attention: a generalization of CNN and RN.
- Able to capture long-range dependencies within a single layer.
- Able to ignore ireelevant long-range dependencies.
- Further generalization via multi-head and multi-head attention

- Weaknesses of self-attention
  1. Quadratic computational complexity of $O(T^2)$
  2. Some operations cannot be done easily: e.g. counting, ....

### How to represent a sentence - RNN

- Online compression of a sequence $O(T)$
  - $h_t = RNN(h_{t-1}, x_t), \text{ where } h_0 = 0.$
- 문장이 길어지면 하나의 vector로 compress하기 어려움
- Bidirectional RNN to account for both sides.
- Inherently sequential processing
  - Less desirable for modern, parallelized, distributed computing infrastructure
  - 문장의 한단어씩 봐야하기 분산 계산을 할 수 없다.
- LSTM and GRU have become de facto standard
  - All standard frameworks impolement them.
  - Efficient GPU kernels are available.

## Summary

### How to represent a sentence

- We have learned five ways to extract a sentence representation:
  - In all but CBoW, we end up with a set of vector representations. $H= {h_1, \cdots, h_T}$
  - These approaches could be "stacked" in an arbitrary way to improve performance.
  - These vectors are often averaged for classification

### We learned in this lecture...

- Token representation
  - How do we represent a discrete token in a neural network?
  - Training this neural network leads to so-called continuous word embedding.
- Sentence representation
  - How do we extract useful representation from a sentence?
  - We learned five different ways to do so: CBoW, RN, CNN, Self-Attention, RNN

