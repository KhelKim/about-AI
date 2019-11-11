# Text Classification & Sentence Representation

## INDEX

1. Overview
2. How to represent sentence & token?
3. CBoW & RN & CNN
4. Self Attention & RNN
5. Summary
6. Questions

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

## Summary

## Questions