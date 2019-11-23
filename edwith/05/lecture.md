# Neural Machine Translation, 강의 내용

## Index

1. Overview: a bit of history remark
2. Encoder & Decoder
3. RNN Neural Machine Translation

## Overview: a bit of history remark

### Machine Translation

- Input: a sentence written in a source language $L_S$

- Output: a corresponding sentence in a target language $L_T$

- Problem statement:

  - Supervised learnings: given the input sentence, output its translation

  - Compute the conditional distribution over all possible translation given the input 

    $p(Y = (y_1, \cdots, y_T)| X = (x_1, \cdots, x_{T'}))$

## Encoder & Decoder

### Token Representation - One-hot Vectors

1. Build source and target vocabularies of unique tokens
   - For each of source and target languages,
     1. Tokenize: separate punctuations, normalize punctuations, ...
     2. Subword segmentation: segment each token into a sequece of subwords
     3. Collect all unique subwords, sort them by their frequencies (descending) and assign indices.
2. Transform each subword token into a corresponding one-hot vector.

### Encoder - Source Sentence Representation

- Encoder the source sentence into a set of sentence representation vectors.

  - \# of encoded vectors is proportional to the source sentence length: often same.

    $H = (h_1, \cdots, h_{T'})$

  - Recurrent networks have been widely used, but CNN and self-attention are used increasingly more often.

- We do not want to collapse them into a single vector.

  - Collapsing often corresponds to information loss.
  - Increasingly more difficult to encode the entire source sentence into a single vector, as the sentence lenght increases
  - When collapsed, the system fails to translate a long sentence correctly.
  - The system translates reasonable up to a certain point, but starts drifting away.
  - We didn't know initially until [Bahdanau et al., 2015]

### Decoder - Language Modelling

- Autoregressive Language modelling with an infinite context

  - Larger context is necessary to generate a coherent sentence.
    - Semantics could be largely provided by the source sentence, but syntactic properties need to be handled by the language model directly.
  - Recurrent networks, self-attention and (dilated) convolutional networks
    - Causal structure must be followed

- Conditional Language modelling

  - The context based on which the next token is predicted is two-fold

    $p(Y|X) = \prod^T_{t=1}p(y_t | y_{<t}, X)$

  ![machine_translation](./images/machine_translation.png)

## RNN Neural Machine Translation

