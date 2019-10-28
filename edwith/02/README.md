# Basic Machine Learning: Supervised Learning

## Index

1. Overview
2. Hypothesis Set
3. Loss Function - Preview
4. Probability in 5 minutes
5. Loss Function
6. Optimization Methods
7. Backpropagation
8. Gradient-Based Optimization
9. Summary
10. Questions

## Overview

- Algorithm이란?
  - 명령어의 집합이며 일련의 명령어들을 하나하나 실행했을 때, 우리의 문제를 해결하는 것
- traditionally
  - input이 주어지면
  - 프로그램을 작성하여
  - output을 구한다
- Machine learning
  - 데이터가 주어지면
  - machine learning algorithm을 훈련시켜
  - input을 넣으면 적절한 output이 출력되는 프로그램을 구한다.

### Supervised Learning - Overview

- Provided:

  - a set of N input-output 'training' examples
  - A per-example loss function
  - Evaluation sets: validation and test examples

- What we  must decide:

  - Hypothesis sets
    - 문제에 적합한 방법을 말한다.
  - Optimization algorithm
    - 어떻게 학습할지를 결정

- Supervised learning finds an appropriate algorithm/model automatically

  1. Training

  - $\hat{M}_m = arg min_{M \in H_m}\sum^N_{n=1}l(M(x_n), y_n)$
    - 한 방법에 대해 적합한(파라미터가 잘 조절된) 모델을 찾는다.

  2. Model Selection

  - $\hat{M} = argmin_{M \in \{\hat{M}_1, \hat{M}_2, \cdots, \hat{M}_M\}}\sum_{(x, y) \in D_{val}} l(M(x), y)$
    - 좋은 모델을 찾는다.

  3. Reporting

  - Report how well the best model would work using the test set loss.

  - $R(\hat{M})\sim \frac{1}{|D_{test}|}\sum_{(x, y) \in D_{test}}l(\hat{M}(x), y)$

    

  - 중요한 부분은 1, 2, 3을 진행하는 데이터들이 구분되어야 한다는 것이다.

- It result in an algorithm $\hat{M}$ with an expected performance of $R(\hat{M})$

Three points to consider both in research an din practice

1. How do we decide/design a hypothesis set?
   1. 문제에 적합한 방법들은 어떤 것들이 있고 무엇을 사용할 지
2. How do we decide a loss function?
   1. 알려진 loss function 외에 문제에 적합한 loss function은 어떻게 정의할지?
3. How do we optimize the loss function?
   1. 그러한 loss function을 어떻게 optimization할지?