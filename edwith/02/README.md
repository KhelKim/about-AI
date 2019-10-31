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

## Hypothesis Set

이번 강의에서는 딥러닝에 한정.

- neural network: An directed acyclic graph
- Forward computation(구조를 사용하는 것): how you "use" a trained neural network.
- Implication in practice
  - Naturally supports high-level abstraction
  - Object-oriented paradigm fits well.
    - Base classes: variable (input/output) node, operation node, ...
    - Define the internal various types of variables and operations by inheritance
  - Maximal code reusability
- You define a hypothesis set by designing a directed acyclic graph.
- The hypothesis space is then a set of all possible parameter settings.

## Loss Function - Preview

training = optimization = loss function의 값이 최대한 작아지게

loss function을 정의해야 할 순간들이 많음.

- So many loss functions
  - Classification: hinge loss, log-loss, ...
  - Regression: mean squared error, mean absolute error, robust loss, ...
- In this lecture, we stick to distribution-based loss functions.
  - distribution이 주어졌을 때, loss function을 정의할 수 있게 됨

## Probability in 5 minimutes

- An "event set" $\Omega$ contains all possible events: $\Omega = \{e_1, e_2, \cdots, e_D\}$
  - Discrete: when there are a finite number of events $|\Omega| < \infin$
  - Continuous: when there are infinitely many events $|\Omega|=\infin$
- A "random variable" $X$ could take any one of these events: $X \in \Omega$
- A probability of an event: $p(X = e_i)$
  - How likely would the $i$-th event happen?
  - How often has the $i$-th event occur relative to the other events?
- Properties
  1. Non-negative: $p(X = e_i) \geq 0$
  2. Unit volume: $\sum_{e \in \Omega}p(X = e) = 1$
- Multiple random variables: consider two here - $X$, $Y$
- A joint probability $p(Y = e^Y_j, X = e^X_i)$
  - How likely would $e^Y_j$ and $e^X_i$ happen together?
- A conditional probability $p(Y = e^Y_j| X = e^X_i)$
  - Given $e^X_i$, how likely would $e^Y_j$ happen?
  - The chance of both happening together divided by that of $e^X_i$ happening regardless of whether $e^Y_j$ happend:
    - $p(Y|X) = \frac{p(X, Y)}{p(X)} \iff p(X, Y) = p(Y|X)p(X)$
- Probability function $p(X)$ returns a probability of $X$(marginal probability)
- A marginal probability $p(Y=e^Y_j)$
  - Regardless of what happen to $X$, how likely is $e^Y_j$?
    - $p(Y=e^Y_j) = \sum_{e \in \Omega_x}p(Y=e^Y_j, X = e)$
- malginalization
  - 동전 예시, 첫번째 동전의 결과값: $X$, 두번째 동전의 결과값: $Y$
  - 알고 있는 것 $p(X, Y)$, 구하고 싶은 것 $p(Y)$