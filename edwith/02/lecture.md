# Basic Machine Learning: Supervised Learning, 강의 내용

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

Three points to consider both in research and in practice

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

## Loss Function

A Neural Network computes a conditional distribution

- Supervised learning: what is $y$ given $x$?
  - $f_\theta(x) = ?$
- In other words, how probable is a certain value $y'$ of $y$ given $x$?
  - $p(y=y'|x) = ?$
- What kind of distributions?
  - Binary classification: Bernoulli distribution
  - Multiclass classification: Categorical distribution
  - Linear regression: Gaussian distribution
  - Multimodel linear regression: Mixture of Gaussians

1. output을 probability distribution으로 만들자
   1. 이 output을 어떤 방식을 통해 distribution으로 만들 수 있을까?
      - Bernoulli distribution(output: sigmoid), Categorical distribution(output: softmax), ...

- Loss Function - negative log-probability

  - Once a neural network outputs a conditional distribution $p_\theta(y|x)$, a natural way to define a loss function arises.

  - Make sure training data is maximally likely:

    - Equiv, to making sure each and every training example is maximally likely,

      $argmax_\theta\text{ } log\text{ } p_\theta(D) = argmax_\theta\sum^N_{n=1}log\text{ }p_\theta(y_n|x_n)$

  - Equivalently, we want to minimize the negative log-probability.

    - A loss function is the sum of negative log-probabilities of correct answers.

      $L(\theta) = \sum^N_{n=1}l(M_\theta(x_n), y_n) = - \sum^N_{n=1}log\text{ }p_\theta(y_n|x_n)$

## Optimization

output이 distribution output이 되게 한다면, loss function이 negative log-loss로 계산되게 할 수 있다.

- Loss Minimization
  - What we now know
    1. How to build a neural network with an arbitrary architecture.
    2. How to define a per-example loss as a negative log-probability.
    3. Define a single directed acyclic graph containing both.
  - What we now need to know
    1. Choose an optimization algorithm.
    2. How to use the optimization algorithm to estimate parameters $\theta$.
- parameter 공간에 무수히 많은 model들이 있는데, 모든 것을 다 시도해보고 최적인 것을 고르기는 어려움.
- 따라서, 아무 곳을 선택한 후에 loss를 낮추는 방향으로 최적화를 진행
  - Local, Iterative Optimization: Random Guided Search
    - 장점: 어떤 비용함수를 사용해도 상관 없음.
    - 단점: 차원이 커질 수로 사용하기 어려움
  - Gradient-based Optimization:
    - 미분을 통해 최적화 할 방향을 정한다
    - 장점: local minimum에 빠질 수 있지만 그래도 loss를 확실하게 낮출 수 있음.
    - 단점: Random Guided search에 비해서 탐색영역이 작음. 학습률에 따라 최적의 값으로 갈 수도 있고 못갈 수도 있음

## Backpropagation

- How do we compute the gradient of the loss function?

1. Manual derivation

   - Relatively doable when the DAG is small and simple.
   - When the DAG is larger and complicated, too much hassle.

2. Automatic differentiation(autograd)

   - Use the chain rule of derivatives
   - The DAG is nothing but a composition of (mostly) differentiable functions.
   - Automatically apply the chain rule of derivatives.

   1. Implement the Jacobian-vector product of each OP node.
      - Can be implemented efficiently without explicitly computing the Jacobian.
      - The same implementation can be reused every time the OP node is called.
   2. Reverse-sweep the DAG starting from the loss function node.
      - Iteratively multiplies the Jacobian of each OP node until the leaf nodes of the parameters.
      - As expensive as forward computation with a constant overhead; O(N), where N: # of node

pytorch 혹은 tensorflow에서 제공하는 automatic differentiation 덕분에 사용자는 front-end를 만드는 것처럼 graph를 만들고 계산할 수 있음

## Gradient-Based Optimization

- train 데이터 전부를 보고 Backpropagation을 통해 loss funtion을 낮추는 gradient를 계산하기엔 연산상의 문제가 있음
- 따라서 stochastic gradient descent를 사용함
  - stochastic gradient descent: Approximate the full loss function (the sum of per-examples losses) using only a small random subset of training examples:
    - $\nabla L \approx \frac{1}{N'}\sum^{N'}_{n=1}\nabla l(M(x_{n}, y_n))$
  - Unbiased estimate of the full gradient.
  - Extremely efficient de facto standard practice.
-  Stochastic gradient descent in practice
  1. Grab a random subset of M training examples
     - $D' = \{(x_1, y_1), \cdots, (x_{N'}, y_{N'}\}$
  2. Compute the minibatch gradient
  3. Update the parameters
     - $\theta \larr \theta + \eta \nabla (\theta ; D')$
  4. Repeat until the validation loss stops improving.
     - validation의 loss가 더 떨어지지 않으면 early stop을 해야한다.
     - An efficient way to prevent overfitting
       - Overfitting: the training loss is low, but the validation loss is not
       - The most serious problem in statistical machine learning
       - one of the solutions: **Early-stop** based on the validation loss
- Adaptive learning rate: Adam, Adadelta, etc, ...
  - 최적의 learning rate를 찾기 위해서 여러 learning rate를 시험해보는 것도 중요하지만 초반 prototype을 만들 때는 adam, adadelta 등을 통해 다른 가능성을 지우는 것도 중요함.

## Summary

Supervised Learning with Neural Networks

1. How do we decide/design a **hypothesis set**?
   - Design a network architecture as a directed acyclic graph
2. How do we decide a **loss function**?
   - Frame the problem as a conditional distribution modelling
   - The per-example loss function is a negative log-probability of a correct answer
3. How do we **optimize** the loss function?
   - Automatic backpropagation: no manual gradient derivation
   - Stochastic gradient descent with early stopping [and adaptive learning rate]