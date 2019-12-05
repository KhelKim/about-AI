# Improving Image Captioning with Conditional Generative Adversarial Nets

## Index

1. Introduction
2. Preliminary
   1. Policy Gradient

## Introduction

## Preliminary

### Policy Gradient

본 논문은 Image Captioning에 decoder 파트에서 강화학습을 사용한다. 이를 이해하기 위해서 NLP에서 sentence generation을 할 때에 강화학습을 적용하는  방식을 알아보자.

강화학습은 환경 environment, 상태 state, 보상 reward, 전략 policy로 구성되어있고, 최종적인 reward를 최대화하기 위해 적절한 policy를 구하는 것이 목적이다.

강화학습(Reinforcement Learning)