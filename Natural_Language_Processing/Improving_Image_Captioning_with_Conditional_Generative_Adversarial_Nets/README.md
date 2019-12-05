# Improving Image Captioning with Conditional Generative Adversarial Nets

## Index

0. Preliminary
   1. Policy Gradient
1. Introduction
2. Image Captioning Via Reinforcement Learning
3. Proposed Conditional Generative Adversarial Training Method
   1. Overall Framework
   2. CNN-based Discriminator Model
   3. RNN-based Discriminator Model
   4. Algorithm

## Preliminary

본 논문은 Image Captioning에 decoder 파트에서 강화학습을 사용한다. 이를 이해하기 위해서 NLP에서 sentence generation을 할 때에 강화학습을 적용하는  방식을 알아보자.

강화학습은 환경 environment, 상태 state, 보상 reward, 전략 policy로 구성되어있고, 최종적인 reward를 최대화하기 위해 적절한 policy를 구하는 것이 목적이다. sentence generation을 하기 위해 RNN을 주로 사용하는데 RNN은 강화학습을 적용하기 좋은 특성을 가지고 있다.

- environment: RNN layer
- state: previous hidden state
- action: next token
- reward: BLUE, CIDEr, SPICE, etc.

이 논문에서 사용하는 목적함수는 다음과 같다.

- $L_G(\theta) = \mathbb{E}_{x^s \sim G_\theta}[r(x^s)]$
  - $G$: caption generator
  - $\theta$: parameters
  - $G_\theta$: caption generator with the parameters $\theta$
  - $x^s$: sentence from the distribution $G_\theta $
  - $r$: language evaluation metric score(CIDEr, BLUE, SPICE, etc.)

이 목적함수는 값이 커져야하며 목적함수의 값을 가장 크게 하는 최적 parameters를 찾기위해 policy gradient인 REINFORCE 알고리즘을 이용할 것이다. Gradient estimate의 variance를 줄이기 위해 baseline function $b$를 사용한다.

- $\bigtriangledown_\theta L_G(\theta) \approx 
\sum^{T_s}_{t=1}(r(x^s_{1:t}) - b) \bigtriangledown_\theta \operatorname{log} G_\theta(x^s_t|x^s_{1:t-1})
$

Rennie et al. 2017의 SCST(self-critical sequence training) method에 따라 baseline function을 reward $r(x^g)$ (obtained by the current model under the greedy decoding algorithm used at test time)를 사용한다. 즉, 

- $\bigtriangledown_\theta L_G(\theta) = 
\sum^{T_s}_{t=1}(r(x^s_{1:t}) - r(x^g)) \bigtriangledown_\theta \operatorname{log} G_\theta(x^s_t|x^s_{1:t-1})
$  

### Policy Gradient

위에서 구한 목적함수의 gradient를 이용해 REINFORCE 알고리즘이 어떻게 적용되는지 간단한 예를 통해 알아보자.

1. sentence generation 과정

   $t=1$일 때, 주어진 state $S_1$는 (\<sos\>)이고, 그에 따른 reward는 $r_1$이다. $S_1$의 $G_\theta$ 값인 discrete probability distribution에서 $x_1$을 추출한다.
   
   $t=2$일 때, 주어진 state $S_2$는 (\<sos>, $x_1$)이고, 그에 따른 reward는 $r_2$이다. $S_2$의 $G_\theta$ 값인 discrete probability distribution에서 $x_2$를 추출한다.
   
   만약 $x_2$가 \<eos\> token이라면 주어진 state $S_3$, (\<sos\>, $x_1$, $x_2$)를 평가한 reward $r_3$를 얻고 프로세스가 종료된다.
   
   | T    | S    | R    |                                                              |
   | ---- | ---- | ---- | ------------------------------------------------------------ |
   |  $t=1$  |  $S_1: \text{<sos>}$  | $R_1: r_1$ |  |
   |  | S => <br>$G_\theta(A_1=a|\text{<sos>})$ |      | <img src="./images/distribution1.png" alt="distribution1.png" style="zoom:50%;" /> |
   | $t=2$ | $$S_2: \text{<sos>}, x_1$$ | $R_2: r_2$ |  |
   | | T, S, R => <br>$G_\theta(A_1=a|\text{<sos>}, x_1)$ |  | <img src="./images/distribution2.png" alt="distribution2.png" style="zoom:50%;" /> |
   | $t=3$ | $S_3: \text{<sos>}, x_1, x_2$ | $R_3:r_3$ |  |
   
2. 한 에피소드가 끝난 후, gradient를 이용한 parameters 업데이트하기

   t기의 policy gradient 업데이트 factor는 다음과 같다.

   - $\alpha\{r^t G_t\bigtriangledown_\theta \operatorname{log}\pi_\theta(A_t|S_t)\}$

   본 논문에서는 discount factor가 1이며, return $G_t$를 $r(x^s_{1:t}) - r(x^g)$로 사용한다.

   - $t = 1$

     $\theta \leftarrow \theta + \alpha\{(r(x^s_{1:1}) - r(x^g))\bigtriangledown_\theta \operatorname{log}G_\theta(x_1|\text{<sos>})\}$

     Here, $r$ is the reward function, $\alpha$ is the learning rate, and there is no discount factor.

   - $t = 2$

     $\theta \leftarrow \theta + \alpha\{(r(x^s_{1:2}) - r(x^g))\bigtriangledown_\theta \operatorname{log}G_\theta(x_2|\text{<sos>}, x_1)\}$

   $t=1$과 $t=2$에 $\theta$가 업데이트 된 것을 합치면

   $\theta \leftarrow \theta + \alpha\sum^{2}_{t=1}(r(x^s_{1:t}) - r(x^g)) \bigtriangledown_\theta \operatorname{log} G_\theta(x^s_t|x^s_{0:t-1})$

   즉,

   $\theta \leftarrow \theta + \alpha \bigtriangledown_\theta L_G(\theta)$

   이렇게 업데이트 된다.

## Introduction



## Image Captioning Via Reinforcement Learning



## Proposed Conditional Generative Adversarial Training Method



### Overall Framework



### CNN-based Discriminator Model



### RNN-based Discriminator Model



### Algorithm