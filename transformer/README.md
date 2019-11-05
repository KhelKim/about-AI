# Transformer

1. residual connection
   1. residual connection이 보는 깊은 딥러닝의 문제
      1. 신경망이 깊어질 때, 필요 이상으로 layer들이 더 추가된다면 이 layer들은 identity mapping이면 충분하다.
      2. 하지만 deep learning은 identity mapping을 만들지 않는다.
   2. 해결방안
      1. 얼마나 변하는지를 관찰하자. $F(x) = H(x) - x$
      2. 만약 identity mapping이 이상적인 mapping이라면 구해야하는 것은 zero mapping이다.
      3. 그러면 solver은 학습하기 더 쉬워질 것이다(zero mapping이 학습하기 쉬운가(?)).
      4. 실험적으로 learned residual functions는 small response를 가지고 있다(response가 matrix의 크기인듯?). it supports our basic motivation that the residual funcitons might be generally closer to zero than non-residual functions.(이 말도 zero mapping이 학습하기 쉽다는 듯이 읽힘)
   3. 논문에 나오진 않지만 추가적인 benefit
      1. sovling vanishing and exploding gradient problem
2. layer normalization
   1. batch normalization vs layer normalization
3. positional embedding(in paper)
4. Scaled Dot-Product Attention(in paper)
5. Multi head self attention(in paper)
   1. self attention?
   2. Multi head attention?





- residual connection
  - [Deep residual learning for image recognition]( https://arxiv.org/pdf/1512.03385.pdf )
  - [라온피플
  - paper: Residual Recurrent Neural Networks for Learning Sequential Representations
- layer normalization
  - [layer normalization](https://arxiv.org/pdf/1607.06450.pdf)