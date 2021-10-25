tensorflow_nlp
# tensorflow_001
- tf.constant(데이터) 텐서를 만들어 준다.
- .ndim 차원을 알려준다
- tf.Variable 는 변경가능 tf.constant는 변경 불가
- 텐서플로우 에서 seed는 이미 값이 정해져 있는 숫자임
- 시드를 이용해서 shuffle 해줄 수 있다. tf.random.shuffle(텐서,시드)
- numpy로 벡터를 생성하고 텐서에 넘겨줄 수 있다.tf.constant(넘파이,형태,데이터 타입)
- 텐서 연산(더하기 곱하기 빼기)
- @ 연산은 행렬곱이다.
- tf.reshape() : 우리가 정의한 shape로 행렬을 바꾸어 주는 역할
- tf.transpose() : 텐서를 이루는 dimension, 즉 axis를 서로 바꾸어 준다!
- 최소 최대 평균 합 표준편차 분산 
- .argmax(행렬).numpy()  최대값 위치
- .reduce_max(행렬).numpy() 최대값
- 축 줄이기 tf.squeeze(행렬)
- tf.one_hot(리스트, 깊이) 원핫 인코딩