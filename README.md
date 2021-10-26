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
- 제곱 tf.square(행렬), 제곱근 tf.sqrt(),로그값 tf.math.log() 
- tf.assign() 텐서 재사용, tf.assign_add() 텐서 더하기(메모리 재사용)
- np.array(텐서형태 행렬) tensor를 ndarray형태로 변환
- tensor.numpy() tensor를 ndarray형태로 변환
- decorate, @tf.function 좀더 효율적으로 바꿔줌
- print(tf.config.list_physical_devices("GPU")) GPU확인
- !nvidia-smi 그래픽 카드 확인


# tensorflow_002
- 산점도 만들기 (그래프 그리기)
- plt.scatter(행렬,행렬) 
- BA와 BI의 차이
- regression 준비
- 텐서플로우로 머신러닝이나 딥러닝 하는 방법
1. 데이터 준비
2. 모델 생성(create)
3. 모델 컴파일(모델성능 측정방법 정하기, 옵티마이저 설정)
4. 모델 학습(fit)
- batch size와 epoch 공부
-  모델 개선하기 위한 방법
- epoch 100으로 실행 (예측성능 좋아짐)
- 데이터 나누기
- Train , validation , Test set을 나누어 준다.
- 데이터 나눈거 시각화 하기
- model.summary 통해서 요약보기 ### feat input_shape =[] ###
- model.fit(verbose = 0) 하면 진행상황 안보여줌
- model.summary 를 도식화 해줄 수 있다.(안되면 코랩에서 해봐)