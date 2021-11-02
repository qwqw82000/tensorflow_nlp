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
- model.summary 통해서 요약보기 ### input_shape =[] ###
- model.fit(verbose = 0) 하면 진행상황 안보여줌
- model.summary 를 도식화 해줄 수 있다.(안되면 코랩에서 해봐)
- 예측 값 시각화
- metrics MAE와 MSE
- evaluate(X_test, y_test) mae 확인
- .squeeze() shape 조정
- model_1 : 이전 모델과 같은데, 학습을 더 시킨다 성능이 더 안좋아짐
- model_2 : 레이어를 더 추가! 성능은 좋아짐
- model_3 : model_2과 같게 하고 epoch를 늘려본다 성능이 이상해짐
- model_results 한눈에 모델들 성능 보기
- 모델 저장 방법
- 예시) model_2.save("best_model_SavedModel_format")
- 예시) model_2.save("best_model_SavedModel_format.h5")
- 모델 불러오기
- 예시)tf.keras.models.load_model("best_model_SavedModel_format.h5")
- insurance csv 파일 불러오기
- 수치형 데이터로 바꿔주기
- 예시)  pd.get_dummies(insurance)
- 테스트와 라벨 데이터 만들기
- 학습시키기
- insurance_model_1 이전 모델로 학습(레이어 두개)
- insurance_model_2 이전 모델로 학습(레이어 세개)
- history 확인
- loss와 epoch 관계 보기 (시각화)
- 모델 개선이 필요하다



# tensorflow_003
- 텐서플로우 테스트용 예제 데이터 from sklearn.datasets import make_circles
- make circles 이용해 데이터셋 만들기(원 형태의 모델임)
- binary classification과 multiclass classification
- 모델을 조금식 개선해보기
- 모델 시각화
- model_1 레이어 하나짜리 딥러닝(BinaryCrossentropy,SGD,accuracy,epochs = 100)
- model_2 레이어 두개짜리 딥러닝
- model_3 레이어 세개짜리 딥러닝 + 아담
- plot decision bundary 함수 작성 (시각화)
- model_4 모델에 액티베이션과, 러닝 레이트 바꿔주기
- model_5 모델에 액티베이션을 relu를 주기

- model_6 레이어 두개에 relu주기
- model_7 마지막 레이어에 sigmoid를 주기(정확도 좋아짐)
- 러닝레이트에 따른 모델의 학습 정도
- model_8 러닝레이트를 0.01로 변경
- history 통해서 이상적 그래프 확인
- model_9 러닝 레이트 점점 커지게 만들기
- 러닝 레이트와 LOSS와의 관계
- model_10 최적의 러닝 레이트로 학습 시키기 0.02
- 주요 metrics 살펴보기(confusion matrix)
- confusion matrix 그려보기
- fashion mnist가지고 multiclass classification 하기
- model_11 fashion mnist의 모델 만들기 (soft max 사용하기)(성능 별로임) 
- normalization하기 0~255 -> 0~ 1
- model_12 fashion mnist 데이터 정규화 후 학습
- model_13 epoch 100번으로 늘림
- 러닝 레이트 점점 크게 해주는 모델 통해서 최적의 러닝 레이트 찾기(그래프)
- model_14 최적의 러닝 레이트로 학습(0.01)(0.03)(0.002) 하지만 성능 별로임

**# transfer learning_001**

- zip file은 데이터앞축 해제용 라이브러리

`os.walk()`는 하위의 폴더들을 for문으로 탐색할 수 있게 해줍니다. 인자로 전달된 path에 대해서 다음 3개의 값이 있는 tuple을 넘겨줍니다.

- root : dir과 files가 있는 path
- dirs : root 아래에 있는 폴더들
- files : root 아래에 있는 파일들
- 이미지 전처리 하기(ImageDataGenerator)
- https://3months.tistory.com/199 ImageDataGenerator 참고 블로그
- callback복습(특별한 기능 추가) epoch마다 학습이 어떻게 되는지 확인
- callback으로 텐서보드 만들기 (함수)
- tensorflow_hub 사용하기
- 전이학습 만들기(함수)
- 텐서보드에 저장하기

**# transfer learning_002**

- 
- 

- 텐서플로우에서 제공하는 모델 사용(EfficientNetB0)
- model0 학습 (feature extraction model)
- 데이터 어규멘 테이션
- model1 학습(1퍼센트 데이터 어규멘테이션)
- model2 학습(10퍼센트 데이터 어규멘테이션)

- checkpoint callback 함수 만들기
- 

## https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html#sklearn.datasets.make_circles
## https://playground.tensorflow.org/#activation=relu&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=2,2&seed=0.93799&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false&regularization_hide=true&regularizationRate_hide=true&batchSize_hide=true