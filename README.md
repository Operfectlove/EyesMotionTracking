# 2022 청소년 노벨캠프 프로젝트
팀명 : Intelligence Brains
조원 : 김지원, 김성윤, 이지원, 이현무

# 제작동기
 자율학습시간에 학습에 집중하지 못하고 조는 친구들을 보며 친구들의 학습집중도를 분석하고 도움이 될 프로그램을 제작하게 됨.

# EyesMotionTracking
dlib와 EAR알고리즘을 이용하여 눈의 감김 정도에 따라 졸음을 인식.
tensorflow를 이용하여 정수리를 인식하고 이를 통해 머리를 숙이고 졸고 있음을 알아냄.

# 이론적배경
실험자의 얼굴 특징점(눈 부분)이 감지되지 않았을 경우 학습에 집중하지 못했다는 사실과 연관지을 수 있다. 따라서 학습에 집중하지 못하는 경우를 눈이 감지되지 않은 경우, 정면을 응시하며 눈을 감은 경우와 고개를 숙이며 잠이드는 경우로 분류하였다. 또한 고개를 숙이고 잠이 든 경우에는 정수리가 감지됨을 알 수 있다. 이를 바탕으로 눈의 면적, 비율변화와 정수리인식을 실시간 영상과 영상 처리 기술에 기반하여 프로그램을 제작하였다. 
 

1. 얼굴특징점검출

  원본 이미지에서 분리된 밝기 채널을 반전하고 원본 그레이스케일 이미지로 구성하여 선명한 이미지를 생성한다. 영상에 있어서 조명의 영향, 특히 그라데이션 조명을 받았을 경우 에러를 일으키는 요소가 되기 때문에 전처리 과정으로 영상에서 조명 영향을 받을 때 그 영향을 최소화하는 작업을 진행한다. 전처리를 위해 영상에서 분리한 Lightness 채널을 반전시키고 Grayscale 된 원본 영상과 합성하여 Clear한 이미지를 생성한다. 
(Luma 기법, LAB 색 공간 모델, 메디안 필터링 사용)
  HOG 얼굴 패턴을 사용하여 Garyscaled-HOG-image에서 얼굴을 찾는다. 그레이스케일링 한 이미지에서 얼굴을 찾기 위해 HOG face pattern을 사용하고 OpenCV와 DLib의 get_frontal_face_detector 함수와 shape_predictor를 이용하여 68개의 얼굴특징점을 검출한다.
    
2. Eyes Aspect Ratio (Algorithm)

  눈의 세로길이 / 가로길이 값을 구하여 졸음을 판단하는 알고리즘으로 거리에 관계없이 일정한 비율을 얻을 수 있다. 각 눈은 6개의 (x, y)좌표로 표시된다. Eyes Aspect Ratio(이하 EAR) 알고리즘은 검출된 안구의 (x, y)좌표에 의해 계산한다. 계산된 EAR은 눈이 열려 있을 때 0보다 큰 값, 눈을 감으면 0에 가까운 값을 가진다. 만약 실험자가 눈을 감을 경우(잠이 들 경우) 눈의 세로 길이가 줄어들면서 EAR 방정식의 수치값이 작아지게 될 것이다. 
추가로 양쪽 눈을 따로 검사할 필요는 없기 때문에 양쪽 눈 각각의 EAR 값을 평균 계산해서 사용한다.

3. 정수리인식

  정수리와 얼굴의 정면 측면사진을 분류하여 전이 학습과 CNN신경망을 이용하여 이미지분류 학습을 진행하여 학습모델을 생성한다.  

# 동작구조
1. 눈 특징점을 추출하여 눈의 감김 비율을 측정하여 임계치 보다 작을 경우 눈을 감았다고 인식함.
2. register에 눈을 감은 시간과 현재시간을 기록함.
3. register에 정수리가 감지된 시간과 현재시간을 기록함.
4. 배열을 csv파일로 기록함.


# 프로젝트 결과 및 보완할점

○ 얼굴특징점 기반의 학습 집중도 분석 시스템 제작
   - 컴퓨터비전(Computer Vision)과 기계학습(Machine Learnig)을 활용하여 자기학습 중 특정 시간대나 과목에서 집중도가 떨어지는 경우에 얼굴특징점(눈 주변부위) 변화를 분석하여 학습 습관 데이터를 수집, 이를 가공한 후 개인별 학습 패턴에 따라 시각화한 자료를 사용자에게 제공하고, 사용자 본인 스스로 문제점을 깨닫고 개선하기 위한 시스템을 제작함.

   - 컴퓨터비전 기술을 접목하여 영상처리를 위해 프로그래밍 언어 Python, C++, OpenCV를 활용하여 얼굴특징점을 추출하였고 사용자의 얼굴 특징점(눈) 변화를 인식함. 이러한 데이터들을 TensorFlow를 이용하여 얼굴특징점 변화에 따라 합성곱 신경망(CNN) 방식으로 제작한 이미지 분류 모델로 분류하여 집중 상태를 파악함. 

   - 분류된 데이터들을 가공하고 데이터 시각화 기법을 활용해 효율적인 분석 그래프 형식으로 사용자 맞춤 자료를 제작하여 사용자에게  제공함. 이를 통해 사용자는 학습 패턴의 문제점을 스스로 개선할 수 있음. 

   - 향후 알고리즘과 자료구조, 프로그래밍 언어(Python, C++)의 특성을 활용하여 소스코드의 품질과 수정 용이성, 안정성을 확보한 질 높은 프로그램으로 개선하여 처음 계획한 <얼굴특징점 기반의 학습 집중도 분석 시스템>이라는 프로젝트를 성공적으로 완성해냄.
 
  ○  학습 집중도 분석 시스템의 적용
   - 본교 학생 20명을 대상으로 인터넷 강의 시청 중 학습 집중도를 분석 시스템을 통해 학습 패턴을 분석하였고 분석한 결과를 바탕으로 제공되는 분석자료를 통해 자신의 학습 습관을 인식하고 스스로 개선해나가게 하여 연구 목적을 이루고 사용자들의 학습효율을 높이는데 기여함. 

   - “<얼굴특징점 기반의 학습 집중도 분석 시스템>을 통해 학습 집중도를 높일 수 있었는가?”라는 질문에 대해 ‘매우만족 7명, 만족 8명, 보통 4명, 불만족 1명, 매우불만족 0명’ 의 설문 결과를 얻었으며 전반적으로 참여자들이 결과에 만족하고 있음을 알 수 있었고 본 프로그램의 실효성과 신뢰성을 검증할 수 있었다. 

   - <얼굴특징점 기반의 학습 집중도 분석 시스템>이 Windows 기반의 동작에 초점을 맞춰 제작되어 최근 학생들이 인터넷 강의 시청을 위해 많이 사용하는 Android나 iPadOS기반의 장치를 사용할 경우 사용을 못한다는 피드백을 받았고 다양한 운영체제에서 활용할 수 있는 크로스플랫폼 앱 개발 프레임워크를 학습하여 시스템을 개선하자는 추가 연구를 계획함.

   - 인간의 집중도, 감정, 사고능력을 해석할 수 있는 부위는 얼굴 특징점 뿐만 아니라 동공 등의 신체 부위가 많다는 점과 조금 더 다양한 측면에서 학습 패턴이 분석, 제공되었으면 한다는 피드백을 수용하여 향후 동공 팽창, 수축을 통해 감정과 정서적, 인지적 신호를 나타낸다는 동공계측학의 원리를 적용한 조금 더 심도 있는 시스템을 연구하기로 계획함. 

#참  고  문  헌

1) Rajneesh, Anudeep Goraya and Gurmeet Singh. Real Time Drivers Drowsiness Detection and alert System by Measuring EAR. International Journal of Computer Applications 181(25):38-45, November 2018.

2)  Oh, Meeyeon, Jeong, Yoosoo, & Park, Kil-Houm. (2016). Driver Drowsiness Detection Algorithm based on Facial Features. Journal of Korea Multimedia Society, 19(11), 1852–1861. 

3) 산업통상자원부 이러닝산업실태조사 2015~2020
