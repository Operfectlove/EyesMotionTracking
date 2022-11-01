
import numpy as np
import imutils
import time
import dlib
import cv2
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread

import make_train_data as mtd
import light_remover as lr
import datetime
import tensorflow
register = []

def eye_aspect_ratio(eye) :
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
    
def init_open_ear() :
    time.sleep(5)
    print("open init time sleep")
    ear_list = []
    for i in range(7) :
        ear_list.append(both_ear)
        time.sleep(1)
    global OPEN_EAR
    OPEN_EAR = sum(ear_list) / len(ear_list)
    print("open list =", ear_list, "\nOPEN_EAR =", OPEN_EAR, "\n")

def init_close_ear() : 
    time.sleep(2)
    th_open.join()
    time.sleep(5)
    ear_list = []
    time.sleep(1)
    for i in range(7) :
        ear_list.append(both_ear)
        time.sleep(1)
    CLOSE_EAR = sum(ear_list) / len(ear_list)
    global EAR_THRESH
    EAR_THRESH = (((OPEN_EAR - CLOSE_EAR) / 2) + CLOSE_EAR) #EAR_THRESH means 50% of the being opened eyes state
    print("close list =", ear_list, "\nCLOSE_EAR =", CLOSE_EAR, "\n")
    print("The last EAR_THRESH's value :",EAR_THRESH, "\n")

#####################################################################################################################
#1. Variables for checking EAR.
#2. Variables for detecting if user is asleep.
#3. When the alarm rings, measure the time eyes are being closed.
#4. When the alarm is rang, count the number of times it is rang, and prevent the alarm from ringing continuously.
#5. We should count the time eyes are being opened for data labeling.
#6. Variables for trained data generation and calculation fps.
#7. Detect face & eyes.
#8. Run the cam.
#9. Threads to run the functions in which determine the EAR_THRESH. 

#1.
OPEN_EAR = 0 #For init_open_ear()
EAR_THRESH = 0 #Threashold value

#2.
#It doesn't matter what you use instead of a consecutive frame to check out drowsiness state. (ex. timer)
EAR_CONSEC_FRAMES = 20 
COUNTER = 0 #Frames counter.

#3.
closed_eyes_time = [] #The time eyes were being offed.
TIMER_FLAG = False #Flag to activate 'start_closing' variable, which measures the eyes closing time.
ALARM_FLAG = False #Flag to check if alarm has ever been triggered.

#4. 
ALARM_COUNT = 0 #Number of times the total alarm rang.
RUNNING_TIME = 0 #Variable to prevent alarm going off continuously.

#5.    
PREV_TERM = 0 #Variable to measure the time eyes were being opened until the alarm rang.

#6. make trained data 
np.random.seed(9)
power, nomal, short = mtd.start(25) #actually this three values aren't used now. (if you use this, you can do the plotting)
#The array the actual test data is placed.
test_data = []
#The array the actual labeld data of test data is placed.
result_data = []
#For calculate fps
prev_time = 0

#7. 
print("loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
predictor = dlib.shape_predictor("C:\sdkassignment\EyesBlinkTracking\shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

#8.
print("starting video stream thread...")
vs = VideoStream(src=0,).start()
time.sleep(1.0)

#9.
th_open = Thread(target = init_open_ear)
th_open.deamon = True
th_open.start()
th_close = Thread(target = init_close_ear)
th_close.deamon = True
th_close.start()
before = datetime.datetime.now()
f = open('t.txt', 'w')
#########################################################################################
model_filename ='C:\sdkassignment\EyesBlinkTracking\keras_model.h5'
#model_filename ='C:\sdkassignment\EyesBlinkTracking\keras_model.h5'

# 케라스 모델 가져오기
model = tensorflow.keras.models.load_model(model_filename)

# 카메라를 제어할 수 있는 객체
capture = cv2.VideoCapture(0)

# 카메라 길이 너비 조절
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# 이미지 처리하기
def preprocessing(frame):
    #frame_fliped = cv2.flip(frame, 1)
    # 사이즈 조정 티쳐블 머신에서 사용한 이미지 사이즈로 변경해준다.
    size = (224, 224)
    frame_resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    
    # 이미지 정규화
    # astype : 속성
    frame_normalized = (frame_resized.astype(np.float32) / 127.0) - 1

    # 이미지 차원 재조정 - 예측을 위해 reshape 해줍니다.
    # keras 모델에 공급할 올바른 모양의 배열 생성
    frame_reshaped = frame_normalized.reshape((1, 224, 224, 3))
    #print(frame_reshaped)
    return frame_reshaped

# 예측용 함수
def predict(frame):
    prediction = model.predict(frame)
    return prediction
#####################################################################################################################

def main():
    global frame
    global preprocessed
    global prediction
    global L, gray
    global rects
    global dt, now, before, both_ear, dt_str
    frame = vs.read()
    frame = imutils.resize(frame, width = 400)
    preprocessed = preprocessing(frame)
    prediction = predict(preprocessed)
    L, gray = lr.light_removing(frame)
    
    rects = detector(gray,0)

    #checking fps. If you want to check fps, just uncomment below two lines.
    #prev_time, fps = check_fps(prev_time)
    #cv2.putText(frame, "fps : {:.2f}".format(fps), (10,130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,30,20), 2)
    if (prediction[0,0] < prediction[0,1]):
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

        #(leftEAR + rightEAR) / 2 => both_ear. 
            both_ear = (leftEAR + rightEAR) * 500  #I multiplied by 1000 to enlarge the scope.

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0,255,0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0,255,0), 1)
        

            if both_ear < 210 :
                now = datetime.datetime.now()
                dt= now - before
                if dt.seconds >=3:
                    cv2.putText(frame,  "event", (250,120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                    dt_str = str(dt.seconds)
                    before_str = str(before)
                    register.append(['eyes', dt_str, before_str])
                    f.write('eyes: '+ dt_str +','+ before_str +'\n')
                    return dt_str
            else:
                before = datetime.datetime.now() 
                register.append(['0', 0, 0])

            cv2.putText(frame, "EAR : {:.2f}".format(both_ear), (300,130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,30,20), 2)
        
        cv2.imshow("Frame",frame)
        key = cv2.waitKey(1) & 0xFF

        
    elif (prediction[0,1] < prediction[0,0]):
        now = datetime.datetime.now()
        dt= now - before
        if dt.seconds >=3:
            cv2.putText(frame,  "event2", (250,120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
            dt_str = str(dt.seconds)
            before_str = str(before)
            register.append(['head', dt_str, before_str])
            f.write('head: '+dt_str +','+ before_str +'\n')
            return dt_str
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1) & 0xFF

'''
def record(x):
    timeout = 2
    start_time = time.time()
    while True:
        # change and to or. if there is an input or timeout
        if k != x and ((time.time() - start_time) > timeout):
            q.write(x)
        else: 
            k = x
'''


if __name__ == '__main__':
    #global q
    #q = open('tf.txt', 'w')
        #dt_str = ''
    while True:
        main()
        cv2.imshow("Frame",frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            for i in range(len(register) - 1):
                a = register[i]
                log1 = a[1]
                b = register[i+1]
                log2 = b[1]
                
                if int(log1) > int(log2):
                    log3 = str(register[i])
                    f.write(log3)
                
                    

            
            break

    cv2.destroyAllWindows()
    vs.stop()
