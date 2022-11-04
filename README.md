# EyesMotionTracking

1. register에 눈을 감은 시간과 현재시간을 기록
2. register에 정수리가 감지된 시간과 현재시간을 기록
3. 배열을 mmap() 함수로 나누어 시간 (i) 번째 시간과 (i+1)의 시간을 비교하고 큰 값을 t.txt파일에 기록
