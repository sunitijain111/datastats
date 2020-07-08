import time
import cv2 as cv
cap = cv.VideoCapture('output.avi')
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('slow_motion.avi', fourcc, 5, (640,  480))

#rate= 4

while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("video ended")
        break
    
    #for i in range(rate):
    out.write(frame)
    
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
out.release()
cv.destroyAllWindows()