import numpy as np
import cv2
 
#Load the haarcascade file
cascPath = "Models/haarcascade_frontalface_default.xml"

video_capture = cv2.VideoCapture(0)

haar_cascade = cv2.CascadeClassifier(cascPath)
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray,
                                         scaleFactor=1.05,
                                         minNeighbors=6,
                                         minSize=(30, 30),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h),(0,255,0), 2)
        # Display the resulting frame
    cv2.imshow('haar_cascade__video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()

