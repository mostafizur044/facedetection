import numpy as np
import cv2

net = cv2.dnn.readNetFromCaffe("Models/deploy.prototxt", "Models/res10_300x300_ssd_iter_140000_fp16.caffemodel")

video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Cannot open camera")
    exit()


while True:
	ret, frame = video_capture.read()
	h, w = frame.shape[0:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
 
	net.setInput(blob)
	detections = net.forward()

	# # loop over the detections
	for i in range(0, detections.shape[2]):
		
		confidence = detections[0, 0, i, 2]
		
		if confidence < 0.5:
			continue

		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 1)
		text = "{:.2f}%".format(confidence * 100)
		cv2.rectangle(frame, (startX, startY), ( endX, startY + 20), (0, 255, 0), cv2.FILLED)
		cv2.putText(frame, text, (startX, startY + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
	
	# show the output frame
	cv2.imshow('detection_deep_learnong_video', frame)
  
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
		
# do a bit of cleanup
cv2.destroyAllWindows()


# python detect_faces.py --image rooster.jpg --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel
# https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/