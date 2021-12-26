import numpy as np
import cv2

net = cv2.dnn.readNetFromCaffe("Models/deploy.prototxt", "Models/res10_300x300_ssd_iter_140000_fp16.caffemodel")
# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
image = cv2.imread("Photos/image.jpg")
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()

# loop over the detections
for i in range(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the
	# prediction
	confidence = detections[0, 0, i, 2]
	# filter out weak detections by ensuring the `confidence` is
	# greater than the minimum confidence
	if confidence > 0.15:
		# compute the (x, y)-coordinates of the bounding box for the
		# object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
 
		# draw the bounding box of the face along with the associated
		# probability
		cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 1)
		text = "{:.2f}%".format(confidence * 100)
		cv2.rectangle(image, (startX, startY), ( endX, startY + 20), (0, 255, 0), cv2.FILLED)
		cv2.putText(image, text, (startX, startY + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)


# python detect_faces.py --image rooster.jpg --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel
# https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/