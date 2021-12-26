import numpy as np
import cv2
 
#Load the lbpcascade file
cascPath = "Models/lbpcascade_frontalface.xml"

img = cv2.imread('Photos/image1.jpg')
  
# Converting image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
# Loading the required haar-cascade xml classifier file
haar_cascade = cv2.CascadeClassifier(cascPath)
  
# Applying the face detection method on the grayscale image
faces, rejectLevels, levelWeights = haar_cascade.detectMultiScale3(gray_img,
                                         scaleFactor=1.05,
                                         minNeighbors=6,
                                         minSize=(30, 30),
                                         flags=cv2.CASCADE_SCALE_IMAGE,
                                         outputRejectLevels=True)

# Iterating through rectangles of detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
  
#Display image
def display(img, frameName="lbpcascade_frontalface"):
    h, w = img.shape[0:2]
    neww = 800
    newh = int(neww*(h/w))
    img = cv2.resize(img, (neww, newh))
    cv2.imshow(frameName, img)
    cv2.waitKey(0)

display(img)

cv2.destroyAllWindows()

# https://www.superdatascience.com/blogs/opencv-face-detection/