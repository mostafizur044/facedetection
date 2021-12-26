import cv2
 
#Load the haarcascade file
cascPath = "Models/haarcascade_frontalface_default.xml"

# For Image 

img = cv2.imread('Photos/image4.jpg')
  
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
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
  
def display(img, frameName="haarcascade_frontalface_default"):
    h, w = img.shape[0:2]
    neww = 800
    newh = int(neww*(h/w))
    img = cv2.resize(img, (neww, newh))
    cv2.imshow(frameName, img)
    cv2.waitKey(0)

display(img)

cv2.destroyAllWindows()

#  demonstrates one of the largest limitations of Haar cascades, namely, false-positive detections
# Haar cascades tend to be very sensitive to your choice in detectMultiScale parameters. 
# The scaleFactor and minNeighbors being the ones you have to tune most often
# When a Haar cascade thinks a face is in a region, it will return a higher confidence score. 
# If there are enough high confidence scores in a given area, then the Haar cascade will report a positive detection.
# Again, the above example highlights the primary limitation of Haar cascades. While they are fast, you pay the price via:
# False-positive detections
# Less accuracy (as opposed to HOG + Linear SVM and deep learning-based face detectors)
# Manual parameter tuning
# That said, in resource-constrained environments, you just cannot beat the speed of Haar cascade face detection.



# text = "{:.2f}%".format(conf * 10)
# cv2.rectangle(img, (x, y), ( x + w, y + 20), (0, 255, 0), cv2.FILLED)
# cv2.putText(img, text, (x, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


# A computer program that decides whether an image is a positive image (face image) or negative image (non-face image) is called a classifier. 
# A classifier is trained on hundreds of thousands of face and non-face images to learn how to classify a new image correctly. 
# OpenCV provides us with two pre-trained and ready to be used for face detection classifiers:
# Haar Classifier
# LBP Classifier

# Both of these classifiers process images in gray scales, 
# basically because we don't need color information to decide if a picture has a face or not (we'll talk more about this later on). 
# As these are pre-trained in OpenCV, their learned knowledge files also come bundled with OpenCV opencv/data/.

# To run a classifier, we need to load the knowledge files first, as if it had no knowledge, just like a newly born baby (stupid babies).
# Each file starts with the name of the classifier it belongs to. For example, a Haar cascade classifier starts off as haarcascade_frontalface_alt.xml.
