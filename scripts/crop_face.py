import cv2
from PIL import Image
import numpy as np


def detect_eyes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    eyes = []
    for (x, y, w, h) in faces:
         #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        faces = img[y:y + h, x:x + w]
        eye = faces[0:int(h/2), int(w/2):w]
        eyes.append(eye)
    return eyes
	

# Read the input image
img = Image.open('../data/normal_images/two_ppl_faces_stock.jpg')

print(type(np.array(img)))
# # Convert into grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Load the cascade
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# # Detect faces
# faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# # Draw rectangle around the faces and crop the faces
# for idx,(x, y, w, h) in enumerate(faces):
#     #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
#     faces = img[y:y + h, x:x + w]
#     cv2.imshow("face",faces)
#     cv2.imwrite(f'face_{idx}.jpg', faces)
#     right_eye = faces[0:int(h/2), int(w/2):w]
#     cv2.imwrite(f'right_eye_{idx}.jpg', right_eye)
	
# # Display the output
# cv2.imwrite('detcted.jpg', img)
# cv2.imshow('img', img)
# cv2.waitKey()
