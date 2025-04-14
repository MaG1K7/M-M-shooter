import cv2
import numpy as np
from helper import *
import pandas as pd
import time
from matplotlib import pyplot as plt


def getCenterMask(image_name):
    M = cv2.moments(image_name)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return cX, cY
    else:
        return 0,0

#
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)
face_img = None
prediction = np.zeros([128,128])
origin_shape = [250,250]
face_x,face_y = 0,0
mouth_x, mouth_y = 0,0
margin=0.4
while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=20)
    # Show the main webcam feed
    cv2.imshow('Webcam Feed', frame)
    if len(faces) > 0:
        faceDetected = True
        (x, y, w, h) = faces[0]

        margin_x = int(w * margin)  # margin for width
        margin_y = int(h * margin)  # margin for height

        # Adjust the coordinates to include the margin
        x = max(x - margin_x, 0)
        y = max(y - margin_y, 0)
        w = min(w + 2 * margin_x, frame.shape[1] - x)
        h = min(h + 2 * margin_y, frame.shape[0] - y)

        face_x, face_y= x,y
        # Crop the face from the original frame
        face_img = frame[y:y + h, x:x + w]
        origin_shape = face_img.shape
        face_img = cv2.resize(face_img, (128, 128))
        # Optionally display the cropped face in another window
    else:
        faceDetected = False
        face_img = np.zeros([128,128,1])
        # face_img = frame
        # face_img = cv2.resize(face_img, (128, 128))
    if faceDetected:
        # predict
        input = np.expand_dims(face_img, axis=0)  # add batch dimension
        prediction = modelLoading("unet_model.keras").predict(input)
        prediction = np.squeeze(prediction)
        prediction = (prediction > 0.5).astype(np.uint8)
        prediction = prediction * 255
        face_with_prediction = np.maximum(face_img, prediction)
        prediction = cv2.resize(prediction, origin_shape)
        mouth_x,mouth_y = face_x+getCenterMask(prediction)[0], face_y+getCenterMask(prediction)[1]
        prediction = prediction*255
        # face_with_prediction = np.maximum(face_img,prediction)
        # show predict
        cv2.imshow('Cropped Face', face_with_prediction)
        plt.figure(figsize=(6, 6))
        plt.imshow(frame)
        plt.plot(mouth_x, mouth_y, 'ro')  # red dot at the centroid
        plt.text(mouth_x + 5, mouth_y, f'({mouth_x},{mouth_y})', color='red')
        plt.axis('off')
        plt.show()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# test_image = cv2.imread("testing/test_002.jpg", cv2.IMREAD_GRAYSCALE)
#
# # Detect faces
# faces = face_cascade.detectMultiScale(test_image, scaleFactor=1.1, minNeighbors=20)
# face_img = None
# if len(faces) > 0:
#     (x, y, w, h) = faces[0]
#     # Draw rectangle (optional)
#     cv2.rectangle(test_image, (x, y), (x+w+100, y+h+100), (255, 0, 0), 2)
#
#     # Crop the face from the original frame
#     face_img = test_image[y-50:y+h+50, x-50:x+w+50]
#     face_img = cv2.resize(face_img, (128, 128))
# else:
#     face_img = test_image
# cv2.imshow("test_image", face_img)
# cv2.waitKey(0)
#
# face_img = cv2.resize(face_img, (128, 128))
# # image = cv2.imread("testing/test_001.jpg", cv2.IMREAD_GRAYSCALE)
# # image =cv2.resize(image, (128, 128))
# face_img = np.expand_dims(face_img, axis=0)  # add batch dimension
# prediction = modelLoading("unet_model.keras").predict(face_img)
# prediction = np.squeeze(prediction)
# prediction = (prediction > 0.5).astype(np.uint8)
# plt.subplot(1,2,1)
# plt.imshow(np.squeeze(face_img), cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(prediction)
# plt.show()
# prediction = prediction
# cv2.imwrite("testing/prediction_002.jpg", prediction)
# cv2.imshow("prediction", prediction)
# cv2.waitKey(0)
#
# getCenterMask(prediction)
