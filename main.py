import cv2
import numpy as np
from helper import *
import pandas as pd
import time
from matplotlib import pyplot as plt


def getCenterMask(image_name:str):
    image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    image = np.array(image)

    Xs, ys = np.where(image == 255)

    x_mean = np.mean(Xs)
    y_mean = np.mean(ys)
    center = int(x_mean), int(y_mean)
    print(center)

    # in this part I try to make the pixel red but or it doesnt work or I cannot find it
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image[int(y_mean), int(x_mean)] = (0, 0, 255)
    image = cv2.resize(image, (256, 256))
    cv2.imshow("image", image)
    cv2.waitKey(0)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)
start_time = time.time()
timer = 0.5
face_img = None
while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=20)
    # Show the main webcam feed
    cv2.imshow('Webcam Feed', frame)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]

        # Crop the face from the original frame
        face_img = frame[y - 50:y + h + 50, x - 50:x + w + 50]
        face_img = cv2.resize(face_img, (128, 128))
        # Optionally display the cropped face in another window
        cv2.imshow('Cropped Face', face_img)
    else:
        face_img = frame
        face_img = cv2.resize(face_img, (128, 128))
    if time.time() - start_time >= timer:
        start_time = time.time()
        face_img = np.expand_dims(face_img, axis=0)  # add batch dimension
        prediction = modelLoading("unet_model.keras").predict(face_img)
        prediction = np.squeeze(prediction)
        prediction = (prediction > 0.5).astype(np.uint8)
        plt.subplot(1, 2, 1)
        plt.imshow(np.squeeze(face_img), cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(prediction)
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
# # face_img = cv2.resize(face_img, (128, 128))
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
# prediction = prediction*255
# cv2.imwrite("testing/prediction_002.jpg", prediction)
# cv2.imshow("prediction", prediction)
# cv2.waitKey(0)
#
# getCenterMask("testing/prediction_002.jpg")
