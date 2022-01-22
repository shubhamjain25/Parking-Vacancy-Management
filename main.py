import cv2
import pickle
import cvzone
import numpy as np

width, height = 107, 48

#Video Feed
cap = cv2.VideoCapture('carPark.mp4')

with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)

def checkParkingSpace(imgProcessed):
    spaceCounter = 0

    for pos in posList:
        x, y = pos

        imgCrop = imgProcessed[y:y + height, x:x + width]
        # cv2.imshow(str(x * y), imgCrop)

        # Counts the number of pixels in the image
        count = cv2.countNonZero(imgCrop)

        if count < 900:
            # Parking Space is vacant
            color = (0, 255, 0)
            thickness = 5
            spaceCounter += 1
        else:
            # Parking Space is occupied
            color = (0, 0, 255)
            thickness = 2

        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        cvzone.putTextRect(img, str(count), (x, y + height - 3), scale=1,
                           thickness=2, offset=0, colorR=color)

    # Shows the real-time status of vacant spaces in the parking
    cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (100, 50), scale=3,
                           thickness=5, offset=20, colorR=(0,200,0))

while True:

    # So that video goes on in a loop(only for this case)
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)

    success, img = cap.read()

    #Converting Image

    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(3,3),1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,25,16)

    # To additionally reduce noise
    imgMedian = cv2.medianBlur(imgThreshold, 5)

    # To make the pixels little bit thick to easily differentiate
    kernel = np.ones((3,3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)


    checkParkingSpace(imgDilate)
    for pos in posList:
        cv2.rectangle(img,pos,(pos[0]+width,pos[1]+height),(255,0,255),2)

    cv2.imshow("ImageGray", imgGray)
    cv2.imshow("ImageThreshold", imgThreshold)
    cv2.imshow("Image",img)

    cv2.waitKey(30)