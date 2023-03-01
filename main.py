
import cv2
import numpy as np

cap = cv2.VideoCapture("Resource/highway_seaside1.mp4")

object_detector = cv2.createBackgroundSubtractorMOG2(history=70, varThreshold=20, detectShadows=10)

while True:
    ret,frame = cap.read()
    frame = cv2.resize(frame, (900,1600))
    gray = cv2.convertScaleAbs(frame, 10, 0.5)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ##Region of interesting
    roi = gray[0:1450,0:500]
    ##Remove background from unstable object
    mask = object_detector.apply(roi)
    _,mask = cv2.threshold(mask,250,255,cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area >80:
            # cv2.drawContours(roi,[contour],-1,(255,0,0),1)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (20,130,255), 2)

    cv2.imshow("Camera capture",frame)
    # mask = cv2.resize(mask, (384*4, 216*4))
    cv2.imshow("Background removal", mask)
    # cv2.imshow("Region of interesting", roi)

    if cv2.waitKey(1) == ord("c"):
        break
cap.release()
cv2.destroyAllWindows()
