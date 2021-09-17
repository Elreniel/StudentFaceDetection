import cv2
import numpy as np

video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cascade_classifier = cv2.CascadeClassifier("frontalFace_default.xml")

template = cv2.imread('template.jpg')
tempHeight = template.shape[0]
tempWidth = template.shape[1]
threshold = 0.7

while True:
    ret, img = video_capture.read()

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected_faces = cascade_classifier.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5)

    for (x, y, width, height) in detected_faces:
        cv2.rectangle(img, (x, y), (x + width, y + height), (0, 0, 255), 1)
        cv2.putText(img, "Face", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    # TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_SQDIFF, TM_SQDIFF_NORMED
    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    yloc, xloc = np.where(result >= threshold)

    rectangles = []
    for (x, y) in zip(xloc, yloc):
        rectangles.append([int(x), int(y), int(tempWidth), int(tempHeight)])
        rectangles.append([int(x), int(y), int(tempWidth), int(tempHeight)])

    rectangles, weights = cv2.groupRectangles(rectangles, 1, 0.2)

    for (x, y, width, height) in rectangles:
        cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 1)
        cv2.putText(img, "Student", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

    cv2.imshow("Face Detection", img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
video_capture.release()
cv2.destroyAllWindows()
