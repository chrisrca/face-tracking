from Frame import *
import numpy as np

ZOOM = 0.25             
SHOWBOX = False         
SCALEFACTOR = 1.22     
MINNEIGHBORS = 8       
MINSIZE = (60, 60)    

faceCascade = cv2.CascadeClassifier("C:\\Users\\goali\\Projects\\Face Tracking\\haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
box = BoundingBox(-1, -1, -1, -1)

while True:
    _, img = cap.read(0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    boxes = faceCascade.detectMultiScale(
        gray,
        scaleFactor=SCALEFACTOR,
        minNeighbors=MINNEIGHBORS,
        minSize=MINSIZE,
    )
    boxes = np.array(boxes)

    if boxes.size > 0:
        boxLrg = largestBox(boxes)
        if box.dim[0] == -1:
            box = boxLrg
        else:
            box.lerpShape(boxLrg)

    frame = Frame(img, box)
    frame.boxIsVisible = SHOWBOX
    frame.setZoom(ZOOM)
    frame.filter()
    box = frame.box

    frame.show()

    k = cv2.waitKey(30)
    if k == 27:
        break
    if k == 49:
        SHOWBOX = not SHOWBOX
    if k == 50:
        ZOOM = max(ZOOM - 0.05, 0.01)
        print(ZOOM)
    if k == 51:
        ZOOM = min(ZOOM + 0.05, 0.99)
        print(ZOOM)

cap.release()