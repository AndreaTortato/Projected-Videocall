import sys
import cv2
import tkinter as tk
import numpy as np

cap = cv2.VideoCapture(0)

w, h = 853, 480 #window size

# Get the display resolution
root = tk.Tk()
w = root.winfo_screenwidth()
h = root.winfo_screenheight()
root.destroy()

cap.set(3, w)
cap.set(4, h)
cap.set(cv2.CAP_PROP_FPS, 60)

sys.path.insert(1, 'segmentation.py')
from segmentation import SelfiSegmentation
segmentor = SelfiSegmentation()

sys.path.insert(1, 'detectionFace.py')
from detectionFace import FaceDetection
face_detector = FaceDetection()

sys.path.insert(1, 'keystone.py')
from keystone import Keystone
keystone = Keystone()

ksPoints = np.float32([[0, 0], [w, 0], [0, h], [w, h]]) # starting keystone points
cornerPoints = np.float32([[0, 0], [w, 0], [0, h], [w, h]])  # original frame corners

key = 0
selected_point = -1

while True:
    success, imgOut = cap.read()  # Read a frame from the webcam

    #imgOut = face_detector.detect_faces(imgOut) # Detect face
    imgOut = segmentor.removeBG(imgOut, (0, 0, 0), cutThreshold=0.8) # Remove background

    # Draw projection corners
    for i, pt in enumerate(ksPoints, start=1):
        cv2.circle(imgOut, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
        cv2.putText(imgOut, str(i), (int(pt[0]) + 10, int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Image Out", cv2.resize(imgOut, (w, h)))

    # Handle key events
    key_event = cv2.waitKey(1) & 0xFF
    if key_event != 255:
        key = key_event

    if ord('1') <= key <= ord('4'):  # Choose a point
        selected_point = key - ord('1')

    if selected_point != -1:

        if key == 97: # Left arrow key A
            ksPoints[selected_point][0] = ksPoints[selected_point][0] - 10
        if key == 100: # Right arrow key D
            ksPoints[selected_point][0] = ksPoints[selected_point][0] + 10
        if key == 119: # Up arrow key W
            ksPoints[selected_point][1] = ksPoints[selected_point][1] - 10
        if key == 115: # Down arrow key S
            ksPoints[selected_point][1] = ksPoints[selected_point][1] + 10

    matrix = cv2.getPerspectiveTransform(ksPoints, cornerPoints)
    result = cv2.warpPerspective(imgOut, matrix, (w, h))
    cv2.imshow('Perspective Transformation', result)

    if key == 27: # ESC
        break

cap.release()
cv2.destroyAllWindows()