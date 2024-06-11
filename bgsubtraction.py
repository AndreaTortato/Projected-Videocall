import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fg_only_frame = fgbg.apply(frame)

    # Convert single-channel mask to three-channel mask
    fg_only_frame = cv2.cvtColor(fg_only_frame, cv2.COLOR_GRAY2BGR)

    # Multiply pixel values of the original frame by the processed frame
    result_frame = cv2.multiply(frame.astype(float), fg_only_frame.astype(float) / 255, dtype=cv2.CV_8UC3)

    cv2.imshow('Result Frame', result_frame.astype(np.uint8))

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()