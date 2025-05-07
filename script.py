import cv2
import numpy as np
import easyocr
from matplotlib import pyplot as plt
import imutils
import random
image = cv2.imread("test.jpeg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
filtered = cv2.bilateralFilter(gray, 11, 17, 17)
canny = cv2.Canny(filtered, 30, 200)
keypoints = cv2.findContours(canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx)== 4:
        location = approx
        break
mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0, 255, -1)
new_image = cv2.bitwise_and(image, image, mask=mask)
(x, y) = np.where(mask == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped = gray[x1:x2+1, y1:y2+1]
reader = easyocr.Reader(['en'])
result = reader.readtext(cropped)
print(result)
cv2.waitKey(0)
cv2.destroyAllWindows
