import cv2
img=cv2.imread(r'C:\Users\Mariam\Downloads\usedon.jpeg')
#img=cv2.line(img,(0,0),(255,255),(0,0,255),5)
GRID_SIZE = 40

"""
height = img.shape[0]
width = img.shape[1]
channels = img.shape[2]
"""
for x in range(0, 20000 -1, GRID_SIZE):
     cv2.line(img, (x,0), (x,20000), ( 0, 0,255), 1, 1)

for x in range(0, 20000 -1, GRID_SIZE):
     cv2.line(img, (0,x), (20000,x), ( 0, 0,255), 1, 1)
#print(img.shape)
"""
import cv2
import numpy as np
import matplotlib
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt

# white color mask

#converted = convert_hls(img)
image = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
lower = np.uint8([0, 200, 0])
upper = np.uint8([255, 255, 255])
white_mask = cv2.inRange(image, lower, upper)
# yellow color mask
lower = np.uint8([10, 0,   100])
upper = np.uint8([40, 255, 255])
yellow_mask = cv2.inRange(image, lower, upper)
# combine the mask
mask = cv2.bitwise_or(white_mask, yellow_mask)
result = img.copy()
cv2.imshow("mask",mask)

print(mask)

width=700
height=700
y1=0
y2=700
x1=500
x2=700

#img=cv2.resize(img,(width,height)) #resize image
roi = img[y1:y2, x1:x2] #region of interest i.e where the rectangles will be
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) #convert roi into gray
Blur=cv2.GaussianBlur(gray,(5,5),1) #apply blur to roi
Canny=cv2.Canny(Blur,10,50) #apply canny to roi

#Find my contours
contours =cv2.findContours(Canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]

#Loop through my contours to find rectangles and put them in a list, so i can view them individually later.
cntrRect = []
for i in contours:
        epsilon = 0.05*cv2.arcLength(i,True)
        approx = cv2.approxPolyDP(i,epsilon,True)
        if len(approx) == 4:
            cv2.drawContours(roi,cntrRect,-1,(0,255,0),2)
            cv2.imshow('Roi Rect ONLY',roi)
            cntrRect.append(approx)
"""
cv2.imshow('image',img)
key = cv2.waitKey(0)
