'''Test and tune image processing for parsing puzzle screenshots.'''

import sys, os
import cv2
import numpy as np
import matplotlib.pyplot as plt

file_path = './tests/standard_rectangular/tower-150.png'
# file_path = './tests/standard_globular/amoeba-150.png'
img = cv2.imread(file_path)
cv2.imshow('img', img); cv2.waitKey(0)
 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mean = np.mean(gray)
print(f"{mean=:.2f}")
edges = cv2.Canny(gray, threshold1=1*mean, threshold2=255, apertureSize=3)
# cv2.imshow('edges1', edges); cv2.waitKey(0)

dilate_kernel_size = 9
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_kernel_size,dilate_kernel_size))
edges = cv2.dilate(edges, dilate_kernel, iterations=1)
cv2.imshow('dilate', edges); cv2.waitKey(0)

erode_kernel_size = 11
erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_kernel_size,erode_kernel_size))
edges = cv2.erode(edges, erode_kernel, iterations=1)
cv2.imshow('erode', edges); cv2.waitKey(0)


lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=800, minLineLength=50, maxLineGap=25)

print(f"{dilate_kernel_size=} {erode_kernel_size=} lines: {len(lines)}")
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1,y1), (x2,y2), (0,255,255), 3)

cv2.imshow('hough lines',img); cv2.waitKey(0)