import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
inputimage=cv.imread("S:\\Hi5\\Accesia\\ExportedXrays\\8.JPG")
gray=cv.cvtColor(inputimage,cv.COLOR_BGR2GRAY)
#plt.imshow(gray,cmap='gray')

gblurr=cv.GaussianBlur(gray,(3,3),0)
dkernel = cv.getStructuringElement(cv.MORPH_RECT, (35, 35))
ekernel=cv.getStructuringElement(cv.MORPH_RECT, (35, 35))

erode=cv.erode(gblurr,ekernel)
#cv.imshow('eroded',erode)
dilate=cv.dilate(erode,dkernel)
#cv.imshow('dilated',dilate)
plt.imshow(dilate,cmap='gray')
ret,threshold=cv.threshold(gblurr,80,200,cv.THRESH_BINARY+cv.THRESH_BINARY_INV)

#
# plt.figure(2)
# plt.imshow(gblurr,cmap='gray')
#
# Applying threshold
# threshold = cv.threshold(gblurr, 60, 200,
# cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]

#cv.imshow("thresholded",threshold)

# Apply the Component analysis function
analysis = cv.connectedComponentsWithStats(threshold,
                                            8,
                                            cv.CV_32S)
(totalLabels, label_ids, values, centroid) = analysis

# Initialize a new image to
# store all the output components
output = np.zeros(gray.shape, dtype="uint8")
print(values)

for i in range(1, totalLabels):

    # Area of the component
    area = values[i, cv.CC_STAT_AREA]
   # print(np.max(area))

    if (area > 500) or (area < 1000):
        componentMask = (label_ids == i).astype("uint8") * 255
        plt.imshow(componentMask,cmap='binary')
        plt.show()
        output = cv.bitwise_or(output, componentMask)

#cv.imshow("Image", inputimage)
cv.imshow("Filtered Components", output)
edges=cv.Canny(output,0,255)
cv.imshow("edges",edges)


plt.show()

cv.waitKey(0)

