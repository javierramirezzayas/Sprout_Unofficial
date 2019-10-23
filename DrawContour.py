import cv2
import numpy as np
import math

img = cv2.imread('control.jpg')
height, width, depth = img.shape
imgScale = 700/width
newX,newY = img.shape[1] * imgScale, img.shape[0] * imgScale
img = cv2.resize(img, (int(newX), int(newY)))
img2 = img.copy()

# Make a mask out of the input image
mask = np.zeros(img.shape[:2],dtype = 'uint8')

# Convert image to grayscale image*
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# convert the grayscale image to binary image*
ret, thresh = cv2.threshold(gray_image, 127, 255, 0)

# calculate moments of binary image
M = cv2.moments(thresh)

# calculate x,y coordinate of center
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])

# Calculate Angle. Creating a right triangle
angle = 22.5
op_side = math.tan(math.radians(angle)) * cX
print(op_side)
c = [cX, cX*2, cX*2]
r = [cY, cY, int(cY-op_side)]

# Generate array of three points that compose the right traingle
rc = np.array((c,r)).T

# Draw contours for wedge
cv2.drawContours(mask,[rc],0,255,-1)
mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)

# Extract Wedge(Countour)
out = np.zeros_like(img)
out[mask == 255] = img[mask == 255]

# Determine Center of Wedge
(h2, w2) = out.shape[:2]
center = (w2/2, h2/2)

# Rotate Wedge
M = cv2.getRotationMatrix2D(center, 90, 1.0)
rotated90 = cv2.warpAffine(out, M, (h2, w2))

# Find countour for wedge
img_dilation = cv2.dilate(rotated90, None, iterations=9)
img_erosion = cv2.erode(img_dilation, None, iterations=8)
edged=cv2.Canny(img_erosion,30,200)
contours, hierarchy=cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(rotated90,contours,-1,(0,255,0),3)

cv2.imshow('Binarized Image', rotated90)
cv2.waitKey(0)

areas = [cv2.contourArea(c) for c in contours]
max_index = np.argmax(areas)
cnt=contours[max_index]
x,y,w,h = cv2.boundingRect(cnt)
rect = cv2.rectangle(rotated90,(x,y),(x+w,y+h),(0,255,0),2)

rotated90 = rotated90[y:y+h, x:x+w]

# # Let's get the starting pixel coordiantes (top left of cropped top)
# start_row, start_col = int(0), int(0)
# # Let's get the ending pixel coordinates (bottom right of cropped top)
# end_row, end_col = int(h * .5), int(w)
# cropped_top = rotated90[start_row:end_row , start_col:end_col]
# start_row, start_col = int(h * .5), int(0)
# # Let's get the ending pixel coordinates (bottom right of cropped bottom)
# end_row, end_col = int(h), int(w)
# cropped_bot = rotated90[start_row:end_row , start_col:end_col]







# # For display
# cv2.drawContours(img,[rc],0,255,2)
#
# cv2.imshow('Binarized Image', img2)
# cv2.waitKey(0)
# cv2.imshow('Binarized Image w/ Contour', img)
# cv2.waitKey(0)
# cv2.imshow('Wedge Image', out)
# cv2.waitKey(0)
# cv2.imshow('Rot Wedge Image',rotated90)
# cv2.imshow('Rot Wedge Image Seg 1',cropped_top)
# cv2.imshow('Rot Wedge Image Seg 2',cropped_bot)
cv2.waitKey(0)