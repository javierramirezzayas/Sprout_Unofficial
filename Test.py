import numpy as np
import cv2

def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return rot_mat

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

img = cv2.imread('control.jpg')
height, width, depth = img.shape
imgScale = 700/width
newX,newY = img.shape[1] * imgScale, img.shape[0] * imgScale
img = cv2.resize(img, (int(newX), int(newY)))

cv2.imshow("Orig Img", img)
cv2.waitKey(0)
cv2.imshow("Orig Img", rotate_bound(img, 90))
cv2.waitKey(0)

angle = 0.9
i=0
while (i<360):
    img2 = rotateImage(img, i)
    i = i + angle
    if (int(i)==360):
        cv2.imshow("Rotated Image", img2)
        cv2.waitKey(0)
