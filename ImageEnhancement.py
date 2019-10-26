import cv2
import numpy as np


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def load_image(img_path):
    """Load an image and scale"""

    # Load image
    img = cv2.imread(img_path)

    # Obtain parameters for scaling
    height, width, depth = img.shape
    imgScale = 700 / width
    newX, newY = img.shape[1] * imgScale, img.shape[0] * imgScale

    # Rescale the image
    return cv2.resize(img, (int(newX), int(newY)))


def show_image(img):
    cv2.imshow('Image', img)
    cv2.waitKey(0)


def main():
    img = load_image('bamboo.jpg')

    gamma = adjust_gamma(img, .25)

    src = cv2.cvtColor(gamma, cv2.COLOR_RGB2YUV)

    # equalize the histogram of the Y channel
    src[:, :, 0] = cv2.equalizeHist(src[:, :, 0])

    # convert the YUV image back to RGB format
    dst = cv2.cvtColor(src, cv2.COLOR_YUV2RGB)

    dst = adjust_gamma(dst, 1.5)

    cv2.imshow('OrigImage', img)
    cv2.imshow('Gamma(0.25)+EqHistImage+Gamma(1.5)', dst)
    cv2.waitKey(0)


# def main():
#     img = load_image('bamboo.jpg')
#
#     gamma = adjust_gamma(img, .25)
#
#     src = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
#
#     # equalize the histogram of the Y channel
#     src[:, :, 0] = cv2.equalizeHist(src[:, :, 0])
#
#     # convert the YUV image back to RGB format
#     dst = cv2.cvtColor(src, cv2.COLOR_YUV2RGB)
#
#     dst = adjust_gamma(img, 0.75)
#
#     cv2.imshow('OrigImage', img)
#     cv2.imshow('Gamma(0.25)+EqHistImage+Gamma(0.75)', dst)
#     cv2.waitKey(0)


if __name__== "__main__":
  main()