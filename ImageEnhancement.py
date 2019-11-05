import cv2
import numpy as np
import os


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def resize_image(img):
    """Load an image and scale"""

    # # Load image
    # img = cv2.imread(img_path)

    # Obtain parameters for scaling
    height, width, depth = img.shape
    imgScale = 800 / width
    newX, newY = img.shape[1] * imgScale, img.shape[0] * imgScale

    # Rescale the image
    return cv2.resize(img, (int(newX), int(newY)))


def store_image(img, img_name, image_type, directory_path):
    """Store a given  image in a specified directory"""

    img_name = img_name+image_type

    # Check if path exists, if not then create the path
    if not os.path.exists(directory_path):
        print("Directory %s does not exist" % directory_path)

    # Store region image
    try:
        cv2.imwrite(os.path.join(directory_path, str(img_name)), img)
    except OSError:
        print("Storage of %s failed on path" % img_name % directory_path)
    else:
        print("Stored: ", os.path.join(directory_path, str(img_name)))


def show_image(img):
    cv2.imshow('Image', img)
    cv2.waitKey(0)


def update_progress_bar(percent):
    """Updating the progress bar of the UI"""
    print("Progress bar updated to " + str(percent) + "%")
    print()


def image_enhancement(image, directory_path, image_type='.jpg'):

    img = resize_image(image)

    # Apply gamma correction to lower the brightness of the image
    gamma = adjust_gamma(img, .25)

    # Change Image Color Encoding
    src = cv2.cvtColor(gamma, cv2.COLOR_RGB2YUV)

    # Equalize the histogram of the Y channel
    src[:, :, 0] = cv2.equalizeHist(src[:, :, 0])

    # convert the YUV image back to RGB format
    dst = cv2.cvtColor(src, cv2.COLOR_YUV2RGB)

    # Apply gamma correction to raise the brightness of the image
    dst = adjust_gamma(dst, 1.5)

    store_image(dst, "Enhanced_Image", image_type, directory_path)

    update_progress_bar(20)

    cv2.imshow('OrigImage', img)
    cv2.imshow('Gamma(0.25)+EqHistImage+Gamma(1.5)', dst)
    cv2.waitKey(0)

    return dst


if __name__ == "__main__":
    image = cv2.imread('bamboo.jpg')

    Enhanced_Image = image_enhancement(image, ".jpg", "C:\\Users\\Caloj\\Desktop\\Sprout_Images")
