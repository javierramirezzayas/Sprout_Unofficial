import cv2
import numpy as np
import math
import os
from random import randrange


def load_image(img_path):
    """Load an image and scale"""

    # Load image
    img = cv2.imread(img_path)

    # Obtain parameters for scaling
    height, width, depth = img.shape
    imgScale = 50 / width
    newX, newY = img.shape[1] * imgScale, img.shape[0] * imgScale

    # Rescale the image
    return cv2.resize(img, (int(newX), int(newY)))


def create_mask(img):
    """Create mask out of input image"""
    return np.zeros_like(img)


def binarize_image(img):
    """Convert input image to binary image (Won't be used when integrated)"""

    # Convert image to grayscale image*
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # convert the grayscale image to binary image*
    #   - src: image source
    #   - threshold_value: The thresh value with respect to which the thresholding operation is made
    #   - max_BINARY_value: The value used with the Binary thresholding operations (to set the chosen pixels)
    ret, thresh = cv2.threshold(gray_image, 127, 255, 0)

    return thresh

def get_centroid(binImg):

    # calculate moments of binary image
    M = cv2.moments(binImg)

    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return cX, cY


def generate_wedge_mask(img, cX, cY, angle):

    # Calculate Angle. Creating a right triangle
    op_side = math.tan(math.radians(angle)) * cX
    c = [cX, cX * 2, cX * 2]
    r = [cY, cY, int(cY - op_side)]

    # Generate map of three points that compose the right triangle
    rc = np.array((c, r)).T

    # Create a mask
    mask = create_mask(img)

    # Draw contours for wedge
    cv2.drawContours(mask, [rc], 0, 255, -1)

    return mask


def extract_wedge(img, binImg, angle):
    """Extract a wedge of a given angle from the input image"""

    cX, cY = get_centroid(binImg)

    wedge_mask = generate_wedge_mask(binImg, cX, cY, angle)

    # Extract Wedge from main image
    wedge = np.zeros_like(img)
    wedge[wedge_mask == 255] = img[wedge_mask == 255]

    # At this point we have extracted the wedge from the main image.
    # Now we proceed to isolate the area that contains only the wedge

    # Find contours for the wedge
    img_dilation = cv2.dilate(wedge, None, iterations=7)
    img_erosion = cv2.erode(img_dilation, None, iterations=7)
    ret, thresh = cv2.threshold(img_erosion, 125, 255, cv2.THRESH_BINARY)
    edged = cv2.Canny(thresh, 250, 500)
    cv2.imshow('edged image', edged)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(wedge,contours,-1,(0,255,0),3)
    # show_image(wedge)

    # Find the contour with the best area that covers the wedge
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]

    # ret, thresh = cv2.threshold(img_erosion, 125, 255, cv2.THRESH_BINARY)
    ret, thresh = cv2.threshold(img_erosion, 125, 255, cv2.THRESH_BINARY_INV)
    x, y, w, h = cv2.boundingRect(cnt)
    wedge2=wedge.copy()
    rect = cv2.rectangle(wedge2, (x, y), (x+w,y+h), (0,255,0), 2)
    show_image(wedge2)

    return wedge[y:y + h, x:x + w], thresh[y:y + h, x:x + w]


def translate_array(data):
    """Translate an image array into a simple binary array of same size of the given image."""

    nrows = data.shape[0]
    ncols = data.shape[1]
    a = np.zeros(data.shape[:2], dtype='uint8')

    b = data[1:, 1:, 1]  # Extract a channel from the input image(RGB will be the same since image is binarized)
    out = np.where(b == 255, 1, 0)
    return out

    # for row in range(nrows):
    #     for col in range(ncols):
    #         if data[row][col][0] == 255:
    #             a[row][col] = 1
    # return a


def find_max_ins_rect(img):
    """Find the largest inscribed rectangle of a given image map"""

    data = translate_array(img)
    nrows, ncols = data.shape
    w = np.zeros(dtype=int, shape=data.shape)
    h = np.zeros(dtype=int, shape=data.shape)
    skip = 1
    area_max = (0, [])

    for r in range(nrows):
        for c in range(ncols):
            if data[r][c] == skip:
                continue
            if r == 0:
                h[r][c] = 1
            else:
                h[r][c] = h[r - 1][c] + 1
            if c == 0:
                w[r][c] = 1
            else:
                w[r][c] = w[r][c - 1] + 1
            minw = w[r][c]
            for dh in range(h[r][c]):
                minw = min(minw, w[r - dh][c])
                area = (dh + 1) * minw
                if area > area_max[0]:
                    area_max = (area, [c - minw + 1, r - dh, c, r])

    rect_coor = area_max[1]

    # # For testing
    # x = rect_coor[0]
    # y = rect_coor[1]
    # w = rect_coor[2]
    # h = rect_coor[3]
    # img2 = img.copy()
    # cv2.rectangle(img2, (x, y), (w, h), (0, 255, 0), 2)
    # show_image(img2)

    return rect_coor


def rotate_bound(image, angle):
    """INSERT DESCRIPTION HERE"""

    # grab the dimensions of the image and then determine the center
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


def extract_rectangle(img, img_mask):
    """INSERT DESCRIPTION HERE"""

    # Get coordinates of the largest inscribed rectangle a given wedge.
    coord = find_max_ins_rect(img_mask)
    # With coordinates of rectangle obtain the rectangle from the wedge.
    res = img[coord[1]:coord[3], coord[0]:coord[2]]
    show_image(res)

    return rotate_bound(res, 90)


def extract_regions(rect_img, n_rings):
    h, w, depth = rect_img.shape
    newh = int(h/n_rings)
    y = 0
    x = 0
    print(h, newh)
    # cv2.imshow('Image1', rect_img)
    for region in range(n_rings):
        out = rect_img[y:y + newh, x:x + w]
        y = y+newh
        region_name = str(randrange(1, 50000))+".jpg"
        print(region_name)
        # cv2.imshow(region_name, out)
        store_region(out, region_name, "C:\\Users\\Caloj\\Desktop")
    # cv2.waitKey(0)


def store_region(img, img_name, directory_path):
    if not os.path.exists(directory_path+'\\regions'):
        try:
            os.mkdir(directory_path + '\\regions')
        except OSError:
            print("Creation of the directory %s failed" % directory_path)
        else:
            print("Successfully created the directory %s " % directory_path)
    print(os.path.join(directory_path + '\\regions', str(img_name)))
    cv2.imwrite(os.path.join(directory_path + '\\regions', str(img_name)), img)



def show_image(img):
    cv2.imshow('Image', img)
    cv2.waitKey(0)


def main():

    angle = 30
    rings = 3
    image = load_image('control.jpg')

    # TODO: Receive User Inputs and Images from Image PreProcessing

    i = 0
    while (i < 360):
        rot_img = rotate_bound(image, i)
        bin_img = binarize_image(rot_img)
        wedge, wedge_mask = extract_wedge(rot_img, bin_img, angle)
        show_image(wedge_mask)
        rot_rect = extract_rectangle(wedge, wedge_mask)
        extract_regions(rot_rect, rings)
        i = i + angle

    # TODO: Send User Inputs to Fiber Density Module. Signal Fiber Density Module



if __name__== "__main__":
  main()