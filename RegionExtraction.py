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
    imgScale = 700 / width
    # imgScale = 200 / width
    newX, newY = img.shape[1] * imgScale, img.shape[0] * imgScale

    # Rescale the image
    return cv2.resize(img, (int(newX), int(newY)))


def create_mask(img):
    """Create mask out of input image"""
    return np.zeros(img.shape[:2],dtype = 'uint8')


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
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # # For testing
    # show_image(mask)
    # image = img.copy()
    # cv2.drawContours(image, [rc], 0, 255, 2)
    # show_image(image)

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

    # Find contour for wedge
    img_dilation = cv2.dilate(wedge, None, iterations=6)
    img_erosion = cv2.erode(img_dilation, None, iterations=3)
    edged = cv2.Canny(img_erosion, 150, 500)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(wedge,contours,-1,(0,255,0),3)

    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]

    # ret, thresh = cv2.threshold(img_erosion, 125, 255, cv2.THRESH_BINARY)
    ret, thresh = cv2.threshold(img_erosion, 125, 255, cv2.THRESH_BINARY_INV)

    x, y, w, h = cv2.boundingRect(cnt)
    # rect = cv2.rectangle(out, (x, y), (x+w,y+h), (0,255,0), 2)

    return wedge[y:y + h, x:x + w], thresh[y:y + h, x:x + w]


def translate_array(data):
    nrows = data.shape[0]
    ncols = data.shape[1]
    a = create_mask(data)
    for row in range(nrows):
        for col in range(ncols):
            if data[row][col][0] == 255:
                a[row][col] = 1
    return a


def find_max_ins_rect(img):
    """Find the largest inscribed rectangle of a given image map"""
    print(img)
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

    # For testing
    rect_coor = area_max[1]
    x = rect_coor[0]
    y = rect_coor[1]
    w = rect_coor[2]
    h = rect_coor[3]
    cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)

    return rect_coor


def rotate_bound(image, angle):
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
    coord = find_max_ins_rect(img_mask)
    res = img[coord[1]:coord[3], coord[0]:coord[2]]
    # show_image(res)
    return rotate_bound(res, 90)


def extract_regions(rect_img, n_rings):
    h, w, depth = rect_img.shape
    newh = int(h/n_rings)
    y = 0
    x = 0
    print(h, newh)
    for region in range(n_rings):
        out = rect_img[y:y + newh, x:x + w]
        y = y+newh
        region_name = str(randrange(1, 50000))+".jpg"
        print(region_name)
        store_region(out, region_name, "C:\\Users\\Caloj\\Desktop")


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

    # image = load_image('filled_wedge.jpg')
    # ret, thresh = cv2.threshold(image, 125, 255, cv2.THRESH_BINARY_INV)
    # rot_rect = extract_rectangle(thresh)
    # show_image(rot_rect)

    angle = 30
    rings = 3
    image = load_image('control.jpg')
    image_2 = image.copy()

    i = 0
    while (i < 360):
        # show_image(image_2)
        bin_img = binarize_image(image_2)
        wedge, wedge_mask = extract_wedge(image_2, bin_img, angle)
        # show_image(wedge)
        # show_image(wedge_mask)
        rot_rect = extract_rectangle(wedge, wedge_mask)
        extract_regions(rot_rect, rings)
        # show_image(rot_rect)
        i = i + angle


if __name__== "__main__":
  main()