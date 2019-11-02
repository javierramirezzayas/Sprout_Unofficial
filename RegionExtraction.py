import cv2
import numpy as np
import math
import os

n_regions = 0 # for test purposes

# Added This
regions_list = dict()

def resize_image(img, scale):
    """Resize an image"""

    # # Load image
    # img = cv2.imread(img_path)

    # Obtain parameters for scaling
    height, width, depth = img.shape
    # height, width = img.shape
    imgScale = scale / width
    newX, newY = img.shape[1] * imgScale, img.shape[0] * imgScale

    # print(width, height)
    # print(newX, newY)
    # print(newX/imgScale, newY/imgScale)

    # Rescale the image
    return cv2.resize(img, (int(newX), int(newY))), imgScale


def rescale_coordinates(coordinates, old_scale):
    return [int(item/old_scale) for item in coordinates]


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

def get_centroid(img):
    """
    Given an image return the center of the image.
    :param img: Input image
    :return: x and y coordinates of the center of the image (cX and cY)
    """

    # calculate moments of binary image
    M = cv2.moments(img)

    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return cX, cY


def generate_wedge_mask(img, cX, cY, angle):

    # Creating a right triangle with the given angle and coordinates
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

    # Generate the mask for the wedge
    wedge_mask = generate_wedge_mask(binImg, cX, cY, angle)

    # Extract Wedge from main image
    wedge = np.zeros_like(img)
    wedge[wedge_mask == 255] = img[wedge_mask == 255]

    # At this point we have extracted the wedge from the main image.
    # Now we proceed to isolate the area that contains only the wedge

    # Find contours for the wedge
    img_dilation = cv2.dilate(wedge, None, iterations=15)
    img_erosion = cv2.erode(img_dilation, None, iterations=15)
    ret, thresh = cv2.threshold(img_erosion, 125, 255, cv2.THRESH_BINARY)
    edged = cv2.Canny(thresh, 250, 500)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Find the contour with the best area that covers the wedge
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]

    # ret, thresh = cv2.threshold(img_erosion, 125, 255, cv2.THRESH_BINARY_INV)
    x, y, w, h = cv2.boundingRect(cnt)

    # # For testing
    # cv2.drawContours(wedge,contours,-1,(0,255,0),3)
    # show_image(resize_image(wedge))
    # wedge2=wedge.copy()
    # rect = cv2.rectangle(wedge2, (x, y), (x+w,y+h), (0,255,0), 2)
    # show_image(resize_image(wedge2))

    return wedge[y:y + h, x:x + w], thresh[y:y + h, x:x + w]

def find_inscribed_rect(img):
    data = img[1:, 1:, 1]


    return None


def find_max_ins_rect(img):
    """Find the largest inscribed rectangle of a given image map"""

    data = img[1:, 1:, 1]  # Extract a channel from the input image(RGB will be the same since image is binarized)
    nrows, ncols = data.shape
    w = np.zeros(dtype=int, shape=data.shape)
    h = np.zeros(dtype=int, shape=data.shape)
    # skip = 255
    skip = 0
    area_max = (0, [])

    for r in range(nrows):
        for c in range(ncols):
            if data[r][c] == skip:
                continue
            # if r == 0:
            if r == 255:
                h[r][c] = 1
            else:
                h[r][c] = h[r - 1][c] + 1
            # if c == 0:
            if c == 255:
                w[r][c] = 1
            else:
                w[r][c] = w[r][c - 1] + 1
            minw = w[r][c]
            for dh in range(h[r][c]):
                minw = min(minw, w[r - dh][c])
                area = (dh + 1) * minw
                if area > area_max[0]:
                    area_max = (area, [c - minw + 1, r - dh, c, r])

    # Return coordinates of largest inscribed rectangle. List has the following order [x, y, w, h]
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
    """Given an image, make a rotation including its boundaries"""

    # If input angle is 0, there is no need of rotating the image
    if angle != 0:
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
    else:
        return image


def extract_rectangle(img, img_mask):
    """Given a bounded wedge, extract its largest inscribed rectangle"""
    rmask, rmask_scale = resize_image(img_mask, 100)
    # Get coordinates of the largest inscribed rectangle a given wedge.
    coord = find_max_ins_rect(rmask)
    rcoord = rescale_coordinates(coord, rmask_scale)
    # With coordinates of rectangle obtain the rectangle from the wedge.
    rect = img[rcoord[1]:rcoord[3], rcoord[0]:rcoord[2]]
    # Rotate image 90 degrees
    res = rotate_bound(rect, 90)

    return res


def extract_regions(rect_img, n_rings, wedge_num, dir_path, img_type):
    """Given a image containing a wedge rectangle, divide that image into regions"""
    h, w, depth = rect_img.shape
    newh = int(h/n_rings)
    y = h
    x = 0
    regions_path = ""
    for region in range(n_rings):
        out = rect_img[y - newh:y, x:x + w]
        y = y-newh
        region_name = "R"+str(region+1)+"W"+str(wedge_num)+img_type
        regions_path = store_region(out, region_name, dir_path)
        append_regions_dict(region_name, out)
        increment_n_regions()
    return regions_path


def store_region(img, img_name, directory_path):

    regions_path = directory_path+'\\regions'

    # Check if regions_path exists, if not then create the path
    if not os.path.exists(regions_path):
        try:
            os.mkdir(directory_path + '\\regions')
        except OSError:
            print("Creation of the directory %s failed" % directory_path)
        else:
            print("Successfully created the directory %s " % directory_path)

    # Store region image
    try:
        cv2.imwrite(os.path.join(regions_path, str(img_name)), img)
    except OSError:
        print("Storage of %s failed on path" % img_name % regions_path)
    else:
        print("Stored: ", os.path.join(regions_path, str(img_name)))
        return regions_path


def increment_n_regions():
    global n_regions
    n_regions = n_regions+1


# ADDED THIS
def append_regions_dict(regionname, region):
    global regions_list
    regions_list.update({regionname: region})


def show_image(img):
    """Show on the screen a given image (For debug purposes)"""
    cv2.imshow('Image', img)
    cv2.waitKey(0)


def region_extraction(bounded_input_image, bounded_binarized_input_image, number_wedges, number_rings, intermediate_path):

    image_type = ".png"

    num_of_regions = number_wedges * number_rings
    print("Number of Expected Regions: ", num_of_regions)

    # Calculate the angle of the wedge given the number of wedges per quadrant
    wedge_angle = (lambda wedges: 360/wedges)(number_wedges)

    # Resize given images
    # bounded_input_image = resize_image(rgb_bimage)
    # bin_image = resize_image(bin_bimage, 1200)
    bin_image = bounded_binarized_input_image

    i = 0
    regions_path = ""
    wedge_number = 1

    while i < 360:
        rot_img = rotate_bound(bin_image, i)
        bin_img = binarize_image(rot_img)
        wedge, wedge_mask = extract_wedge(rot_img, bin_img, wedge_angle)
        # store_region(wedge, "Wedge"+str(wedge_number)+".png", dir_path)
        # store_region(wedge_mask, "WedgeMask" + str(wedge_number) + ".png", dir_path)
        # show_image(resize_image(wedge))
        rot_rect = extract_rectangle(wedge, wedge_mask)
        # show_image(rot_rect)
        regions_path = extract_regions(rot_rect, num_of_rings, wedge_number, intermediate_path, image_type)
        i = i + wedge_angle
        wedge_number = wedge_number + 1

    print("Total Number of Regions: ", n_regions)

    return regions_path


if __name__ == "__main__":
    num_wedges_p_q = 12
    num_of_rings = 3
    dir_path = "C:\\Users\\Caloj\\Desktop\\Sprout_Images"
    image_type = ".png"
    rgb_image = cv2.imread('control_rgb.jpg')
    bin_image = cv2.imread('control.png')

    Regions_Path = region_extraction(rgb_image, bin_image, num_wedges_p_q, num_of_rings, dir_path)

    # Call Fiber Density Module

    print("Stored Regions at: " + Regions_Path)
