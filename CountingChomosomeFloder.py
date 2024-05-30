import os
import numpy as np
from PIL import Image
import cv2
from skimage.morphology import skeletonize
import math

# Set the input and output folders
input_folder = "E:/A_Old_D/vj/emb/Overlap/emb"
output_folder = "output_folder_Pim_emb_test_overlap"
os.makedirs(output_folder, exist_ok=True)

def gammaCorrection(img):
    gamma = 22
    binary_img = img / 255.0
    gammaCor = cv2.pow(binary_img, gamma)
    img_gammaCor = np.uint8(gammaCor * 255)
    return img_gammaCor


def otsuThreshold(img):
    ret, threshold = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return threshold


def flood_fill(img):
    imgCopy = img.copy()
    h, w = imgCopy.shape
    mask = np.zeros((h+2, w+2), np.uint8)
    _, flood_fill_img, _, _ = cv2.floodFill(imgCopy, mask, (0, 0), 255, cv2.FLOODFILL_FIXED_RANGE)
    flood_fill_imgInv = cv2.bitwise_not(flood_fill_img)
    img_flood_fill = img.astype(np.int_) | flood_fill_imgInv.astype(np.int_)
    img_flood_fill = np.uint8(img_flood_fill)
    return img_flood_fill


def dilation(img):
    kernel = np.ones((3, 3), np.uint8)
    img_dilation = cv2.dilate(img, kernel)
    return img_dilation


def erosion(img):
    kernel = np.ones((5, 5), np.uint8)
    img_erosion = cv2.erode(img, kernel)
    return img_erosion


def customize_skeleton(img):
    y, x = img.shape
    for i in range(0, y):
        n = np.sum(img[i] == 255)
        if n > 0:
            for j in range(0, x):
                if img[i][j] == 255 and img[i][j + 1] == 0:
                    if (img[i-1][j+1] == 255 or img[i+1][j+1] == 255) and img[i][j+2] == 255:
                        img[i][j+1] = 255
                        if img[i-2][j] != 255 and img[i-2][j+1] != 255 and img[i-2][j+2] != 255:
                            img[i-1][j+1] = 0
                        if img[i+2][j] != 255 and img[i+2][j+1] != 255 and img[i+2][j+2] != 255:
                            img[i+1][j+1] = 0
                        if img[i][j+3] != 255 and img[i-1][j+3] != 255 and img[i+1][j+3] != 255 and img[i-1][j+2] != 255 and img[i+1][j+2] != 255:
                            img[i][j+2] = 0
                        if img[i][j-1] != 255 and img[i-1][j-1] != 255 and img[i+1][j-1] != 255 and img[i-1][j] != 255 and img[i+1][j] != 255:
                            img[i][j] = 0
                if img[i][j] == 255 and img[i + 1][j] == 0:
                    if (img[i+1][j+1] == 255 or img[i+1][j-1] == 255) and img[i+2][j] == 255:
                        img[i+1][j] = 255
                        if img[i][j-2] != 255 and img[i+1][j-2] != 255 and img[i+2][j-2] != 255:
                            img[i + 1][j - 1] = 0
                        if img[i][j+2] != 255 and img[i+1][j+2] != 255 and img[i+2][j+2] != 255:
                            img[i + 1][j + 1] = 0
                        if img[i-1][j] != 255 and img[i-1][j-1] != 255 and img[i-1][j+1] != 255 and img[i][j-1] != 255 and img[i][j+1] != 255:
                            img[i][j] = 0
                        if img[i+3][j] != 255 and img[i+3][j-1] != 255 and img[i+3][j+1] != 255 and img[i+2][j-1] != 255and img[i+2][j+1] != 255:
                            img[i+2][j] = 0
    return img


def skelaton(img):
    skel_out = skeletonize(img, method='lee')
    skel_out_customize = customize_skeleton(skel_out)
    return skel_out_customize


def distance_points(cInP, cEnd, numE):
    # Convert tuple to list if cInP is originally a tuple
    cInP = list(cInP)

    if len(cInP) == 0:
        if len(cEnd) == 3:
            numE = 2
        elif len(cEnd) == 4:
            numE = 4
    else:
        arr_distance = []
        distance = 0
        for i in range(len(cInP)):
            x1, y1 = cInP[i][0][0][0], cInP[i][0][0][1]
            for j in range(len(cEnd)):
                x2, y2 = cEnd[j][0][0][0], cEnd[j][0][0][1]
                d = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                arr_distance.append(d)
                if d > distance:
                    distance = d

        if len(cInP) == 2:
            dist_x1, dist_y1 = cInP[0][0][0][0], cInP[0][0][0][1]
            dist_x2, dist_y2 = cInP[1][0][0][0], cInP[1][0][0][1]
            d = math.sqrt((dist_x1 - dist_x2) ** 2 + (dist_y1 - dist_y2) ** 2)
            count_dist = sum(1 for dist in arr_distance if dist < (d / 2))
            if count_dist >= 3:
                numE = 2
                cInP.pop(0)
                cInP.pop(0)
        else:
            arr_distance.remove(distance)
            count_dist = sum(1 for dist in arr_distance if dist < distance / 2)
            if count_dist == 2:
                numE = 2
                cInP.pop(0)

    return numE, cInP



def skeleton_endpoints(img):
    ret, skel = cv2.threshold(img, 0, 1, 0)
    skel = np.uint8(skel)
    kernel = np.uint8([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]])
    src_depth = -1
    filtered = cv2.filter2D(skel, src_depth, kernel)
    out = np.zeros_like(skel)
    out[np.where(filtered == 11)] = 255
    contoursEnd, _ = cv2.findContours(
        out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n = np.sum(out == 255)
    skel[np.where(skel == 1)] = 255
    skel_rgb = cv2.cvtColor(skel, cv2.COLOR_GRAY2RGB)
    # make end point to red color
    for i in range(0, len(contoursEnd)):
        cv2.drawContours(skel_rgb, contoursEnd, i, (0, 0, 255), 2)
    return contoursEnd, n, skel_rgb


def intersection_points(input_img, img_end):
    skel = input_img.copy()
    array = []
    branch1 = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1],
    [1, 1, 1, 1, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 0, 1, 0]
])

    branch2 = np.array([
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0],
        [1, 1, 1, 1, 0],
        [0, 0, 0, 1, 0]
    ])

    branch3 = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 1, 0, 0, 0]
    ])

    branch4 = np.array([
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1]
    ])

    branch5 = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])

    branch6 = np.array([
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0],
        [1, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0]
    ])

    branch7 = np.array([
        [0, 0, 0, 0, 0],
        [1, 1, 0, 0, 1],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0]
    ])

    branch8 = np.array([
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0]
    ])

    branch9 = np.array([
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 1, 0, 0, 1],
        [1, 0, 0, 0, 0]
    ])

    branch10 = np.array([
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0],
        [1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1]
    ])

    branch11 = np.array([
        [1, 0, 0, 0, 1],
        [0, 1, 0, 0, 1],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0]
    ])

    branch12 = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0]
    ])

    branch13 = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 0, 0]
    ])

    branch14 = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 0, 0]
    ])

    branch15 = np.array([
        [0, 0, 0, 0, 1],
        [1, 0, 0, 1, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0]
    ])

    branch16 = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0]
    ])

    branch17 = np.array([
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [1, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0]
    ])


    branch19 = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [1, 1, 0, 0, 1]
    ])

    branch20 = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0]
    ])

    branch21 = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 1, 1],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])

    branch22 = np.array([
        [0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0]
    ])


    branch24 = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1]
    ])

    array.append(
        [branch1, branch2, branch3, branch4, branch5, branch6, branch7, branch8, branch9, branch10, branch11, branch12, branch13, branch14, branch15, branch16, branch17,  branch19, branch20, branch21, branch22,branch24 ])
    hm = np.full(skel.shape, 0)
    for j in range(len(array[0])):
        kernel = array[0][j]
        for k in range(4):  # Rotate kernel 0, 90, 180, 270 degrees
            hm += cv2.morphologyEx(skel, cv2.MORPH_HITMISS, np.rot90(kernel, k))
    hm = np.uint8(hm)
    contours, _ = cv2.findContours(hm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Highlight intersections in the output image
    for contour in contours:
        cv2.drawContours(img_end, [contour], -1, (0, 0, 255), 2)
        
    return contours, img_end


def processImage(img_path):
    img_original = cv2.imread(img_path, 0)
    if img_original is None:
        return None
    img = cv2.copyMakeBorder(img_original, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    img_gam = gammaCorrection(img)
    img_otsu = otsuThreshold(img_gam)
    img_ff = flood_fill(img_otsu)
    img_di = dilation(img_ff)
    img_ff2 = flood_fill(img_di)
    img_ero = erosion(img_ff2)
    sk = skelaton(img_ero)
    cEnd, numE, skel_rgb = skeleton_endpoints(sk)
    cInt, img_interesting = intersection_points(sk, skel_rgb)
    newEnd, new_cInt = distance_points(cInt, cEnd, numE)
    new_numE = math.ceil(newEnd / 2)
    img_show = Image.fromarray(img_interesting)
    img_show.save(os.path.join(output_folder, os.path.basename(img_path)))
    print(f"Processed image: {os.path.basename(img_path)}, Number of chromosomes: {new_numE}")
    return new_numE

def main():
    count_one_chromosome = 0
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            numE = processImage(img_path)
            if numE == 1:
                count_one_chromosome += 1
    print(f"Total images with exactly one chromosome: {count_one_chromosome}")

if __name__ == "__main__":
    main()