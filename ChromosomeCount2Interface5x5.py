import tkinter
from tkinter import *
import numpy as np
from PIL import Image, ImageTk
import PIL.Image
from tkinter import filedialog
import cv2
from skimage.morphology import skeletonize
import math

master = tkinter.Tk()
master.title("Counting Chromosome")
master.resizable(1, 1)
master.geometry("1000x800+0+0".format(master.winfo_screenwidth(), master.winfo_screenheight() - 20))
labelName = tkinter.Label(master, text="Counting Chromosome", font=('times', 20, 'bold')).pack(pady=5)

pathName = ""
name_img = ""


def openFile():
    global pathName, name_img
    file_types = [('Image files', '*.jpg')]
    pathName = filedialog.askopenfilename(parent=master, title='Choose a file', filetypes=file_types)
    if pathName != "":
        img_original = PIL.Image.open(pathName)
        path_split = pathName.split("/")
        nameJPG = path_split[len(path_split) - 1]
        name_img = nameJPG.split(".")[0]
        label.config(text=" ")
        showImage(img_original)


def showImage(img_original):
    img_show = img_original.resize((500, 500),PIL.Image.LANCZOS)
    master.imgTk = ImageTk.PhotoImage(img_show)
    canvas.delete(all)
    canvas.create_image(500 / 2, 500 / 2, image=master.imgTk)


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


def distance_points(cInP, cEnP, numE):
    if len(cInP) == 0:
        if len(cEnP) == 3:
            numE = 2
        elif len(cEnP) == 4:
            numE = 4
    else:
        arr_distance = []
        distance = 0
        for i in range(0, len(cInP)):
            x1 = cInP[i][0][0][0]
            y1 = cInP[i][0][0][1]
            for j in range(0, len(cEnP)):
                x2 = cEnP[j][0][0][0]
                y2 = cEnP[j][0][0][1]
                d = math.sqrt((math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2)))
                arr_distance.append(d)
                if d > distance:
                    distance = d
        count_dist = 0
        if len(cInP) == 2:
            dist_x1 = cInP[0][0][0][0]
            dist_y1 = cInP[0][0][0][1]
            dist_x2 = cInP[1][0][0][0]
            dist_y2 = cInP[1][0][0][1]
            d = math.sqrt((math.pow(dist_x1 - dist_x2, 2) + math.pow(dist_y1 - dist_y2, 2)))
            for i in range(0, len(arr_distance)):
                if arr_distance[i] < (d / 2):
                    count_dist = count_dist + 1
            if count_dist == 4 or count_dist == 3:
                numE = 2
                cInP.remove(cInP[0])
                cInP.remove(cInP[0])
        else:
            arr_distance.remove(distance)
            for j in range(0, len(arr_distance)):
                if arr_distance[j] < distance / 2:
                    count_dist = count_dist + 1
            if count_dist == 2:
                numE = 2
                cInP.remove(cInP[0])
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
    contoursEnd, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n = np.sum(out == 255)
    skel[np.where(skel == 1)] = 255
    skel_rgb = cv2.cvtColor(skel, cv2.COLOR_GRAY2RGB)
    skel_rgb[np.where(filtered == 11)] = [0, 0, 255]
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
    for j in range(0, len(array[0])):
        kernel = array[0][j]
        hm1 = cv2.morphologyEx(skel, cv2.MORPH_HITMISS, kernel)
        hm2 = cv2.morphologyEx(skel, cv2.MORPH_HITMISS, np.rot90(kernel, 1))
        hm3 = cv2.morphologyEx(skel, cv2.MORPH_HITMISS, np.rot90(kernel, 2))
        hm4 = cv2.morphologyEx(skel, cv2.MORPH_HITMISS, np.rot90(kernel, 3))
        hm = hm + hm1 + hm2 + hm3 + hm4
    hm = np.uint8(hm)
    contours, _ = cv2.findContours(hm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        for i in range(0, len(contours)):
            if len(contours[i]) > 1:
                contours[i] = contours[i][[0]]
    for i in range(0, len(contours)):
        img_end[contours[i][0][0][1], contours[i][0][0][0]] = [0, 0, 255]
    return contours, img_end


def count():
    img_original = cv2.imread(pathName, 0)
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
    label.config(text=int(new_numE))
    img_show = PIL.Image.fromarray(img_interesting)
    showImage(img_show)


Frame2 = tkinter.Frame(master)
Frame2.pack(side="left", anchor=tkinter.W, padx=25)
canvas = tkinter.Canvas(Frame2, height=450, width=450, bd=5, bg='white', relief=tkinter.RIDGE)
canvas.pack(pady="20")
bt10 = tkinter.Button(Frame2, text="Upload", font='15', height=3, width=15, bg="#FFCCFF", command=openFile).pack(
    side="left", padx="50")
bt11 = tkinter.Button(Frame2, text="Counting", font='15', height=3, width=15, bg="#FFCCFF", command=count).pack(
    side="right", padx="50")

Frame3 = tkinter.Frame(master)
Frame3.pack(side="left", anchor=tkinter.W)
label_frame = tkinter.LabelFrame(Frame3, text="Total number of\n single chromosomes ", font=('Courier', 14, 'bold'))
label_frame.pack(fill="both", expand="yes")
label = tkinter.Label(label_frame, text="", font=('times', 20, 'bold'), width=5, height=3)
label.pack()


tkinter.mainloop()
