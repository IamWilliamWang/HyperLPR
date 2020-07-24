import cv2 as cv
import cv2
import numpy as np
import os
from align import *

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw


def drawRectBox(image, rect):
    """
    在image上画一个带文字的方框
    :param image: 原先的ndarray
    :param rect: [x, y, width, height]
    :param addText: 要加的文字
    :return: 画好的图像
    """
    cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), (0, 0, 255), 2,
                  cv2.LINE_AA)
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    imagex = np.array(img)
    return imagex


def detect(im, cvNet):
    # im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    # to_draw = im.copy()
    pixel_means = [0.406, 0.456, 0.485]
    pixel_stds = [0.225, 0.224, 0.229]
    pixel_scale = 255.0
    rows, cols, c = im.shape
    im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
    im = im.astype(np.float32)
    for i in range(3):
        im_tensor[0, i, :, :] = (im[:, :, 2 - i] / pixel_scale - pixel_means[2 - i]) / pixel_stds[2 - i]
    cvNet.setInput(im_tensor)
    # print(im_tensor.shape)
    import time
    cvOut = cvNet.forward()
    for _ in range(0):
        t0 = time.time()
        cvOut = cvNet.forward()
        print(time.time() - t0)
    rectList = []
    for detection in cvOut[0, 0, :, :]:
        score = float(detection[2])
        if score > 0.85:
            left = int(detection[3] * cols)
            top = int(detection[4] * rows)
            right = int(detection[5] * cols)
            bottom = int(detection[6] * rows)
            rectList += [(left, top, right - left, bottom - top)]
            # to_draw= drawRectBox(to_draw,[left,top,right-left,bottom-top])
            # cropped = to_draw[top:bottom, left:right]
            # cv2.imshow("cropped" , cropped)
            # cv2.waitKey(1)
    # cv2.imshow('image' , to_draw)
    # cv2.waitKey(1)
    return rectList


if __name__ == '__main__':
    folderk = r"C:\Users\william\Desktop\SG"
    for filename in os.listdir(folderk):
        path = os.path.join(folderk, filename)
        if filename.lower().endswith(".jpg"):
            image = cv2.imread(path)
            detect(image)
