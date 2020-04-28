import sys
# reload(sys)
# sys.setdefaultencoding("utf-8")


import os
import time


def SpeedTest(image_path):
    grr = cv2.imread(image_path)
    model = pr.LPR("model/cascade.xml", "model/model12.h5", "model/ocr_plate_all_gru.h5")
    model.SimpleRecognizePlateByE2E(grr)
    t0 = time.time()
    for x in range(20):
        model.SimpleRecognizePlateByE2E(grr)
    t = (time.time() - t0) / 20.0
    print("Image size :" + str(grr.shape[1]) + "x" + str(grr.shape[0]) + " need " + str(round(t * 1000, 2)) + "ms")


from testVideos import ImageUtil, VideoUtil
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

fontC = ImageFont.truetype("./Font/platech.ttf", 14, 0)


def drawRectBox(image, rect, addText):
    cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), (0, 0, 255), 2,
                  cv2.LINE_AA)
    cv2.rectangle(image, (int(rect[0] - 1), int(rect[1]) - 16), (int(rect[0] + 115), int(rect[1])), (0, 0, 255), -1,
                  cv2.LINE_AA)
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    draw.text((int(rect[0] + 1), int(rect[1] - 16)), addText, (255, 255, 255), font=fontC)
    imagex = np.array(img)
    return imagex


import HyperLPRLite as pr
import cv2
import numpy as np


def editDistance(word1: str, word2: str) -> int:
    rows = len(word1) + 1
    cols = len(word2) + 1
    distanceMatrix = np.zeros((rows, cols), dtype=int)
    for j in range(1, cols):
        distanceMatrix[0][j] = j
    for i in range(1, rows):
        distanceMatrix[i][0] = i
    for i in range(1, rows):
        for j in range(1, cols):
            if word1[i - 1] == word2[j - 1]:
                distanceMatrix[i][j] = distanceMatrix[i - 1][j - 1]
            else:
                distanceMatrix[i][j] = min(1 + distanceMatrix[i - 1][j], 1 + distanceMatrix[i][j - 1],
                                           1 + distanceMatrix[i - 1][j - 1])
    print('Edit distance between %s and %s=%d' % (word1, word2, int(distanceMatrix[len(word1)][len(word2)])))
    return int(distanceMatrix[len(word1)][len(word2)])


def detect(originImg: np.ndarray, frameIndex=-1) -> np.ndarray:
    def analyzePlate(plateStr: str, confidence: float) -> (str, float):
        global vehiclePlates, vehiclePlatesConfidence, vehiclePlatesStartAt, vehiclePlatesEndAt
        if vehiclePlates == [] or editDistance(vehiclePlates[-1], plateStr) > 3:  # new vehicle
            vehiclePlates += [plateStr]
            vehiclePlatesConfidence += [confidence]
            vehiclePlatesStartAt += [frameIndex]  # start time
            return plateStr, confidence
        if confidence > vehiclePlatesConfidence[-1]:  # this plate can be better
            vehiclePlates[-1] = plateStr
            vehiclePlatesConfidence[-1] = confidence
            if len(vehiclePlatesEndAt) <= len(vehiclePlatesStartAt) - 1:
                vehiclePlatesEndAt += [frameIndex]
            else:
                vehiclePlatesEndAt[-1] = frameIndex
            return plateStr, confidence
        else:  # saved plate can be better
            if len(vehiclePlatesEndAt) <= len(vehiclePlatesStartAt) - 1:
                vehiclePlatesEndAt += [frameIndex]
            else:
                vehiclePlatesEndAt[-1] = frameIndex
            return vehiclePlates[-1], vehiclePlatesConfidence[-1]

    image = None
    detections = model.SimpleRecognizePlateByE2E(originImg)
    for plateStr, confidence, rect in sorted(detections, key=lambda detectionList: detectionList[1]):
        if confidence > 0.8:
            plateStr, confidence = analyzePlate(plateStr, confidence)
            image = drawRectBox(originImg, rect, plateStr + " " + str(round(confidence, 3)))
            print("plate_str: %s, confidence: %f" % (plateStr, confidence))
        break  # 每帧只处理最有可能的车牌号
    return image if image is not None else originImg


def detectShow(arg, frameIndex=-1):
    image = ImageUtil.Imread(arg)
    detect(image, frameIndex)
    cv2.imshow("image", image)
    cv2.waitKey(1)
    # SpeedTest("images_rec/2_.jpg")


def demoPhotos():
    dir = r'E:\PycharmProjects\License_Plate_Detection_Pytorch-master\dataset\raw_pics'
    for file in os.listdir(dir):
        if not file.startswith('2020'):
            continue
        print('<<<<<< ' + file + ' >>>>>>')
        detectShow(os.path.join(dir, file))


def demoVideo(showDetection=False):
    inStream = VideoUtil.OpenInputVideo(r"E:\项目\车牌检测\所有录像\Record20200422-1_clip.mp4")
    outStream = VideoUtil.OpenOutputVideo('20200422_2.mp4', inStream)
    frameIndex = 0
    frameLimit = VideoUtil.GetVideoFileFrameCount(inStream)
    frameLimit = 10000 if frameLimit > 10000 else frameLimit
    fps = VideoUtil.GetFps(inStream)
    while True:
        frame = VideoUtil.ReadFrame(inStream)
        if frame == [] or frameIndex > frameLimit:
            break
        frameDrawed = detect(frame, frameIndex)
        if showDetection:
            cv2.imshow('frame', frameDrawed)
            cv2.waitKey(1)
        VideoUtil.WriteFrame(outStream, frameDrawed)
        frameIndex += 1
        print('\t %d / %d' % (frameIndex, frameLimit))
    if showDetection:
        cv2.destroyAllWindows()
    VideoUtil.CloseVideos(inStream, outStream)
    # 写日志
    with open('20200422_2.txt', 'a') as fpLog:
        print('以下是检测到的车牌号：')
        for i in range(len(vehiclePlates)):
            try:
                print(vehiclePlates[i], vehiclePlatesConfidence[i],
                      '(%.2f - %.2f)' % (vehiclePlatesStartAt[i], vehiclePlatesEndAt[i]))
                if vehiclePlatesStartAt[i] != -1:
                    fpLog.write('%s %.3f [%.2f-%.2f秒]\n' % (
                    vehiclePlates[i], vehiclePlatesConfidence[i], vehiclePlatesStartAt[i] / fps,
                    vehiclePlatesEndAt[i] / fps))
                else:
                    fpLog.write('%s %.3f\n' % (vehiclePlates[i], vehiclePlatesConfidence[i]))
            except:  # 如果start和end数组长度不一样，跳出
                print(
                    '写入结果无法继续。当前的变量：vehiclePlates={}\nvehiclePlatesConfidence={}\nvehiclePlatesStartAt={}\nvehiclePlatesEndAt={}'.format(
                        vehiclePlates, vehiclePlatesConfidence, vehiclePlatesStartAt, vehiclePlatesEndAt))
                break


if __name__ == '__main__':
    model = pr.LPR("model/cascade.xml", "model/model12.h5", "model/ocr_plate_all_gru.h5")
    vehiclePlates = []
    vehiclePlatesConfidence = []
    vehiclePlatesStartAt = []
    vehiclePlatesEndAt = []
    demoVideo()
