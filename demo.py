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
    cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.rectangle(image, (int(rect[0] - 1), int(rect[1]) - 16), (int(rect[0] + 115), int(rect[1])), (0, 0, 255), -1, cv2.LINE_AA)
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    draw.text((int(rect[0] + 1), int(rect[1] - 16)), addText, (255, 255, 255), font=fontC)
    imagex = np.array(img)
    return imagex


import HyperLPRLite as pr
import cv2
import numpy as np


from typing import List, Iterator
from collections import namedtuple
class Tractor:
    class Plate:
        def __init__(self, plateStr: str, confidence: float, left: float, right: float, top: float, bottom: float, width: float, height: float, startTime: int, endTime: int):
            self.plateStr, self.confidence, self.left, self.right, self.top, self.bottom, self.width, self.height, self.startTime, self.endTime = \
                plateStr, confidence, left, right, top, bottom, width, height, startTime, endTime

        def __str__(self) -> str:
            return "Plate{str='%s', confidence=%f, left=%f, right=%f, top=%f, bottom=%f, width=%f, height=%f, startTime=%d, endTime=%d}" % (self.plateStr, self.confidence, self.left, self.right, self.top, self.bottom, self.width, self.height, self.startTime, self.endTime)

    def __init__(self, lifeTimeLimit=24):
        self.VehiclePlate = namedtuple('vehicle_plate', 'str confidence left right top bottom width height')
        self._movingPlates: List[Tractor.Plate] = []
        self._deadPlates: List[Tractor.Plate] = []
        self._lifeTimeLimit = lifeTimeLimit

    def _removeDeadPlates(self, time: int) -> None:
        for mvPlate in self._movingPlates:
            if time - mvPlate.endTime > self._lifeTimeLimit:
                self._deadPlates.append(mvPlate)
                self._movingPlates.remove(mvPlate)

    def _getSimilarSavedPlates(self, nowPlateTuple: namedtuple) -> Iterator[Plate]:
        # def pointIn(x: int, y: int, borderLeft: int, borderRight: int, borderTop: int, borderBottom: int) -> bool:
        #     return borderLeft <= x <= borderRight and borderTop <= y <= borderBottom
        def computeIntersect(rectangle1: List[float], rectangle2: List[float]):
            left1, right1, top1, bottom1 = rectangle1
            left2, right2, top2, bottom2 = rectangle2

            left = max(left1, left2)
            right = min(right1, right2)
            top = max(top1, top2)
            bottom = min(bottom1, bottom2)

            if left <= right and top <= bottom:
                return (right - left) * (bottom - top)
            return 0

        for i in range(len(self._movingPlates) - 1, -1, -1):
            savedPlate = self._movingPlates[i]
            editDistance = self.editDistance(savedPlate.plateStr, nowPlateTuple.str)
            if editDistance < 2:
                yield savedPlate
            elif editDistance < 4:
                # border = savedPlate.left, savedPlate.right, savedPlate.top, savedPlate.bottom
                # if pointIn(*(nowPlate.right, nowPlate.top), *border) or pointIn(*(nowPlate.right, nowPlate.bottom), *border) or pointIn(
                #     *(nowPlate.left, nowPlate.top), *border) or pointIn(*(nowPlate.left, nowPlate.bottom), *border):
                rect1 = [savedPlate.left, savedPlate.right, savedPlate.top, savedPlate.bottom]
                rect2 = [nowPlateTuple.left, nowPlateTuple.right, nowPlateTuple.top, nowPlateTuple.bottom]
                if computeIntersect(rect1, rect2) != 0:
                    yield savedPlate
            # elif editDistance < 5:
                # border = savedPlate.left + savedPlate.width // 2, savedPlate.right - savedPlate.width // 2, savedPlate.top + savedPlate.height // 2, savedPlate.bottom - savedPlate.height // 2
                # if pointIn(*(nowPlate.right, nowPlate.top), *border) or pointIn(*(nowPlate.right, nowPlate.bottom), *border) or pointIn(
                #     *(nowPlate.left, nowPlate.top), *border) or pointIn(*(nowPlate.left, nowPlate.bottom), *border):
                #     yield savedPlate

    def analyzePlate(self, nowPlateTuple: namedtuple, time: int) -> (str, float):
        if 'A' <= nowPlateTuple.str[0] <= 'Z' and nowPlateTuple.str[0] != 'S':
            return nowPlateTuple.str, nowPlateTuple.confidence

        similarPlates = list(self._getSimilarSavedPlates(nowPlateTuple))
        if not similarPlates:
            # initPlateList = [nowPlate[attr] for attr in 'str confidence left right top bottom width height'.split()] + [time] * 2
            initPlateList = list(nowPlateTuple) + [time] * 2
            self._movingPlates.append(Tractor.Plate(*initPlateList))
            return nowPlateTuple.str, nowPlateTuple.confidence
        self._removeDeadPlates(time)
        savedPlate = sorted(similarPlates, key=lambda plate: plate.confidence, reverse=True)[0]
        if savedPlate.confidence < nowPlateTuple.confidence:
            savedPlate.plateStr = nowPlateTuple.str
            savedPlate.confidence = nowPlateTuple.confidence
            savedPlate.left = nowPlateTuple.left
            savedPlate.right = nowPlateTuple.right
            savedPlate.top = nowPlateTuple.top
            savedPlate.bottom = nowPlateTuple.bottom
            savedPlate.width = nowPlateTuple.width
            savedPlate.height = nowPlateTuple.height
            savedPlate.endTime = time
            return nowPlateTuple.str, nowPlateTuple.confidence
        else:
            savedPlate.endTime = time
            return savedPlate.plateStr, savedPlate.confidence

    def _purge(self) -> None:
        self._deadPlates = sorted(self._deadPlates, key=lambda plate: plate.startTime)
        self._movingPlates = sorted(self._movingPlates, key=lambda plate: plate.startTime)
        for plate in self._deadPlates:
            if plate.startTime + 2 >= plate.endTime:
                self._deadPlates.remove(plate)
        for plate in self._movingPlates:
            if plate.startTime + 2 >= plate.endTime:
                self._movingPlates.remove(plate)

    def getAll(self) -> List[Plate]:
        self._purge()
        return self._deadPlates + self._movingPlates

    # 下面都是Util
    def getTupleFromList(self, detectionList: List) -> namedtuple:
        x, y, width, height = detectionList.pop(2)
        detectionList += [x, x + width, y, y + height, width, height]
        return self.VehiclePlate(*detectionList)

    def editDistance(self, word1: str, word2: str) -> int:
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
        # print('Edit distance between %s and %s=%d' % (word1, word2, int(distanceMatrix[len(word1)][len(word2)])))
        return int(distanceMatrix[len(word1)][len(word2)])


def detect(originImg: np.ndarray, frameIndex=-1) -> np.ndarray:
    image = None
    for plateStr, confidence, rect in model.SimpleRecognizePlateByE2E(originImg):
        if confidence > 0.8:
            vehiclePlate = tracker.getTupleFromList([plateStr, confidence, rect])
            plateStr, confidence = tracker.analyzePlate(vehiclePlate, frameIndex)
            image = drawRectBox(originImg, rect, plateStr + " " + str(round(confidence, 3)))
            print("plate_str: %s, confidence: %f" % (plateStr, confidence))
        break  # 每帧只处理最有可能的车牌号
    return image if image is not None else originImg


def detectShow(originImg: np.ndarray, frameIndex=-1) -> np.ndarray:
    drawedImg = detect(originImg, frameIndex)
    cv2.imshow("detecting frame", drawedImg)
    cv2.waitKey(1)
    # SpeedTest("images_rec/2_.jpg")
    return drawedImg


def demoPhotos():
    dir = r'E:\PycharmProjects\License_Plate_Detection_Pytorch-master\dataset\raw_pics'
    for file in os.listdir(dir):
        if not file.startswith('2020'):
            continue
        print('<<<<<< ' + file + ' >>>>>>')
        detectShow(ImageUtil.Imread(os.path.join(dir, file)))


def demoVideo(args, showDetection=False):
    inStream = VideoUtil.OpenInputVideo(args.video)
    outStream = VideoUtil.OpenOutputVideo(inStream, args.output)
    frameIndex = 0
    frameLimit = VideoUtil.GetVideoFramesCount(inStream)
    # frameLimit = 1000 if frameLimit > 1000 else frameLimit
    fps: int = VideoUtil.GetFps(inStream)
    global tracker
    tracker = Tractor(fps)
    while True:
        frame = VideoUtil.ReadFrame(inStream)
        if frame.shape[0] == 0 or frameIndex > frameLimit:
            break
        frameDrawed = detectShow(frame, frameIndex) if showDetection else detect(frame, frameIndex)
        VideoUtil.WriteFrame(outStream, frameDrawed)
        frameIndex += 1
        print('\t已处理 %d / %d帧' % (frameIndex, frameLimit))
    if showDetection:
        cv2.destroyAllWindows()
    VideoUtil.CloseVideos(inStream, outStream)
    # 写日志
    import os
    with open(os.path.join(os.path.dirname(args.output), os.path.basename(args.output).split('.')[0]) + '.txt', 'a') as fpLog:
        print('以下是检测到的车牌号：')
        allResult = tracker.getAll()
        for resultPlate in allResult:
            if resultPlate.startTime / fps // 60 < 3:
                line = '%s %.3f [%.2f-%.2f秒]' % (resultPlate.plateStr, resultPlate.confidence, resultPlate.startTime/fps, resultPlate.endTime/fps)
            else:
                seconds1, seconds2 = resultPlate.startTime / fps, resultPlate.endTime / fps
                line = '%s %.3f [%d分%.2f秒 - %d分%.2f秒]' % (resultPlate.plateStr, resultPlate.confidence, seconds1//60, seconds1 % 60, seconds2//60, seconds2 % 60)
            print(line)
            fpLog.write(line + '\n')


if __name__ == '__main__':
    model = pr.LPR("model/cascade.xml", "model/model12.h5", "model/ocr_plate_all_gru.h5")
    tracker = None
    import argparse
    parser = argparse.ArgumentParser(description='车牌识别程序')
    parser.add_argument('-v', '--video', type=str, help='想检测的视频文件名', default=None)
    parser.add_argument('-out', '--output', type=str, help='输出的视频名', default=None)
    demoVideo(parser.parse_args())
