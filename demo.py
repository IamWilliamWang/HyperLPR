import sys
# reload(sys)
# sys.setdefaultencoding("utf-8")


import os
import time


def SpeedTest(image_path):
    """
    原先demo自带的speedtest
    :param image_path:
    :return:
    """
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
    """
    在image上画一个带文字的方框
    :param image: 原先的ndarray
    :param rect: [x, y, width, height]
    :param addText: 要加的文字
    :return: 画好的图像
    """
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
    """
    基于编辑距离的简易追踪器。负责优化和合并检测结果
    """
    class Plate:
        """
        数据类，作为储存车牌的基础单元
        """
        def __init__(self, plateStr: str, confidence: float, left: float, right: float, top: float, bottom: float, width: float, height: float, startTime: int, endTime: int):
            """
            初始化车牌
            :param plateStr: 车牌号
            :param confidence: 置信度
            :param left: 车牌框的左侧x坐标
            :param right: 车牌框的右侧x坐标
            :param top: 车牌框的上侧y坐标
            :param bottom: 车牌框的下侧y坐标
            :param width: 车牌框的宽度
            :param height: 车牌框的高度
            :param startTime: 车牌框开始出现的时间
            :param endTime: 车牌框完全消失的时间
            """
            self.plateStr, self.confidence, self.left, self.right, self.top, self.bottom, self.width, self.height, self.startTime, self.endTime = \
                plateStr, confidence, left, right, top, bottom, width, height, startTime, endTime

        def __str__(self) -> str:
            return "Plate{str='%s', confidence=%f, left=%f, right=%f, top=%f, bottom=%f, width=%f, height=%f, startTime=%d, endTime=%d}" % \
                   (self.plateStr, self.confidence, self.left, self.right, self.top, self.bottom, self.width, self.height, self.startTime, self.endTime)

    def __init__(self, lifeTimeLimit=24):
        """
        初始化追踪器
        :param lifeTimeLimit: 车牌消失多久就算离开屏幕（越大越准确，但是计算越慢）
        """
        self.VehiclePlate = namedtuple('vehicle_plate', 'str confidence left right top bottom width height')  # 车牌元组
        self._movingPlates: List[Tractor.Plate] = []
        self._deadPlates: List[Tractor.Plate] = []
        self._lifeTimeLimit = lifeTimeLimit  # 每个车牌的寿命时长

    def _killMovingPlates(self, nowTime: int) -> None:
        """
        将超时的车牌从movingPlates里挪到deadPlates
        :param nowTime: 当前的时间
        :return:
        """
        for plate in self._movingPlates:
            if nowTime - plate.endTime > self._lifeTimeLimit:
                self._deadPlates.append(plate)
                self._movingPlates.remove(plate)

    def _getSimilarSavedPlates(self, nowPlateTuple: namedtuple) -> Iterator[Plate]:
        """
        根据当前的车牌获取movingPlate中相似的车牌
        :param nowPlateTuple: 当前的车牌tuple，类型是self.VehiclePlate
        :return: 相似车牌的generator
        """
        def computeIntersect(rectangle1: List[float], rectangle2: List[float]):
            """
            计算两个矩形相交部分的面积
            :param rectangle1:
            :param rectangle2:
            :return:
            """
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
            savedPlate = self._movingPlates[i]  # 保存的车牌
            editDistance = self.editDistance(savedPlate.plateStr, nowPlateTuple.str)
            if editDistance < 4:  # 编辑距离低于阈值，不比较方框位置
                yield savedPlate
            elif editDistance < 5:  # 编辑距离适中，比较方框的位置有没有重合
                rect1 = [savedPlate.left, savedPlate.right, savedPlate.top, savedPlate.bottom]
                rect2 = [nowPlateTuple.left, nowPlateTuple.right, nowPlateTuple.top, nowPlateTuple.bottom]
                if computeIntersect(rect1, rect2) != 0:
                    yield savedPlate

    def analyzePlate(self, nowPlateTuple: namedtuple, nowTime: int) -> (str, float):
        """
        根据当前车牌，进行分析。返回最大可能的车牌号和置信度
        :param nowPlateTuple: 当前车牌，类型：self.VehiclePlate
        :param nowTime: 当前时间
        :return: 最大可能的车牌号和置信度
        """
        import re
        # 预处理车牌部分：
        # 跳过条件：车牌字符串太短
        if len(nowPlateTuple.str) < 7:
            return nowPlateTuple.str, nowPlateTuple.confidence
        # 跳过条件：以英文字母开头（S和X除外）
        if 'A' <= nowPlateTuple.str[0] <= 'R' or 'T' <= nowPlateTuple.str[0] <= 'W' or 'Y' <= nowPlateTuple.str[0] <= 'Z':
            return nowPlateTuple.str, nowPlateTuple.confidence
        # 符合特殊车牌条件，修改其车牌号
        specialPlateReMatch = re.match(r'.*([SX厂]).*([GL内])(.+)', nowPlateTuple.str)
        if specialPlateReMatch:
            plateStr = ''
            for i in range(1, 4):
                plateStr += specialPlateReMatch.group(i)
            tmp = list(nowPlateTuple)
            tmp[0] = plateStr
            nowPlateTuple = self.VehiclePlate(*tmp)

        # 开始分析：在储存的里找相似的车牌号
        similarPlates = list(self._getSimilarSavedPlates(nowPlateTuple))
        if not similarPlates:  # 找不到相似的车牌号，插入新的
            initPlateList = list(nowPlateTuple) + [nowTime] * 2  # 初始化列表
            self._movingPlates.append(Tractor.Plate(*initPlateList))
            return nowPlateTuple.str, nowPlateTuple.confidence
        self._killMovingPlates(nowTime)  # 将寿命过长的车牌杀掉
        savedPlate = sorted(similarPlates, key=lambda plate: plate.confidence, reverse=True)[0]  # 按照置信度排序，取最高的
        if savedPlate.confidence < nowPlateTuple.confidence:  # 储存的置信度较低，保存当前的
            savedPlate.plateStr, savedPlate.confidence, savedPlate.left, savedPlate.right, savedPlate.top,\
                savedPlate.bottom, savedPlate.width, savedPlate.height, savedPlate.endTime = \
                nowPlateTuple.str, nowPlateTuple.confidence, nowPlateTuple.left, nowPlateTuple.right, nowPlateTuple.top,\
                nowPlateTuple.bottom, nowPlateTuple.width, nowPlateTuple.height, nowTime
            return nowPlateTuple.str, nowPlateTuple.confidence
        else:  # 储存的置信度高
            savedPlate.endTime = nowTime
            return savedPlate.plateStr, savedPlate.confidence

    def _mergeSamePlates(self) -> None:
        """
        相同车牌结果合并到一起
        :return:
        """
        def purgeAndMerge(plateList: List[Tractor.Plate]) -> List[Tractor.Plate]:
            if len(plateList) < 2:
                return plateList

            plateList = sorted(plateList, key=lambda plate: plate.startTime)  # 按照出现时间进行排序，相同的车牌会相邻
            # 除掉存在时间极短的车牌
            # for plate in plateList:
            #     if plate.endTime - plate.startTime <= 3:
            #         plateList.remove(plate)
            # 合并相邻的相似车牌
            for i in range(len(plateList) - 1, 0, -1):
                this, previous = plateList[i], plateList[i-1]
                if self.editDistance(this.plateStr, previous.plateStr) < 4:  # 合并相邻的编辑距离较小的车牌号
                    endTime = max(this.endTime, previous.endTime)
                    if this.confidence > previous.confidence:
                        this.startTime = previous.startTime
                        this.endTime = endTime
                        plateList[i], plateList[i-1] = plateList[i-1], plateList[i]
                    else:
                        previous.endTime = endTime
                    del plateList[i]
            return plateList

        self._deadPlates = purgeAndMerge(self._deadPlates)
        self._movingPlates = purgeAndMerge(self._movingPlates)

    def getAll(self) -> List[Plate]:
        """
        后期处理后返回所有的车牌List
        :return:
        """
        self._mergeSamePlates()
        return self._deadPlates + self._movingPlates

    # 下面都是Util
    def getTupleFromList(self, detectionList: List) -> namedtuple:
        """
        将识别出的List转换成self.VehiclePlate类型的Tuple
        :param detectionList:
        :return:
        """
        x, y, width, height = detectionList.pop(2)
        detectionList += [x, x + width, y, y + height, width, height]
        return self.VehiclePlate(*detectionList)

    def editDistance(self, word1: str, word2: str) -> int:
        """
        计算两个字符串的最小编辑距离
        :param word1:
        :param word2:
        :return:
        """
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
        # print('Edit distance from %s to %s = %d' % (word1, word2, int(distanceMatrix[len(word1)][len(word2)])))
        return int(distanceMatrix[len(word1)][len(word2)])


def detect(originImg: np.ndarray, frameIndex=-1) -> np.ndarray:
    """
    检测核心函数（不显示）
    :param originImg:
    :param frameIndex:
    :return:
    """
    image = None
    for plateStr, confidence, rect in model.SimpleRecognizePlateByE2E(originImg):
        if confidence > 0.85:
            vehiclePlate = tracker.getTupleFromList([plateStr, confidence, rect])
            plateStr, confidence = tracker.analyzePlate(vehiclePlate, frameIndex)
            image = drawRectBox(originImg, rect, plateStr + " " + str(round(confidence, 3)))
            print("%s (%.5f)" % (plateStr, confidence))
        break  # 每帧只处理最有可能的车牌号
    return image if image is not None else originImg


def detectShow(originImg: np.ndarray, frameIndex=-1) -> np.ndarray:
    """
    检测核心函数（显示），可中断
    :param originImg:
    :param frameIndex:
    :return:
    """
    drawedImg = detect(originImg, frameIndex)
    cv2.imshow("detecting frame", drawedImg)
    if cv2.waitKey(1) == 27:
        return np.array([])
    return drawedImg


def demoPhotos():
    dir = r'E:\PycharmProjects\License_Plate_Detection_Pytorch-master\dataset\raw_pics'
    for file in os.listdir(dir):
        if not file.startswith('2020'):
            continue
        print('<<<<<< ' + file + ' >>>>>>')
        detectShow(ImageUtil.Imread(os.path.join(dir, file)))


def demoVideo(args, showDetection=True):
    """
    测试视频
    :param args:
    :param showDetection: 显示输出窗口
    :return:
    """
    inStream = VideoUtil.OpenInputVideo(args.video)
    outStream = VideoUtil.OpenOutputVideo(inStream, args.output)
    frameIndex = 0
    frameLimit = VideoUtil.GetVideoFramesCount(inStream)
    frameLimit = 10000 if frameLimit > 10000 else frameLimit
    fps: int = VideoUtil.GetFps(inStream)
    global tracker
    tracker = Tractor(fps * 3)  # 每个车牌两秒的寿命
    while True:
        frame = VideoUtil.ReadFrame(inStream)
        if frame.shape[0] == 0 or frameIndex > frameLimit:
            break
        frameDrawed = detectShow(frame, frameIndex) if showDetection else detect(frame, frameIndex)
        if frameDrawed.shape[0] == 0:
            break
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
            if resultPlate.startTime / fps // 60 < 2:
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
