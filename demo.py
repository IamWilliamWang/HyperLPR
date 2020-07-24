import os
import time
from PIL import Image, ImageChops, ImageDraw, ImageFont


def drawRectBox(image, rect, addText=None, rect_color=(0, 0, 255), text_color=(255, 255, 255)):
    """
    在image上画一个带文字的方框
    :param image: 原先的ndarray
    :param rect: [x, y, width, height]
    :param addText: 要加的文字
    :return: 画好的图像
    """
    cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), rect_color, 2,
                  cv2.LINE_AA)
    img = Image.fromarray(image)
    if addText:
        cv2.rectangle(image, (int(rect[0] - 1), int(rect[1]) - 16), (int(rect[0] + 115), int(rect[1])), rect_color, -1,
                      cv2.LINE_AA)
        img = Image.fromarray(image)
        draw = ImageDraw.Draw(img)
        draw.text((int(rect[0] + 1), int(rect[1] - 16)), addText, text_color, font=fontC)
    imagex = np.array(img)
    return imagex


from typing import List, Tuple, Iterator
import cv2
import core as pr
import gc
from matplotlib import pyplot as plt
import numpy as np
from queue import Queue
import re
import threading
import traceback
from sys import getsizeof as sizeof
from Util import ImageUtil, VideoUtil, ffmpegUtil, Serialization


class ReaderThread:
    def __init__(self, inStream: cv2.VideoCapture, rtsp: str):
        self._inStream: cv2.VideoCapture = inStream
        self._rtspAddress: str = rtsp
        self._queue: Queue = Queue()
        self._qsize = 0
        self._thread = threading.Thread(target=self._readFrame, name='read frame thread')
        self.alive = False

    def _readFrame(self):
        self.alive = True
        while self.alive:
            # 尝试读取一帧
            try:
                frame = VideoUtil.ReadFrame(self._inStream)
                if frame.shape[0] == 0:
                    if 'rtsp' in self._rtspAddress:  # 是rtsp的话就重新建立连接
                        VideoUtil.CloseVideos(self._inStream)
                        time.sleep(1)
                        self._inStream = VideoUtil.OpenInputVideo(self._rtspAddress)
                        print('Readed rtsp failed! Reseting input stream...')
                        continue
                    else:  # 是视频的话就退出
                        break
            except:  # cv::OutOfMemoryError
                traceback.print_exc()
                self.alive = False
                return
            self._queue.put(frame)
            self._qsize += 1
        self.alive = False

    def start(self):
        if self.alive:
            raise RuntimeError("Reading thread is busy now, please call stop the thread first!")
        self._thread.start()
        # self._inStream = VideoUtil.OpenInputVideo(self._rtspAddress)

    def stop(self):
        self.alive = False

    def get(self, timeout=30):
        self._qsize -= 1
        return self._queue.get(timeout=timeout)
        # return VideoUtil.ReadFrame(self._inStream)

    def qsize(self):
        return self._qsize


class Tractor:
    """
    简易追踪器。负责优化和合并检测结果
    """

    class Plate:
        """
        数据类，作为储存车牌的基础单元
        """

        def __init__(self, plateStr: str, confidence: float, left: float, right: float, top: float, bottom: float,
                     width: float, height: float, startTime: int, endTime: int):
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
                   (self.plateStr, self.confidence, self.left, self.right, self.top, self.bottom, self.width,
                    self.height, self.startTime, self.endTime)

    def __init__(self, lifeTimeLimit=48):
        """
        初始化追踪器
        :param lifeTimeLimit: 车牌消失多久就算离开屏幕（越大越准确，但是计算越慢）
        """
        # self.VehiclePlate = namedtuple('vehicle_plate', 'str confidence left right top bottom width height')  # 车牌元组
        self._movingPlates: List[Tractor.Plate] = []
        self._deadPlates: List[Tractor.Plate] = []
        self._lifeTimeLimit = lifeTimeLimit  # 每个车牌的寿命时长
        self.multiTracker = CvMultiTracker()

    def VehiclePlate(self, *args) -> dict:
        if len(args) != 8:
            return {}
        return {'plateStr': args[0], 'confidence': args[1], 'left': args[2], 'right': args[3], 'top': args[4],
                'bottom': args[5], 'width': args[6], 'height': args[7]}

    def _killMovingPlates(self, nowTime: int) -> None:
        """
        将超时的车牌从movingPlates里挪到deadPlates
        :param nowTime: 当前的时间
        :return:
        """
        for plate in self._movingPlates:
            if nowTime - plate.endTime > self._lifeTimeLimit:
                # 避免内存溢出，这里删除部分的属性
                # plate.height = plate.width = plate.left = plate.right = plate.top = plate.bottom = None
                self._deadPlates.append(plate)
                self._movingPlates.remove(plate)

    def _getSimilarSavedPlates(self, nowPlateInfo: dict, nowTime: int) -> Iterator[Plate]:
        """
        根据当前的车牌获取movingPlate中相似的车牌
        :param nowPlateInfo: 当前的车牌tuple，类型是self.VehiclePlate
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
            editDistance = self.editDistance(savedPlate.plateStr, nowPlateInfo['plateStr'])
            if editDistance < 4 and nowTime - savedPlate.endTime < self._lifeTimeLimit // 2:  # 编辑距离低于阈值，不比较方框位置
                yield savedPlate
            elif editDistance < 5:  # 编辑距离适中，比较方框的位置有没有重合
                rect1 = [savedPlate.left, savedPlate.right, savedPlate.top, savedPlate.bottom]
                rect2 = [nowPlateInfo['left'], nowPlateInfo['right'], nowPlateInfo['top'], nowPlateInfo['bottom']]
                if computeIntersect(rect1, rect2) != 0:
                    yield savedPlate

    def analyzePlate(self, nowPlateInfo: dict, nowTime: int) -> (str, float):
        """
        根据当前车牌，进行分析。返回最大可能的车牌号和置信度
        :param nowPlateInfo: 当前车牌，类型：self.VehiclePlate
        :param nowTime: 当前时间
        :return: 最大可能的车牌号和置信度
        """

        def getBetterPlate(beAssignedPlate: str, assignPlate: str) -> str:
            """
            禁止高优先级的车牌前缀被赋值成低优先级车牌前缀的赋值函数。用于代替 plate被赋值=plate赋值 语句
            :param beAssignedPlate: 要被赋值的车牌号
            :param assignPlate: 赋值的车牌号
            :return: 真正被赋值成什么车牌号
            """
            specialPrefixes = ['厂内', 'SG', 'XL']  # 特殊车牌一律最大优先级。如果有单次出现特殊车牌后一定要固定住特殊前缀
            prefixes = ['粤', '湘', '豫', '川', '冀', '贵', '苏', '赣', '甘', '陕', '沪', '鲁', '黑', '辽', '皖', '鄂', '浙', '宁', '琼',
                        '闽', '蒙', '渝', '吉', '桂', '京', '新', '云']  # 根据大数据统计得出的车牌出现的频率从高到低（川、豫除外）
            finalPrefixes, finalOthers = '', ''
            # 一共分为四类讨论。特-特、特-非特、非特-特、非特-非特
            if beAssignedPlate[:2] in specialPrefixes:  # 第一个为特殊车牌，则固定第一个的特殊前缀
                finalPrefixes = beAssignedPlate[:2]
                if assignPlate[:2] in specialPrefixes:  # 车牌的其余部分是第二个的剩余部分
                    finalOthers = assignPlate[2:]
                else:
                    finalOthers = assignPlate[1:]
            else:  # 第一个不是特殊车牌
                if assignPlate[:2] in specialPrefixes:  # 第二个是特殊车牌
                    finalPrefixes = assignPlate[:2]
                    finalOthers = assignPlate[2:]
                else:
                    priority1 = len(prefixes) - prefixes.index(beAssignedPlate[0]) if beAssignedPlate[
                                                                                          0] in prefixes else -1
                    priority2 = len(prefixes) - prefixes.index(assignPlate[0]) if assignPlate[0] in prefixes else -1
                    if priority1 <= priority2:
                        finalPrefixes = assignPlate[0]
                        finalOthers = assignPlate[1:]
                    else:
                        finalPrefixes = beAssignedPlate[0]
                        finalOthers = assignPlate[1:]
            # 如果是特殊车牌，经常出现重叠字母的情况
            return finalPrefixes + finalOthers if finalPrefixes[-1] != finalOthers[0] else finalPrefixes + finalOthers[
                                                                                                           1:]

        # 预处理车牌部分：
        # 跳过条件：车牌字符串太短
        if len(nowPlateInfo['plateStr']) < 7:
            return nowPlateInfo['plateStr'], nowPlateInfo['confidence']
        # 跳过条件：以英文字母开头（S和X除外）
        if 'A' <= nowPlateInfo['plateStr'][0] <= 'R' or 'T' <= nowPlateInfo['plateStr'][0] <= 'W' or 'Y' <= nowPlateInfo['plateStr'][
            0] <= 'Z':
            return nowPlateInfo['plateStr'], nowPlateInfo['confidence']
        # 符合特殊车牌条件，修改其车牌号以符合特殊车牌的正常结构
        # regexMatch特殊车牌 = re.match(r'.*([SX厂]).*([GL内])(.+)', nowPlateTuple.str)
        # if regexMatch特殊车牌:
        #     plateStr = ''
        #     for i in range(1, 4):  # 把三个括号里的拿出来拼接就是车牌号
        #         plateStr += regexMatch特殊车牌.group(i)
        #     tmp = list(nowPlateTuple)
        #     tmp[0] = plateStr
        #     nowPlateTuple = self.VehiclePlate(*tmp)
        regexMatchesSpecialPlate = re.match(r'^.+?(2[1234][01]\d{2,}).*$', nowPlateInfo['plateStr'])
        if regexMatchesSpecialPlate:
            # '210', '211', '220', '221', '230', '240'
            nowPlateInfo['plateStr'] = '厂内' + regexMatchesSpecialPlate.group(1)[:5]
        # 开始分析：在储存的里找相似的车牌号
        similarPlates = list(self._getSimilarSavedPlates(nowPlateInfo, nowTime))
        if not similarPlates:  # 找不到相似的车牌号，插入新的
            # nowPlateInfo = list(nowPlateInfo) + [nowTime] * 2  # 初始化列表
            nowPlateInfo.update({'startTime': nowTime, 'endTime': nowTime})
            # （取巧部分）统计显示 95.9% 的概率成立
            if nowPlateInfo['plateStr'][1] == 'F' or nowPlateInfo['plateStr'][2] == 'F':
                nowPlateInfo['plateStr'] = '粤' + nowPlateInfo['plateStr'][nowPlateInfo['plateStr'].find('F'):]
            self._movingPlates.append(Tractor.Plate(**nowPlateInfo))
            return self._movingPlates[-1].plateStr, nowPlateInfo['confidence']
        # 如果有相似的车牌
        self._killMovingPlates(nowTime)  # 将寿命过长的车牌杀掉
        savedPlate = sorted(similarPlates, key=lambda plate: plate.confidence, reverse=True)[0]  # 按照置信度排序，取最高的
        if savedPlate.confidence < nowPlateInfo['confidence']:  # 储存的置信度较低，保存当前的
            # （取巧部分）在高置信度向低置信度进行赋值时。禁止将低频度的前缀赋给高频度的前缀
            savedPlate.plateStr = getBetterPlate(savedPlate.plateStr, nowPlateInfo['plateStr'])
            # 剩余的属性进行赋值，并记录更新endTime
            savedPlate.confidence, savedPlate.left, savedPlate.right, savedPlate.top, savedPlate.bottom, \
            savedPlate.width, savedPlate.height, savedPlate.endTime = \
                nowPlateInfo['confidence'], nowPlateInfo['left'], nowPlateInfo['right'], nowPlateInfo['top'], \
                nowPlateInfo['bottom'], nowPlateInfo['width'], nowPlateInfo['height'], nowTime
            return nowPlateInfo['plateStr'], nowPlateInfo['confidence']
        else:  # 储存的置信度高，只更新endTime
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
            # 合并相邻的相似车牌
            for i in range(len(plateList) - 1, 0, -1):
                this, previous = plateList[i], plateList[i - 1]
                if self.editDistance(this.plateStr,
                                     previous.plateStr) < 4 and this.startTime <= previous.endTime:  # 合并相邻的编辑距离较小的车牌号
                    endTime = max(this.endTime, previous.endTime)
                    if this.confidence > previous.confidence:
                        this.startTime = previous.startTime
                        this.endTime = endTime
                        plateList[i], plateList[i - 1] = plateList[i - 1], plateList[i]
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
        print('整理数据：大小从 %d ' % (len(self._deadPlates) + len(self._movingPlates)), end='')
        self._mergeSamePlates()
        # 删去概率低于90的普通车牌
        for i in range(len(self._deadPlates) - 1, -1, -1):
            if self._deadPlates[i].confidence < 0.9 and '厂内' not in self._deadPlates[i].plateStr:
                del self._deadPlates[i]
        print('到 %d' % (len(self._deadPlates) + len(self._movingPlates)))
        return sorted(self._deadPlates + self._movingPlates, key=lambda plate: plate.startTime)

    # 下面都是Util
    def getInfoDictFromList(self, detectionList: List) -> dict:
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

    def serialization(self, binaryFilename=''):
        # 只保留数据，缩小文件的大小
        if not binaryFilename:
            binaryFilename = time.strftime("%Y%m%d%H%M%S", time.localtime()) + '.tractor'
        binary = Serialization()
        binary.append(self._movingPlates)
        binary.append(self._deadPlates)
        binary.append(self._lifeTimeLimit)
        binary.append(self.multiTracker)
        binary.save(binaryFilename)

    def deserialization(self, binaryFilename: str):
        binary = Serialization()
        binary.load(binaryFilename)
        self._movingPlates = binary.popLoaded()
        self._deadPlates = binary
        self._lifeTimeLimit = binary.popLoaded()
        self.multiTracker = binary.popLoaded()

    def __len__(self):
        return sizeof(self._movingPlates) + sizeof(self._deadPlates) + len(self.multiTracker)


class CvMultiTracker:
    def __init__(self):
        self._trackers: List[cv2.TrackerCSRT] = []
        self._lastNewRects: List[Tuple[float]] = []
        self._lifeTimeLimit: List[int] = []
        self._lifeTimeLimitInit: int = 24

    def isNewRectangle(self, rect: Tuple[float]) -> bool:
        """
        判断rect是否和上一次检测的框几乎重合
        :param rect:
        :return:
        """
        if not rect:
            return True
        for lastRects in self._lastNewRects:
            if np.std(np.array(lastRects) - np.array(rect)) <= 25:
                return False
        return True

    def appendTrackerCSRT(self, initImage: np.ndarray, initBox: List[float]) -> None:
        """
        使用当前的image和初始box来添加新的csrtTracker
        :param initImage:
        :param initBox:
        :return:
        """
        if initImage is None or not initBox or len(initBox) != 4:
            return
        # 扩大一点车牌的范围，四个方向各扩展10%
        initBox[0] -= initBox[2] * 0.1
        initBox[1] -= initBox[3] * 0.1
        initBox[2] *= 1.2
        initBox[3] *= 1.2
        newTracker = cv2.TrackerCSRT_create()
        newTracker.init(initImage, tuple(initBox))
        self._trackers.append(newTracker)
        self._lifeTimeLimit.append(self._lifeTimeLimitInit)

    def update(self, image: np.ndarray, purgeMissedTracker=True) -> List[Tuple[float]]:
        """
        使用当前的image更新追踪器，返回新的Boxes
        :param image:
        :param purgeMissedTracker:
        :return:
        """
        newBoxes = []
        assert len(self._trackers) == len(self._lifeTimeLimit)
        purgeIndexes = []
        for i, trackerCSRT in enumerate(self._trackers):
            success, newBox = trackerCSRT.update(image)
            newBox = tuple(newBox)
            if not success:  # 如果cvTracker追踪失败
                self._lifeTimeLimit[i] -= 1
                if self._lifeTimeLimit[i] == 0:
                    purgeIndexes.append(i)
                continue
            if purgeMissedTracker:
                if not self.isNewRectangle(newBox):  # 如果这个框与之前的框大部分重合
                    self._lifeTimeLimit[i] -= 1
                if self._lifeTimeLimit[i] == 0:
                    purgeIndexes.append(i)
                    continue
            newBoxes.append(newBox)
        self._lastNewRects = newBoxes
        purgeIndexes.sort(reverse=True)
        for purseAt in purgeIndexes:
            self.purgeAt(purseAt)
        return self._lastNewRects

    def purgeAt(self, n: int) -> None:
        """
        删除第n个追踪器
        :param n:
        :return:
        """
        if 0 <= n < len(self._trackers):
            del self._trackers[n]
            del self._lifeTimeLimit[n]

    def reborn(self, n: int) -> None:
        """
        重置第n个追踪器的生命计时器
        :param n:
        :return:
        """
        if 0 <= n < len(self._trackers):
            self._lifeTimeLimit[n] = self._lifeTimeLimitInit

    def workingTrackerCount(self) -> int:
        """
        获得正在工作的追踪器个数
        :return:
        """
        return len(self._trackers)

    def __len__(self):
        return sizeof(self._trackers) + sizeof(self._lastNewRects) + sizeof(self._lifeTimeLimit)


def detect(originImg: np.ndarray, frameIndex=-1) -> np.ndarray:
    """
    检测核心函数（不显示）
    :param originImg:
    :param frameIndex:
    :return:
    """
    image = None
    # 检测，或使用bin文件进行回忆型检测
    if args.load_binary is None:
        resultList = model.SimpleRecognizePlateByE2E(originImg, tracker.multiTracker)
    else:
        resultList = binary.popLoaded()
    if args.save_binary is not None:
        binary.append(resultList)
    for plateStr, confidence, rect in resultList:
        # regexMatch厂内车牌 = re.match(r'^.+?(2[1234][01]\d{2,}).*$', plateStr)
        # if regexMatch厂内车牌:  # 一般厂内车牌置信度不高，强行给开绿灯
        #     plateStr = '厂内' + regexMatch厂内车牌.group(1)
        #     confidence = max(confidence, 0.8500001)
        if confidence > 0.85:
            if args.video:
                vehiclePlate = tracker.getInfoDictFromList([plateStr, confidence, rect])
                plateStr, confidence = tracker.analyzePlate(vehiclePlate, frameIndex)
            if tracker.multiTracker.workingTrackerCount() == 0:
                image = drawRectBox(originImg, rect, plateStr + " " + str(round(confidence, 3)), (0, 0, 255),
                                    (255, 255, 255))
                tracker.multiTracker.reborn(0)
            else:
                image = drawRectBox(originImg, rect, plateStr + " " + str(round(confidence, 3)), (0, 255, 255),
                                    (0, 0, 0))
            print("%s (%.5f)" % (plateStr, confidence))
        break  # 每帧只处理最有可能的车牌号
    return image if image is not None else originImg


def detectShow(originImg: np.ndarray, frameIndex=-1, wait=1) -> np.ndarray:
    """
    检测核心函数（显示），可中断
    :param originImg:
    :param frameIndex:
    :return:
    """
    drawedImg = detect(originImg, frameIndex)
    cv2.imshow("Frame after detection", drawedImg)
    if cv2.waitKey(wait) == 27:
        return np.array([])
    return drawedImg


def demoPhotos():
    # 处理所有照片
    for file in os.listdir(args.img_dir):
        # if not file.startswith('2020'):
        if not file.endswith('jpg'):
            continue
        print('<<<<<< ' + file + ' >>>>>>')
        detectShow(ImageUtil.Imread(os.path.join(args.img_dir, file)), wait=0)


def demoVideo(showDialog=True):
    """
    测试视频
    :param args:
    :param showDialog: 显示输出窗口
    :return:
    """
    global tracker
    # gc_nextTime = time.time() + 60  # 一分钟收集一次
    inStream, readThread = None, None
    placeCaptureStream, noSkipStream, recordingStream = None, None, None
    try:
        inStream = VideoUtil.OpenInputVideo(args.video)
        readThread = ReaderThread(inStream, args.video)
        readThread.start()
        if args.output:  # 有output才会打开这些输出的流
            if 'd' in args.video_write_mode:  # 只写入没被跳过的检测结果帧
                placeCaptureStream = ffmpegUtil.OpenOutputVideo(args.output, VideoUtil.GetFps(inStream) / args.drop)
            if 's' in args.video_write_mode:  # 全程不跳帧，比dynamic多了被跳过的帧
                insertIdx = args.output.rfind('.')
                videoName = args.output[:insertIdx] + '.noskip' + args.output[insertIdx:]
                noSkipStream = ffmpegUtil.OpenOutputVideo(videoName, VideoUtil.GetFps(inStream) / args.drop)
            if 'r' in args.video_write_mode:  # 实时录像，不做任何处理（如果是rtsp就是录像，如果是video就是转码）
                insertIdx = args.output.rfind('.')
                videoName = args.output[:insertIdx] + '.record' + args.output[insertIdx:]
                recordingStream = ffmpegUtil.OpenOutputVideo(videoName, VideoUtil.GetFps(inStream))
        frameIndex = 0
        frameLimit = VideoUtil.GetVideoFramesCount(inStream) if 'rtsp' not in args.video else 2 ** 31 - 1
        fps: int = VideoUtil.GetFps(inStream)
        if args.load_binary:
            binary.load(args.load_binary)
        frameLimit = min(frameLimit, 200000)  # 限制最大帧数，只处理视频前多少帧
        tracker = Tractor(fps * 3)  # 每个车牌两秒的寿命
        lastFrame = None
        while True:
            try:
                frame = readThread.get(args.exitTimeout)
                ffmpegUtil.WriteFrame(recordingStream, frame)
                if args.rtsp and readThread.qsize() > 1500:  # 当队列大于1500帧画面，就扔掉头部的几帧
                    continue
            except:  # 当30秒取不到任何帧
                readThread.stop()
                break
            # 终止读取
            if frameIndex > frameLimit:
                break
            # if time.time() > gc_nextTime:
            #     print('Collected garbages:', gc.collect())
            #     gc_nextTime = time.time() + 600  # 十分钟后再gc
            # 对原始帧的操作
            if showDialog:  # 保证每一帧都imshow过
                cv2.imshow('Raw frame', frame)
                if cv2.waitKey(1) == 27:
                    break
            if args.drop != 1:  # imshow完了再跳过
                if frameIndex % args.drop != 0:
                    frameIndex += 1
                    continue
            # 开始处理原始帧
            height, width, channel = frame.shape
            frame = frame[int(height * 0.3):, int(width * 0.3):]
            if lastFrame is not None:
                oldpil = Image.fromarray(cv2.cvtColor(lastFrame, cv2.COLOR_BGR2RGB))  # PIL图像和cv2图像转化
                nowpil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                diff = ImageChops.difference(oldpil, nowpil)  # PIL图片库函数
                # plt.imshow(diff);plt.show()
                try:
                    std: float = np.std(diff)
                    print('{%.3f}<%d> ' % (std, readThread.qsize()), end='')
                except:
                    std = 10000
                if std < 8.9:
                    frameIndex += 1
                    print('\t已处理 %d / %d帧' % (frameIndex, frameLimit))
                    ffmpegUtil.WriteFrame(noSkipStream, frame)
                    continue
            startTime = time.time()
            lastFrame = frame
            # <<<<< 核心函数 >>>>>
            frameDrawed = detectShow(frame, frameIndex) if showDialog else detect(frame, frameIndex)
            if frameDrawed.shape[0] == 0:
                break
            try:
                ffmpegUtil.WriteFrame(placeCaptureStream, frameDrawed)
                ffmpegUtil.WriteFrame(noSkipStream, frameDrawed)
            except ValueError as e:
                traceback.print_exc()
            frameIndex += 1
            print('\t已处理 %d / %d帧 (用时%f s)' % (frameIndex, frameLimit, time.time() - startTime))
        # 写日志
        if not args.output:
            return
        import os
        with open(os.path.join(os.path.dirname(args.output), os.path.basename(args.output).split('.')[0]) + '.txt',
                  'a') as fpLog:
            print('以下是检测到的车牌号：')
            allResult = tracker.getAll()
            for resultPlate in allResult:
                if resultPlate.startTime / fps // 60 < 2:
                    line = '%s %.3f [%.2f-%.2f秒]' % (
                        resultPlate.plateStr, resultPlate.confidence, resultPlate.startTime / fps,
                        resultPlate.endTime / fps)
                else:
                    seconds1, seconds2 = resultPlate.startTime / fps, resultPlate.endTime / fps
                    line = '%s %.3f [%d分%.2f秒 - %d分%.2f秒]' % (
                        resultPlate.plateStr, resultPlate.confidence, seconds1 // 60, seconds1 % 60, seconds2 // 60,
                        seconds2 % 60)
                print(line)
                fpLog.write(line + '\n')
    except:  # 任何地方报错了不要管
        traceback.print_exc()
        tracker.serialization()
        print('检测结果已被保保存')
    finally:
        if showDialog:
            cv2.destroyAllWindows()
        if args.save_binary is not None:
            binary.save(args.save_binary)
        readThread.stop()
        VideoUtil.CloseVideos(inStream)
        ffmpegUtil.CloseVideos(placeCaptureStream, noSkipStream, recordingStream)


if __name__ == '__main__':
    # 初始化
    # gc.disable()
    fontC = ImageFont.truetype("./Font/platech.ttf", 14, 0)
    model = pr.LPR("model/cascade.xml", "model/model12.h5", "model/ocr_plate_all_gru.h5")
    # model = pr.LPR("model/lpr.caffemodel", "model/model12.h5", "model/ocr_plate_all_gru.h5")
    binary = Serialization()
    tracker = Tractor(1)  # 设为1，意为禁用追踪功能
    import argparse

    parser = argparse.ArgumentParser(description='车牌识别程序')
    parser.add_argument('-dir', '--img_dir', type=str, help='要检测的图片文件夹', default=None)
    parser.add_argument('-v', '--video', type=str, help='想检测的视频文件名', default=None)
    parser.add_argument('-vwm', '--video_write_mode', type=str,
                        help='写入模式设置。(d)ynamic：保留动态帧(视频抓拍)。(s)tatic：保留静态帧。(r)ecord：保留原始录像(仅用于录制rtsp)', default='dr')
    parser.add_argument('-rtsp', '--rtsp', type=str, help='使用rtsp地址的视频流进行检测', default=None)
    parser.add_argument('-out', '--output', type=str, help='输出的视频名', default=None)
    parser.add_argument('-drop', '--drop', type=int, help='每隔多少帧保留一帧', default=1)
    parser.add_argument('-save_bin', '--save_binary', type=str, help='每一帧的检测结果保存为什么文件名', default=None)
    parser.add_argument('-load_bin', '--load_binary', type=str, help='加载每一帧的检测结果，不使用video而是用加载的结果进行测试', default=None)
    args = parser.parse_args()
    # 传参
    # args.video = r"C:\Users\william\Desktop\厂内\Record20200326-厂内.mp4"
    # args.video = r"E:\PycharmProjects\HyperLPR\20200711rtsp_3.record.mp4"
    # args.rtsp = "rtsp://admin:klkj6021@172.19.13.27"
    # args.output = '20200727rtsp.mp4'
    # args.drop = 1
    # args.video_write_mode = 'sdr'
    # 检测到时rtsp则赋值进video
    if args.rtsp:
        args.video = args.rtsp
    args.exitTimeout = 15 if args.rtsp else 1  # 设置不同模式下的超时秒数
    # 开始执行总程序
    globalStartTime = time.time()
    if args.img_dir is None:
        demoVideo()
    else:
        demoPhotos()
    # 统计执行的时长
    globalTimeSeconds = time.time() - globalStartTime
    globalTimeHours = globalTimeSeconds // 3600
    globalTimeMinutes = (globalTimeSeconds - globalTimeHours * 3600) // 60
    globalTimeSeconds = globalTimeSeconds % 60
    globalTime = '%d时%d分%.3f秒' % (
        globalTimeHours, globalTimeMinutes, globalTimeSeconds) if globalTimeHours != 0 else '%d分%.3f秒' % (
        globalTimeMinutes, globalTimeSeconds)
    print('总用时：' + globalTime)
    # python -m cProfile -s cumulative demo.py >> profile.log
