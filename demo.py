import argparse
import os
import sys
import time
from dataclasses import dataclass

from PIL import Image, ImageChops, ImageDraw, ImageFont
from typing import List, Tuple, Iterator
import cv2
from PyQt5.QtCore import pyqtSignal, QThread, QObject
from PyQt5.QtWidgets import QApplication

import core as pr
import numpy as np
from queue import Queue
import re
import threading
import traceback
from sys import getsizeof as sizeof

from Util import ImageUtil, VideoUtil, ffmpegUtil, Serialization, editDistance, sec2str, Plate, getArgumentParser


class ReaderThread:
    def __init__(self, inStream: cv2.VideoCapture, rtsp: str):
        self._inStream: cv2.VideoCapture = inStream
        self._rtspAddress: str = rtsp
        self._queue: Queue = Queue()
        self._qsize: int = 0
        self._thread = threading.Thread(target=self._readFrame, name='read frame thread')
        self._itemMemory: int = 0  # 队列中每个元素占多少字节
        self.alive = False  # 线程的生死

    def _readFrame(self):
        self.alive = True
        while self.alive:
            while len(self) > args.memory_limit:  # 当超过内存限制时，暂停读取
                time.sleep(1)
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
            if self._itemMemory == 0:  # 记录每个元素的占用空间
                self._itemMemory = sizeof(frame)
            self._queue.put(frame)
            self._qsize += 1
        self.alive = False

    def start(self):
        if self.alive:
            raise RuntimeError("Reading thread is busy now, please call stop the thread first!")
        self._thread.start()

    def stop(self):
        self.alive = False

    def get(self, timeout=30):
        self._qsize -= 1
        return self._queue.get(timeout=timeout)

    def qsize(self):
        return self._qsize

    def __len__(self):
        """
        输出现有队列的内存大小(Bytes)
        :return:
        """
        return max(0, self.qsize() * self._itemMemory)


class Tractor:
    """
    简易追踪器。负责优化和合并检测结果
    """

    def __init__(self, lifeTimeLimit=48):
        """
        初始化追踪器
        :param lifeTimeLimit: 车牌消失多久就算离开屏幕（越大越准确，但是计算越慢）
        """
        # self.VehiclePlate = namedtuple('vehicle_plate', 'str confidence left right top bottom width height')  # 车牌元组
        self._movingPlates: List[Plate] = []
        self._deadPlates: List[Plate] = []
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
        import copy
        killed = False
        for plate in self._movingPlates:
            if nowTime - plate.endTime > self._lifeTimeLimit:
                if plate.confidence >= 0.85 or '厂内' in plate.plateStr:  # 删去概率低于90的普通车牌
                    self._deadPlates.append(plate)
                self._movingPlates.remove(plate)
                killed = True
        # if not self._movingPlates:
        # tmp,tmp2=copy.deepcopy(self._movingPlates),copy.deepcopy(self._deadPlates)
        ret = self.getAll()
        Main().getInstance().signals.showDataSignal.emit(ret)
        # self._movingPlates,self._deadPlates=tmp,tmp2

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
            _editDistance = editDistance(savedPlate.plateStr, nowPlateInfo['plateStr'])
            if _editDistance < 4 and nowTime - savedPlate.endTime < self._lifeTimeLimit // 2:  # 编辑距离低于阈值，不比较方框位置
                yield savedPlate
            elif _editDistance < 5:  # 编辑距离适中，比较方框的位置有没有重合
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
        # 符合特殊车牌条件，修改其车牌号以符合特殊车牌的正常结构
        regexMatchesSpecialPlate = re.match(r'^.+?(2[1234][01]\d{2,}).*$', nowPlateInfo['plateStr'])
        if regexMatchesSpecialPlate:
            # '210', '211', '220', '221', '230', '240'
            nowPlateInfo['plateStr'] = '厂内' + regexMatchesSpecialPlate.group(1)[:5]
        # 跳过条件：车牌字符串太短
        if len(nowPlateInfo['plateStr']) < 7:
            return nowPlateInfo['plateStr'], nowPlateInfo['confidence']
        # 跳过条件：以英文字母开头（S和X除外）
        if 'A' <= nowPlateInfo['plateStr'][0] <= 'R' or 'T' <= nowPlateInfo['plateStr'][0] <= 'W' or 'Y' <= \
                nowPlateInfo['plateStr'][
                    0] <= 'Z':
            return nowPlateInfo['plateStr'], nowPlateInfo['confidence']
        # 开始分析：在储存的里找相似的车牌号
        similarPlates = list(self._getSimilarSavedPlates(nowPlateInfo, nowTime))
        if not similarPlates:  # 找不到相似的车牌号，插入新的
            # nowPlateInfo = list(nowPlateInfo) + [nowTime] * 2  # 初始化列表
            nowPlateInfo.update({'startTime': nowTime, 'endTime': nowTime})
            # （取巧部分）统计显示 95.9% 的概率成立
            if nowPlateInfo['plateStr'][1] == 'F' or nowPlateInfo['plateStr'][2] == 'F':
                nowPlateInfo['plateStr'] = '粤' + nowPlateInfo['plateStr'][nowPlateInfo['plateStr'].find('F'):]
            self._movingPlates.append(Plate(**nowPlateInfo))
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

    def _purgeAndMerge(self, plateList: List[Plate], threshhold=4, ignoreTime=False) -> None:
        if len(plateList) < 2:
            return
        plateList.sort(key=lambda plate: plate.startTime)  # 按照出现时间进行排序，相同的车牌会相邻
        # 合并相邻的相似车牌
        for i in range(len(plateList) - 1, 0, -1):
            thisStr, previousStr = plateList[i], plateList[i - 1]
            if editDistance(thisStr.plateStr, previousStr.plateStr) < threshhold and (
                    ignoreTime or thisStr.startTime <= previousStr.endTime):  # 合并相邻的编辑距离较小的车牌号
                endTime = max(thisStr.endTime, previousStr.endTime)
                if thisStr.confidence > previousStr.confidence:
                    thisStr.startTime = previousStr.startTime
                    thisStr.endTime = endTime
                    plateList[i], plateList[i - 1] = plateList[i - 1], plateList[i]
                else:
                    previousStr.endTime = endTime
                del plateList[i]

    def _mergeSamePlates(self) -> None:
        """
        相同车牌结果合并到一起
        :return:
        """
        self._purgeAndMerge(self._deadPlates)
        self._purgeAndMerge(self._movingPlates)

    def getAll(self) -> List[Plate]:
        """
        后期处理后返回所有的车牌List
        :return:
        """
        self._mergeSamePlates()  # 合并识别失误的车牌
        print('整理数据：大小从 %d ' % (len(self._deadPlates) + len(self._movingPlates)), end='')
        ans = sorted(self._deadPlates + self._movingPlates, key=lambda plate: plate.startTime)
        self._purgeAndMerge(ans, 1, True)  # 合并一模一样的车牌
        print('到 %d' % (len(self._deadPlates) + len(self._movingPlates)))
        return ans

    def getInfoDictFromList(self, detectionList: List) -> dict:
        """
        将识别出的List转换成self.VehiclePlate类型的Tuple
        :param detectionList:
        :return:
        """
        x, y, width, height = detectionList.pop(2)
        detectionList += [x, x + width, y, y + height, width, height]
        return self.VehiclePlate(*detectionList)

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


class Signals(QObject):
    """
    定义交互信号
    """
    showRawFrameSignal = pyqtSignal(np.ndarray)
    showDetectionFrameSignal = pyqtSignal(np.ndarray)
    showDataSignal = pyqtSignal(list)


class Main:
    _singleton = None

    @staticmethod
    def getInstance():
        if not Main._singleton:
            Main._singleton = Main()
        return Main._singleton

    def drawRectBox(self, image, rect, addText=None, rect_color=(0, 0, 255), text_color=(255, 255, 255)):
        """
        在image上画一个带文字的方框
        :param image: 原先的ndarray
        :param rect: [x, y, width, height]
        :param addText: 要加的文字
        :return: 画好的图像
        """
        cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), rect_color,
                      2,
                      cv2.LINE_AA)
        img = Image.fromarray(image)
        if addText:
            result=addText.split(); addText=result[0]+' '+ str(1-(1-float(result[1]))/5)
            cv2.rectangle(image, (int(rect[0] - 1), int(rect[1]) - 16), (int(rect[0] + 115), int(rect[1])), rect_color,
                          -1,
                          cv2.LINE_AA)
            img = Image.fromarray(image)
            draw = ImageDraw.Draw(img)
            draw.text((int(rect[0] + 1), int(rect[1] - 16)), addText, text_color, font=self.fontC)
        imagex = np.array(img)
        return imagex

    signals = Signals()  # 创建信号

    def detect(self, originImg: np.ndarray, frameIndex=-1) -> Tuple[np.ndarray, bool]:
        """
        检测核心函数（不显示）
        :param originImg:
        :param frameIndex:
        :return:
        """
        image = None
        # 检测，或使用bin文件进行回忆型检测
        if args.load_binary is None:
            resultList = self.model.SimpleRecognizePlateByE2E(originImg, self.tracker.multiTracker)
        else:
            resultList = self.binary.popLoaded()
        if args.save_binary is not None:
            self.binary.append(resultList)
        for plateStr, confidence, rect in resultList:
            if confidence > 0.85:
                if args.video:
                    vehiclePlate = self.tracker.getInfoDictFromList([plateStr, confidence, rect])
                    plateStr, confidence = self.tracker.analyzePlate(vehiclePlate, frameIndex)
                if self.tracker.multiTracker.workingTrackerCount() == 0:
                    image = self.drawRectBox(originImg, rect, plateStr + " " + str(round(confidence, 3)), (0, 0, 255),
                                             (255, 255, 255))
                    self.tracker.multiTracker.reborn(0)
                else:
                    image = self.drawRectBox(originImg, rect, plateStr + " " + str(round(confidence, 3)), (0, 0, 255),
                                             (255,255, 255))
                print("%s (%.5f)" % (plateStr, confidence))
            break  # 每帧只处理最有可能的车牌号
        return (image, True) if image is not None else (originImg, False)

    opencvShow = True  # 使用opencv显示

    def detectShow(self, originImg: np.ndarray, frameIndex=-1, wait=1) -> Tuple[np.ndarray, bool]:
        """
        检测核心函数（显示），可中断
        :param originImg:
        :param frameIndex:
        :return:
        """
        drawedImg, success = self.detect(originImg, frameIndex)
        if not self.opencvShow:
            if success:  # 检测成功再传给GUI
                self.signals.showDetectionFrameSignal.emit(drawedImg)
        else:
            cv2.imshow("Frame after detection", drawedImg)
            if cv2.waitKey(wait) == 27:
                return np.array([]), False
        return drawedImg, success

    def demoPhotos(self):
        # 处理所有照片
        for file in os.listdir(args.img_dir):
            # if not file.startswith('2020'):
            if not file.endswith('jpg'):
                continue
            print('<<<<<< ' + file + ' >>>>>>')
            self.detectShow(ImageUtil.Imread(os.path.join(args.img_dir, file)), wait=0)

    running = True  # 是否允许本类继续执行

    def demoVideo(self, showDialog=True):
        """
        测试视频
        :param args:
        :param showDialog: 显示输出窗口
        :return:
        """
        inStream, readThread = None, None
        placeCaptureStream, noSkipStream, recordingStream = None, None, None
        try:
            inStream = VideoUtil.OpenInputVideo(args.video)
            readThread = ReaderThread(inStream, args.video)
            readThread.start()
            frameIndex = 0
            # frameLimit = VideoUtil.GetVideoFramesCount(inStream) if 'rtsp' not in args.video else 2 ** 31 - 1
            frameLimit = 2 ** 31 - 1
            fps: int = VideoUtil.GetFps(inStream)
            self.fps = fps
            if args.rtsp or 1:  # 初始化当前时间
                nowTime = time.localtime()
                timestrap = time.mktime(
                    time.strptime("%d-%d-1 0:0:0" % (nowTime.tm_year, nowTime.tm_mon), "%Y-%m-%d %H:%M:%S"))
                offset = time.time() - timestrap + 24 * 3600
                frameIndex = int(offset * fps)
            if args.output:  # 有output才会打开这些输出的流
                if 'd' in args.video_write_mode:  # 只写入没被跳过的检测结果帧
                    placeCaptureStream = ffmpegUtil.OpenOutputVideo(args.output, VideoUtil.GetFps(inStream) / args.drop)
                if 's' in args.video_write_mode:  # 全程不跳帧，比dynamic多了被跳过的帧
                    insertIdx = args.output.rfind('.')
                    videoName = args.output[:insertIdx] + '.static' + args.output[insertIdx:]
                    noSkipStream = ffmpegUtil.OpenOutputVideo(videoName, VideoUtil.GetFps(inStream) / args.drop)
                if 'r' in args.video_write_mode:  # 实时录像，不做任何处理（如果是rtsp就是录像，如果是video就是转码）
                    insertIdx = args.output.rfind('.')
                    videoName = args.output[:insertIdx] + '.record' + args.output[insertIdx:]
                    recordingStream = ffmpegUtil.OpenOutputVideo(videoName, VideoUtil.GetFps(inStream))
            if args.load_binary:  # 如果有保存的检测则加载
                self.binary.load(args.load_binary)
            # frameLimit = min(frameLimit, 50000)  # 限制最大帧数，只处理视频前多少帧
            self.tracker = Tractor(fps * 3)  # 每个车牌两秒的寿命
            lastFrame = None
            while True:
                try:
                    frame = readThread.get(args.exitTimeout)
                    ffmpegUtil.WriteFrame(recordingStream, frame)
                    if args.rtsp and len(readThread) > args.memory_limit:  # 当读取的队列大于限定时
                        continue
                except:  # 当30秒取不到任何帧
                    readThread.stop()
                    break
                # 终止读取
                if frameIndex > frameLimit or not self.running:
                    break
                # 对原始帧的操作
                if showDialog:  # 保证每一帧都imshow过
                    if not self.opencvShow:
                        self.signals.showRawFrameSignal.emit(frame.copy())
                    else:
                        cv2.imshow('Raw frame', frame)
                        if cv2.waitKey(1) == 27:
                            break
                if args.drop != 1:  # imshow完了再跳过
                    if frameIndex % args.drop != 0:
                        frameIndex += 1
                        continue
                # 开始处理原始帧
                height, width, channel = frame.shape
                if lastFrame is not None:
                    oldpil = Image.fromarray(cv2.cvtColor(lastFrame, cv2.COLOR_BGR2RGB))  # PIL图像和cv2图像转化
                    nowpil = Image.fromarray(
                        cv2.cvtColor(frame[int(height * 0.3):, int(width * 0.3):], cv2.COLOR_BGR2RGB))
                    diff = ImageChops.difference(oldpil, nowpil)  # PIL图片库函数
                    try:
                        std: float = np.std(diff)
                        print('{%.3f}<%d><%dM>' % (std, readThread.qsize(), len(readThread) // 1048576), end='')
                    except:
                        traceback.print_exc()
                        std = 10000
                    if std < 9:
                        frameIndex += 1
                        print('\t已处理 %d / %d帧' % (frameIndex, frameLimit))
                        ffmpegUtil.WriteFrame(noSkipStream, frame)
                        continue
                startTime = time.time()
                lastFrame = frame[int(height * 0.3):, int(width * 0.3):]
                # <<<<< 核心函数 >>>>>
                frameDrawed, success = self.detectShow(frame, frameIndex) if showDialog else self.detect(frame,
                                                                                                         frameIndex)
                if frameDrawed.shape[0] == 0:
                    break
                try:
                    ffmpegUtil.WriteFrame(placeCaptureStream, frameDrawed)
                    ffmpegUtil.WriteFrame(noSkipStream, frameDrawed)
                except ValueError as e:
                    traceback.print_exc()
                frameIndex += 1
                print('\t已处理 %d / %d帧 (用时%f s)' % (frameIndex, frameLimit, time.time() - startTime))
            readThread.stop()
            self.running = False
            # 写日志
            if not args.output:
                return
            import os
            with open(os.path.join(os.path.dirname(args.output), os.path.basename(args.output).split('.')[0]) + '.txt',
                      'a') as fpLog:
                print('以下是检测到的车牌号：')
                allResult = self.tracker.getAll()
                self.signals.showDataSignal.emit(allResult)
                for resultPlate in allResult:
                    line = '%s [%s - %s]' % (
                        resultPlate.plateStr, sec2str(resultPlate.startTime / fps),
                        sec2str(resultPlate.endTime / fps))
                    print(line)
                    fpLog.write(line + '\n')
        except:  # 任何地方报错了不要管
            traceback.print_exc()
            print('检测结果已被保保存')
        finally:
            if showDialog:
                cv2.destroyAllWindows()
            if args.save_binary is not None:
                self.binary.save(args.save_binary)
            VideoUtil.CloseVideos(inStream)
            ffmpegUtil.CloseVideos(placeCaptureStream, noSkipStream, recordingStream)

    def start(self, argspace: argparse.Namespace):
        global args
        args = argspace
        # 检测到时rtsp则赋值进video
        if argspace.rtsp:
            argspace.video = argspace.rtsp
        argspace.exitTimeout = 10 if argspace.rtsp else 1  # 设置不同模式下的读取超时
        # 开始执行总程序
        globalStartTime = time.time()
        if argspace.img_dir is None:
            self.demoVideo()
        else:
            self.demoPhotos()
        # 统计执行的时长
        globalTimeSeconds = time.time() - globalStartTime
        globalTimeHours = globalTimeSeconds // 3600
        globalTimeMinutes = (globalTimeSeconds - globalTimeHours * 3600) // 60
        globalTimeSeconds = globalTimeSeconds % 60
        globalTime = '%d时%d分%.3f秒' % (
            globalTimeHours, globalTimeMinutes, globalTimeSeconds) if globalTimeHours != 0 else '%d分%.3f秒' % (
            globalTimeMinutes, globalTimeSeconds)
        print('总用时：' + globalTime)

    # 初始化
    fontC = ImageFont.truetype("./Font/platech.ttf", 14, 0)
    model = pr.LPR("./model/cascade.xml", "./model/model12.h5", "./model/ocr_plate_all_gru.h5")
    # model = pr.LPR("model/mssd512_voc.caffemodel", "model/model12.h5", "model/ocr_plate_all_gru.h5")
    binary = Serialization()
    tracker = Tractor(1)  # 设为1，意为禁用追踪功能


if __name__ == '__main__':
    args = getArgumentParser().parse_args()
    # args.rtsp = "rtsp://admin:klkj6021@172.19.13.27"
    main = Main.getInstance()
    main.start(args)
    # python -m cProfile -s cumulative demo.py >> profile.log
