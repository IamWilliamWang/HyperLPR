import argparse
import pickle
import sys
import warnings
from dataclasses import dataclass
from typing import List

import cv2
import imageio
import numpy as np


class ImageUtil:
    @staticmethod
    def Imread(filename_unicode: str) -> np.ndarray:
        """
        读取可能含有unicode文件名的图片
        :param filename_unicode: 可能含有unicode的图片名
        :return: 图片帧
        """
        return cv2.imdecode(np.fromfile(filename_unicode, dtype=np.uint8), -1)

    @staticmethod
    def Imwrite(filename_unicode: str, frame: np.ndarray) -> None:
        """
        向文件写入该帧
        :param filename_unicode: 可能含有unicode的图片名
        :param frame: 要写入的帧
        """
        extension = filename_unicode[filename_unicode.rfind('.'):]
        cv2.imencode(extension, frame)[1].tofile(filename_unicode)

    @staticmethod
    def IsGrayImage(grayOrImg: np.ndarray) -> bool:
        """
        检测是否为灰度图，灰度图为True，彩图为False
        :param grayOrImg: 图片帧
        :return: 是否为灰度图
        """
        return len(grayOrImg.shape) is 2


class VideoUtil:
    @staticmethod
    def OpenVideos(inputVideoSource: str = None, outputVideoFilename: str = None, outputVideoEncoding='DIVX') -> (
            cv2.VideoCapture, cv2.VideoWriter):  # MPEG-4编码
        """
        打开读取和输出视频文件
        :param inputVideoSource: 输入文件名或视频流
        :param outputVideoFilename: 输出文件名或视频流
        :param outputVideoEncoding: 输出文件的视频编码
        :return: 输入输出文件流
        """
        videoInput = None
        videoOutput = None
        if inputVideoSource is not None:
            videoInput = VideoUtil.OpenInputVideo(inputVideoSource)  # 打开输入视频文件
        if outputVideoFilename is not None:
            videoOutput = VideoUtil.OpenOutputVideo(outputVideoFilename, videoInput, outputVideoEncoding)
        return videoInput, videoOutput

    @staticmethod
    def OpenInputVideo(inputVideoSource: str) -> cv2.VideoCapture:
        """
        打开要读取的视频文件
        :param inputVideoSource: 输入文件名或视频流
        :return: 可读取的文件流
        """
        return cv2.VideoCapture(inputVideoSource)

    @staticmethod
    def OpenOutputVideo(inputFileStream: cv2.VideoCapture, outputVideoFilename: str,
                        outputVideoEncoding='DIVX') -> cv2.VideoWriter:
        """
        打开输出视频文件
        :param inputFileStream: 输入文件流（用户获得视频基本信息）
        :param outputVideoFilename: 输出文件名
        :param outputVideoEncoding: 输出文件编码
        :return: 输出文件流
        """
        # 获得码率及尺寸
        fps = int(inputFileStream.get(cv2.CAP_PROP_FPS))
        size = (int(inputFileStream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(inputFileStream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # return cv2.VideoWriter(outputVideoFilename, 0x00000021, fps, size, False)
        if outputVideoEncoding.upper() in ['DIVX', 'XVID']:
            return cv2.VideoWriter(outputVideoFilename.split('.')[0] + '.avi',
                                   cv2.VideoWriter_fourcc(*outputVideoEncoding), fps, size, False)
        elif outputVideoEncoding.upper() == 'MP4V':
            return cv2.VideoWriter(outputVideoFilename.split('.')[0] + '.mp4',
                                   cv2.VideoWriter_fourcc(*outputVideoEncoding), fps, size, False)
        return cv2.VideoWriter(outputVideoFilename, cv2.VideoWriter_fourcc(*outputVideoEncoding), fps, size, False)

    @staticmethod
    def ReadFrame(stream: cv2.VideoCapture) -> np.ndarray:
        """
        从输入流中读取一帧，如果没有读取则返回array([])
        :param stream: 可读取的输入流
        :return: 读取的一帧
        """
        ret, frame = stream.read()
        if ret is False:
            return np.array([])
        return frame

    @staticmethod
    def ReadFrames(stream: cv2.VideoCapture, readFramesCount: int = sys.maxsize) -> List[np.ndarray]:
        """
        从输入流中读取最多readFramesCount个帧并返回，如果没有读取则返回[]
        :param stream: 输入流
        :param readFramesCount: 要读取的帧数
        :return: 读取的帧列表
        """
        frames = []
        while stream.isOpened():
            ret, frame = stream.read()
            if ret is False:
                break
            frames.append(frame)
            if len(frames) >= readFramesCount:
                break
        return frames

    @staticmethod
    def WriteFrame(stream: cv2.VideoWriter, frame: np.ndarray) -> None:
        """
        向输出流写入一帧
        :param stream: 可进行写入的视频输出流
        :param frame: 要写的一帧
        """
        if stream:
            stream.write(frame)

    @staticmethod
    def WriteFrames(stream: cv2.VideoWriter, frames: List[np.ndarray]) -> None:
        """
        向输出流写入多帧
        :param stream: 可进行写入的视频输出流
        :param frames: 要写的多帧
        :return:
        """
        if stream:
            for writeFrame in frames:
                stream.write(writeFrame)

    @staticmethod
    def GetNowPosition(stream: cv2.VideoCapture) -> int:
        """
        获得视频输入流指向的当前帧位置
        :param stream: 可读取的视频输入流
        :return: 当前指向的位置
        """
        return stream.get(cv2.CAP_PROP_POS_FRAMES)

    @staticmethod
    def SetNowPosition(stream: cv2.VideoCapture, framesPosition: int) -> None:
        """
        设定视频输入流指向的当前帧位置
        :param stream: 可读取的视频输入流
        :param framesPosition: 要指向的位置
        """
        stream.set(cv2.CAP_PROP_POS_FRAMES, framesPosition)

    @staticmethod
    def SkipNowPosition(stream: cv2.VideoCapture, skippedFramesCount: int) -> None:
        """
        视频输入流指向的帧位置跳过多少帧。等价于SetPosition(GetPosition+skippedFramesCount)
        :param stream: 可读取的视频输入流
        :param skippedFramesCount: 想要跳过多少帧
        """
        VideoUtil.SetNowPosition(stream, VideoUtil.GetNowPosition(stream) + skippedFramesCount)

    @staticmethod
    def GetFps(videoStream: cv2.VideoCapture) -> int:
        """
        获得视频流的FPS
        :param videoStream: 可读取的视频输入流
        :return: 每秒多少帧
        """
        return int(videoStream.get(cv2.CAP_PROP_FPS))

    @staticmethod
    def GetVideoFramesCount(videoFileStream: cv2.VideoCapture) -> int:
        """
        获得视频文件的总帧数
        :param videoFileStream: 可读取的视频输入流
        :return: 视频文件的总帧数
        """
        return videoFileStream.get(cv2.CAP_PROP_FRAME_COUNT)

    @staticmethod
    def GetWidthAndHeight(videoStream: cv2.VideoCapture) -> (int, int):
        """
        获得视频流的宽度和高度
        :param videoStream: 可读取的视频输入流
        :return: 视频流的宽度和高度
        """
        return int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @staticmethod
    def CloseVideos(*videoStreams) -> None:
        """
        关闭所有视频文件
        :param videoStreams: 所有可读取的视频输入流
        """
        for videoStream in videoStreams:
            if videoStream is not None:
                videoStream.release()


class ffmpegUtil:
    @staticmethod
    def OpenOutputVideo(outName: str, fps):
        return imageio.get_writer(outName, fps=fps)

    @staticmethod
    def WriteFrame(writer, img):
        if writer:
            writer.append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    @staticmethod
    def CloseVideos(*streams) -> None:
        """
        关闭所有视频文件
        :param streams: 所有可读取的视频输入流
        """
        for videoStream in streams:
            if videoStream is not None:
                videoStream.close()


class Serialization:
    def __init__(self):
        self._database = []
        self._pointer = 0

    def append(self, obj):
        self._database += [obj]

    def save(self, binFilename: str):
        with open(binFilename, 'wb') as f:
            pickle.dump(self._database, f)

    def load(self, binFilename: str):
        with open(binFilename, 'rb') as f:
            self._database = pickle.load(f)
        return self._database

    def popLoaded(self):
        if self._pointer >= len(self._database):
            warnings.warn('错误：提取数据失败！预加载队列全部出队完毕！', stacklevel=2)
            return []
        popedElement = self._database[self._pointer]
        self._pointer += 1
        return popedElement  # [pointer++]


def editDistance(word1: str, word2: str) -> int:
    """
        计算两个字符串的最小编辑距离
        :param word1:
        :param word2:
        :return:
        """
    if '厂内' in word1 and '厂内' in word2:
        return 0 if word1 == word2 else 2 ** 10
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


@dataclass
class Plate:
    """
    数据类，作为储存车牌的基础单元
    """
    plateStr: str  # 车牌号
    confidence: float  # 置信度
    left: float  # 车牌框的左侧x坐标
    right: float  # 车牌框的右侧x坐标
    top: float  # 车牌框的上侧y坐标
    bottom: float  # 车牌框的下侧y坐标
    width: float  # 车牌框的宽度
    height: float  # 车牌框的高度
    startTime: int  # 车牌框开始出现的时间
    endTime: int  # 车牌框完全消失的时间

    def __str__(self) -> str:
        return "Plate{str='%s', confidence=%f, left=%f, right=%f, top=%f, bottom=%f, width=%f, height=%f, startTime=%d, endTime=%d}" % \
               (self.plateStr, self.confidence, self.left, self.right, self.top, self.bottom, self.width,
                self.height, self.startTime, self.endTime)


def sec2tuple(seconds) -> tuple:
    d, h, m, s = 0, 0, 0, seconds
    if s >= 60:
        m = s // 60
        s -= m * 60
    if m >= 60:
        h = m // 60
        m -= h * 60
    if h >= 24:
        d = h // 24
        h -= d * 24
    return d, h, m, s


def sec2str(seconds) -> str:
    d, h, m, s = sec2tuple(seconds)
    if d:
        return '%d日%d时%d分%.2f秒' % (d, h, m, s)
    if h:
        return '%d时%d分%.2f秒' % (h, m, s)
    if m:
        return '%d分%.2f秒' % (m, s)
    return '%.2f秒' % s


def getArgumentParser():
    parser = argparse.ArgumentParser(description='车牌识别程序')
    parser.add_argument('-dir', '--img_dir', type=str, help='要检测的图片文件夹', default=None)
    parser.add_argument('-v', '--video', type=str, help='想检测的视频文件名', default=None)
    parser.add_argument('-vwm', '--video_write_mode', type=str,
                        help='写入模式设置。(d)ynamic：只保留动态帧的检测结果(输出抓拍的视频合集)。(s)tatic：保留所有帧的检测结果。(r)ecord：保留原始录像(仅用于录制rtsp)',
                        default='d')
    parser.add_argument('-rtsp', '--rtsp', type=str, help='使用rtsp地址的视频流进行检测', default=None)
    parser.add_argument('-out', '--output', type=str, help='输出的视频名', default=None)
    parser.add_argument('-drop', '--drop', type=int, help='每隔多少帧保留一帧', default=1)
    parser.add_argument('-save_bin', '--save_binary', type=str, help='每一帧的检测结果保存为什么文件名', default=None)
    parser.add_argument('-load_bin', '--load_binary', type=str, help='加载每一帧的检测结果，不使用video而是用加载的结果进行测试', default=None)
    parser.add_argument('-memory', '--memory_limit', type=int, help='内存上限限制为多少字节', default=1024 ** 3 * 1)
    return parser
