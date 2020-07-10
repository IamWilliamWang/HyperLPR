import pickle
import sys
import warnings
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


# import av
#
# duration = 4
# fps = 24
# total_frames = duration * fps
# container = av.open('test.mp4', mode='w')
# stream = container.add_stream('mpeg4', rate=fps)
# stream.width = 480
# stream.height = 320
# stream.pix_fmt = 'yuv420p'
#
# for frame_i in range(total_frames):
#     img = np.empty((480, 320, 3))
#     img[:, :, 0] = 0.5 + 0.5 * np.sin(2 * np.pi * (0 / 3 + frame_i / total_frames))
#     img[:, :, 1] = 0.5 + 0.5 * np.sin(2 * np.pi * (1 / 3 + frame_i / total_frames))
#     img[:, :, 2] = 0.5 + 0.5 * np.sin(2 * np.pi * (2 / 3 + frame_i / total_frames))
#
#     img = np.round(255 * img).astype(np.uint8)
#     img = np.clip(img, 0, 255)
#
#     frame = av.VideoFrame.from_ndarray(img, format='rgb24')
#     for packet in stream.encode(frame):
#         container.mux(packet)
#
# #Flush stream
# for packet in stream.encode():
#     container.mux(packet)
#
# #Close the file
# container.close()
