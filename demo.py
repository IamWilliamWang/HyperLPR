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


from testVideos import ImageUtil
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
    def Imwrite(filename_unicode: str, frame) -> None:
        """
        向文件写入该帧
        :param filename_unicode: 可能含有unicode的图片名
        :param frame: 要写入的帧
        """
        extension = filename_unicode[filename_unicode.rfind('.'):]
        cv2.imencode(extension, frame)[1].tofile(filename_unicode)

    @staticmethod
    def IsGrayImage(grayOrImg) -> bool:
        """
        检测是否为灰度图，灰度图为True，彩图为False
        :param grayOrImg: 图片帧
        :return: 是否为灰度图
        """
        return len(grayOrImg.shape) is 2


class VideoUtil:
    @staticmethod
    def OpenVideos(inputVideoSource=None, outputVideoFilename=None, outputVideoEncoding='DIVX') -> tuple(
        [cv2.VideoCapture, cv2.VideoWriter]):  # MPEG-4编码
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
    def OpenOutputVideo(outputVideoFilename: str, inputFileStream: cv2.VideoCapture,
                        outputVideoEncoding='DIVX') -> cv2.VideoWriter:
        """
        打开输出视频文件
        :param outputVideoFilename: 输出文件名
        :param inputFileStream: 输入文件流（用户获得视频基本信息）
        :param outputVideoEncoding: 输出文件编码
        :return: 输出文件流
        """
        # 获得码率及尺寸
        fps = int(inputFileStream.get(cv2.CAP_PROP_FPS))
        size = (int(inputFileStream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(inputFileStream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # return cv2.VideoWriter(outputVideoFilename, cv2.VideoWriter_fourcc(*outputVideoEncoding), fps, size, False)
        return cv2.VideoWriter(outputVideoFilename, 0x00000021, fps, size, False)

    @staticmethod
    def ReadFrames(stream: cv2.VideoCapture, readFramesCount: int) -> list:
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
            frames += [frame]
            if len(frames) >= readFramesCount:
                break
        return frames

    @staticmethod
    def ReadFrame(stream: cv2.VideoCapture) -> np.ndarray:
        """
        从输入流中读取一帧，如果没有读取则返回[]
        :param stream: 可读取的输入流
        :return: 读取的一帧
        """
        read = VideoUtil.ReadFrames(stream, 1)
        return np.asarray(read[0]) if read else []

    @staticmethod
    def WriteFrame(stream: cv2.VideoWriter, frame: np.ndarray) -> None:
        """
        向输出流写入一帧
        :param stream: 可进行写入的视频输出流
        :param frame: 要写的一帧
        """
        stream.write(frame)

    @staticmethod
    def GetPosition(stream: cv2.VideoCapture) -> int:
        """
        获得视频输入流指向的当前帧位置
        :param stream: 可读取的视频输入流
        :return: 当前指向的位置
        """
        return stream.get(cv2.CAP_PROP_POS_FRAMES)

    @staticmethod
    def SetPosition(stream: cv2.VideoCapture, framesPosition: int) -> None:
        """
        设定视频输入流指向的当前帧位置
        :param stream: 可读取的视频输入流
        :param framesPosition: 要指向的位置
        """
        stream.set(cv2.CAP_PROP_POS_FRAMES, framesPosition)

    @staticmethod
    def SkipReadFrames(stream: cv2.VideoCapture, skippedFramesCount: int) -> None:
        """
        视频输入流指向的帧位置跳过多少帧。等价于SetPosition(GetPosition+skippedFramesCount)
        :param stream: 可读取的视频输入流
        :param skippedFramesCount: 想要跳过多少帧
        """
        VideoUtil.SetPosition(stream, VideoUtil.GetPosition(stream) + skippedFramesCount)

    @staticmethod
    def GetFps(videoStream: cv2.VideoCapture) -> int:
        """
        获得视频流的FPS
        :param videoStream: 可读取的视频输入流
        :return: 每秒多少帧
        """
        return int(videoStream.get(cv2.CAP_PROP_FPS))

    @staticmethod
    def GetVideoFileFrameCount(videoFileStream: cv2.VideoCapture) -> int:
        """
        获得视频文件的总帧数
        :param videoFileStream: 可读取的视频输入流
        :return: 视频文件的总帧数
        """
        return videoFileStream.get(cv2.CAP_PROP_FRAME_COUNT)

    @staticmethod
    def GetWidthAndHeight(videoStream: cv2.VideoCapture) -> tuple([int, int]):
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
            videoStream.release()


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


def detect(originImg: np.ndarray, frameIndex = -1) -> np.ndarray:
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
            if len(vehiclePlatesEndAt) == len(vehiclePlatesStartAt) - 1:
                vehiclePlatesEndAt += [frameIndex]
            else:
                vehiclePlatesEndAt[-1] = frameIndex
            return plateStr, confidence
        else:  # saved plate can be better
            if len(vehiclePlatesEndAt) == len(vehiclePlatesStartAt) - 1:
                vehiclePlatesEndAt += [frameIndex]
            else:
                vehiclePlatesEndAt[-1] = frameIndex
            return vehiclePlates[-1], vehiclePlatesConfidence[-1]
    image = None
    for plateStr, confidence, rect in model.SimpleRecognizePlateByE2E(originImg):
        if confidence > 0.8:
            plateStr, confidence = analyzePlate(plateStr, confidence)
            image = drawRectBox(originImg, rect, plateStr + " " + str(round(confidence, 3)))
            print("plate_str: %s, confidence: %f" % (plateStr, confidence))
    return image if image is not None else originImg


def detectShow(arg, frameIndex = -1):
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
    outStream = VideoUtil.OpenOutputVideo('20200422.mp4', inStream)
    frameIndex = 0
    frameLimit = VideoUtil.GetVideoFileFrameCount(inStream)
    frameLimit = 100 if frameLimit > 100 else frameLimit
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
    with open('20200422.txt', 'a') as fpLog:
        print('以下是检测到的车牌号：')
        for i in range(len(vehiclePlates)):
            print(vehiclePlates[i], vehiclePlatesConfidence[i])
            if vehiclePlatesStartAt[i] != -1:
                fpLog.write('%s %.3f [%.2f-%.2f秒]\n' % (vehiclePlates[i], vehiclePlatesConfidence[i], vehiclePlatesStartAt[i]/fps, vehiclePlatesEndAt[i]/fps))
            else:
                fpLog.write('%s %.3f\n' % (vehiclePlates[i], vehiclePlatesConfidence[i]))


if __name__ == '__main__':
    model = pr.LPR("model/cascade.xml", "model/model12.h5", "model/ocr_plate_all_gru.h5")
    vehiclePlates = []
    vehiclePlatesConfidence = []
    vehiclePlatesStartAt = []
    vehiclePlatesEndAt = []
    demoVideo()
