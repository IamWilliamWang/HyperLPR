import argparse
import threading
import time
from typing import List
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QMessageBox, QInputDialog, QFileDialog, \
    QAbstractItemView, QHeaderView, QTableWidgetItem
import cv2
import sys
from PyQt5.QtWidgets import QWidget, QLabel, QApplication
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QPoint, QTimer, QElapsedTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QStandardItem

from Util import getArgumentParser, Plate, sec2tuple, sec2str


class QUtil:
    @staticmethod
    def toQImage(frame: np.ndarray):
        height, width, bytesPerComponent = frame.shape
        bytesPerLine = bytesPerComponent * width
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)
        image = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
        return image

    @staticmethod
    def toQPixmap(image):
        if type(image) == np.ndarray:
            return QPixmap.fromImage(QUtil.toQImage(image))
        if type(image) == QImage:
            return QPixmap.fromImage(image)


class GLWidget(QtWidgets.QOpenGLWidget):
    def __init__(self, parentWidget, drop=1):
        """

        :param parentWidget:
        :param drop: 每几帧保留一帧
        """
        super(GLWidget, self).__init__(parentWidget)
        self.sleep = False  # 休眠模式
        self._dropCount = drop - 1  # 每次绘图后丢弃几帧
        self._timer = 1

    @pyqtSlot(np.ndarray)
    def paintSlot(self, image: np.ndarray):
        """
        绘图动作，被观察者触发事件后执行
        :param image:
        :return:
        """
        if self.sleep:  # 休眠状态不绘图
            return
        if self._dropCount:  # 丢帧策略
            self._timer -= 1
            self._timer %= self._dropCount + 1
            if self._timer:
                return
        # 开始绘图
        painter = QPainter(self)
        painter.beginNativePainting()
        painter.drawImage(self.rect(), QUtil.toQImage(image))
        painter.endNativePainting()
        painter.end()

    def paintEvent(self, e: QtGui.QPaintEvent) -> None:
        """
        虚函数paintEvent在绘图后会触发。每次paint后刷新缓存，让Widet实时显示
        :param e:
        :return:
        :url: https://doc.qt.io/qtforpython/PySide2/QtWidgets/QOpenGLWidget.html#PySide2.QtWidgets.PySide2.QtWidgets.QOpenGLWidget.initializeGL
        """
        self.update()


class TableView(QtWidgets.QTableWidget):
    def __init__(self, parentWidget, cols):
        super().__init__(0, cols, parentWidget)

    @pyqtSlot(list)
    def showDataSlot(self, data: List[Plate]):
        import demo

        self.clearContents()
        for i in range(self.rowCount(), -1, -1):
            self.removeRow(i)

        for i, plate in enumerate(data):
            row = self.rowCount()
            self.insertRow(row)

            itemPlate = QTableWidgetItem()
            itemPlate.setTextAlignment(Qt.AlignCenter)
            itemPlate.setText(plate.plateStr)
            self.setItem(row, 0, itemPlate)

            itemTime1 = QTableWidgetItem()
            itemTime1.setTextAlignment(Qt.AlignCenter)
            d, h, m, s = sec2tuple(plate.startTime / demo.Main.getInstance().fps)
            nowTime = time.localtime()
            if d:
                itemTime1.setText('%d/%d/%d %d:%02d:%02d' % (nowTime.tm_year, nowTime.tm_mon, d, h, m, s))
            else:
                itemTime1.setText(sec2str(plate.startTime / demo.Main.getInstance().fps))
            self.setItem(row, 1, itemTime1)

            itemTime2 = QTableWidgetItem()
            itemTime2.setTextAlignment(Qt.AlignCenter)
            d, h, m, s = sec2tuple(plate.endTime / demo.Main.getInstance().fps)
            nowTime = time.localtime()
            if d:
                itemTime2.setText('%d/%d/%d %d:%02d:%02d' % (nowTime.tm_year, nowTime.tm_mon, d, h, m, s))
            else:
                itemTime2.setText(sec2str(plate.endTime / demo.Main.getInstance().fps))
            self.setItem(row, 2, itemTime2)


class Ui_MainWindow(QMainWindow):
    def __init__(self, args=None):
        super().__init__()
        self.initComponents()  # 初始化组件
        self.initTexts()  # 初始化显示文字
        self.initActions()  # 注册所有Click事件
        if not args:
            self.args = getArgumentParser().parse_args()
        else:
            self.args = args
            # 检测到时rtsp则赋值进video
            if args.rtsp:
                args.video = args.rtsp
        self.demoThread = None

    def initActions(self):
        self.toolStripMenuItemLaunch.triggered.connect(self.toolStripMenuItemLaunchClicked)
        self.toolStripMenuItemSetSourceAddress.triggered.connect(self.toolStripMenuItemSetSourceAddressClicked)
        self.toolStripMenuItemSetOutputFile.triggered.connect(self.toolStripMenuItemSetOutputFileClicked)
        self.toolStripMenuItemSetCatchMode.triggered.connect(self.toolStripMenuItemSetCatchModeClicked)
        self.toolStripMenuItemSetDropout.triggered.connect(self.toolStripMenuItemSetDropoutClicked)
        self.toolStripMenuItemSetMemory.triggered.connect(self.toolStripMenuItemSetMemoryClicked)
        self.toolStripMenuItemPopWindow.triggered.connect(self.toolStripMenuItemPopWindowClicked)
        self.toolStripMenuItemAbout.triggered.connect(self.toolStripMenuItemAboutClicked)
        self.toolStripMenuItemExit.triggered.connect(self.toolStripMenuItemExitClicked)
        self.tabControl.currentChanged.connect(self.tabSelectedIndexChanged)

    # region Qt Components Actions
    @pyqtSlot()
    def demoThreadExited(self):
        QMessageBox.information(self, '后台程序已结束', '所有帧已处理完毕，如果过早结束请检查地址是否正确')

    @pyqtSlot()
    def toolStripMenuItemLaunchClicked(self):
        def runnable():
            def connectConsoleAndGUI():
                console.signals.showRawFrameSignal.connect(self.openGLBoxRtsp.paintSlot)
                console.signals.showDetectionFrameSignal.connect(self.openGLBoxDetection.paintSlot)
                console.signals.showDataSignal.connect(self.tableView.showDataSlot)
                console.signals.threadExitSignal.connect(self.demoThreadExited)
                console.opencvShow = False

            import demo
            console = demo.Main.getInstance()
            # 连接demo的signals和GUI的slots
            connectConsoleAndGUI()
            console.start(self.args)

        if self.demoThread and self.demoThread.is_alive():  # 如果已经启动过一次
            QMessageBox.warning(self, '请勿重复启动', '可以尝试重启本系统后再启动检测')
            return
        self.demoThread = threading.Thread(target=runnable)
        self.demoThread.start()
        QMessageBox.information(self, '正在启动后台', '正在启动后台，需要等待大约10秒')
        self.toolStripMenuItemSetSourceAddress.setVisible(False)
        self.toolStripMenuItemSetOutputFile.setVisible(False)
        self.toolStripMenuItemSetCatchMode.setVisible(False)
        self.toolStripMenuItemSetDropout.setVisible(False)
        self.toolStripMenuItemPopWindow.setVisible(True)
        self.toolStripMenuItemLaunch.setVisible(False)

    @pyqtSlot()
    def toolStripMenuItemSetSourceAddressClicked(self):
        content = ['当前地址 %s' % (self.args.video if self.args.video else '(无地址)')]
        content += ['请输入新的摄像头地址（也支持视频文件）: ']
        inputText, okClicked = QInputDialog.getText(self, '设置摄像头地址', '\n'.join(content))
        if not inputText:
            return
        source: str = inputText.strip().replace('"', '')
        if 'rtsp' in source.lower():
            self.args.rtsp = source
        else:
            self.args.rtsp = ''
            self.args.video = source
        QMessageBox.information(self, '设置成功', '地址已指向' + source)

    @pyqtSlot()
    def toolStripMenuItemSetOutputFileClicked(self):
        filename, _ = QFileDialog.getSaveFileName(self, '选择要保存的位置',r"C:\Users\william\Desktop", filter='mp4 媒体文件 (*.mp4)')
        if not filename:
            return
        filename: str = filename.strip()
        if not filename.endswith('.mp4'):
            filename = filename[:filename.index('.')] + '.mp4'
        self.args.output = filename
        QMessageBox.information(self, '设置成功', '输出文件将保存为' + filename)

    @pyqtSlot()
    def toolStripMenuItemSetCatchModeClicked(self):
        inputText, okClicked = QInputDialog.getText(self, '切换抓拍模式', '\n'.join(
            ['当前模式%s, 模式介绍：' % self.args.video_write_mode, '(d)ynamic：输出抓拍的视频合集', '(s)tatic：输出所有帧的检测结果',
             '(r)ecord：输出原始录像(仅用于录制rtsp)', '输入括号内的字符: ']))
        if not okClicked:
            return
        inputText = inputText.strip()
        if len([ch for ch in inputText if ch in 'dsr']) != len(inputText):
            QMessageBox.warning(self, '输入有误', '请勿输入非法字符')
            return
        self.args.video_write_mode = inputText
        QMessageBox.information(self, '设置成功', '抓拍模式已经设置为' + (inputText if inputText else '(空)'))

    @pyqtSlot()
    def toolStripMenuItemSetDropoutClicked(self):
        inputText, okClicked = QInputDialog.getText(self, '性能设置', '\n'.join(
            ['当前性能等级 %d' % self.args.drop, '数字越大性能越快, 但丢帧率也会提高\n请输入一个正整数: ']))
        if not okClicked:
            return
        try:
            inputNumber = int(inputText) * 1024 ** 2
            if inputNumber <= 0:
                raise ValueError
        except:
            QMessageBox.warning(self, '输入有误', '请输入正整数')
            return
        self.args.drop = inputNumber
        QMessageBox.information(self, '设置成功', '性能已经设置为' + inputText)

    @pyqtSlot()
    def toolStripMenuItemSetMemoryClicked(self):
        inputText, okClicked = QInputDialog.getText(self, '设置内存限制', '\n'.join(
            ['当前最大内存限制 %d M' % (self.args.memory_limit // 1024 ** 2), '请输入一个整数(MB为单位)：']))
        if not okClicked:
            return
        try:
            inputNumber = float(inputText) * 1024 ** 2
            if inputNumber <= 0:
                raise ValueError
        except:
            QMessageBox.warning(self, '输入有误', '请输入正整数')
            return
        self.args.memory_limit = inputNumber
        QMessageBox.information(self, '设置成功', '最大内存限制已经设置为 ' + inputText + ' M')

    @pyqtSlot()
    def toolStripMenuItemPopWindowClicked(self):
        import demo
        if demo.Main.getInstance().opencvShow:
            cv2.destroyAllWindows()
            demo.Main.getInstance().opencvShow = False
            self.toolStripMenuItemPopWindow.setText(QtCore.QCoreApplication.translate("MainWindow", "弹出显示窗口"))
            QMessageBox.information(self, '设置成功', '已还原显示窗口')
        else:
            demo.Main.getInstance().opencvShow = True
            self.toolStripMenuItemPopWindow.setText(QtCore.QCoreApplication.translate("MainWindow", "还原显示窗口"))
            QMessageBox.information(self, '设置成功', '已弹出显示窗口')

    @pyqtSlot()
    def toolStripMenuItemAboutClicked(self):
        QMessageBox.about(self, '关于软件', '厂内车牌检测系统 v1.0')

    @pyqtSlot()
    def toolStripMenuItemExitClicked(self):
        import demo
        demo.Main.getInstance().running = False
        self.close()

    @pyqtSlot()
    def tabSelectedIndexChanged(self):
        """
        Tab切换事件
        :return:
        """
        if self.tabControl.currentIndex():
            self.openGLBoxRtsp.sleep = True
            self.tabControl.setTabText(self.tabControl.indexOf(self.tabPageDetection),
                                       QtCore.QCoreApplication.translate("MainWindow", "抓拍中..."))
        else:
            self.openGLBoxRtsp.sleep = False
            self.tabControl.setTabText(self.tabControl.indexOf(self.tabPageDetection),
                                       QtCore.QCoreApplication.translate("MainWindow", "检测抓拍"))

    # endregion

    # region Auto-generated code. DO NOT EDIT!
    def initComponents(self):
        desktop = QApplication.desktop()
        screenRect = desktop.screenGeometry()
        screenHeight = screenRect.height() - 70
        screenWidth = screenRect.width()
        self.setObjectName("MainWindow")
        self.resize(screenWidth, screenHeight)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.tableView = TableView(self.centralwidget, 3)
        self.tableView.setGeometry(QtCore.QRect(int(0.7861*screenWidth), int(0.0341*screenHeight), int(0.2074*screenWidth), int(0.9226*screenHeight)))
        self.tableView.setObjectName("tableView")
        self.tableView.setHorizontalHeaderLabels(['车牌号', '出现时间', '离开时间'])
        self.tableView.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.tableView.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tableView.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableView.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tableView.setEditTriggers(QAbstractItemView.NoEditTriggers)
        # self.tableView.cellClicked.connect(self.recognize_one_license_plate)
        self.tabControl = QtWidgets.QTabWidget(self.centralwidget)
        self.tabControl.setGeometry(QtCore.QRect(int(0.00591*screenWidth), int(0.01138*screenHeight), int(0.7748*screenWidth), int(0.9454*screenHeight)))
        self.tabControl.setObjectName("tabControl")
        self.tabPageRtsp = QtWidgets.QWidget()
        self.tabPageRtsp.setObjectName("tabPageRtsp")
        self.openGLBoxRtsp = GLWidget(self.tabPageRtsp, 4)
        self.openGLBoxRtsp.setGeometry(QtCore.QRect(int(0.00591*screenWidth), int(0.01138*screenHeight), (0.7630*screenWidth), int(0.8998*screenHeight)))
        self.openGLBoxRtsp.setObjectName("openGLBoxRtsp")
        self.tabControl.addTab(self.tabPageRtsp, "")
        self.tabPageDetection = QtWidgets.QWidget()
        self.tabPageDetection.setObjectName("tabPageDetection")
        self.openGLBoxDetection = GLWidget(self.tabPageDetection)
        self.openGLBoxDetection.setGeometry(QtCore.QRect(int(0.00591*screenWidth), int(0.01138*screenHeight), (0.7630*screenWidth), int(0.8998*screenHeight)))
        self.openGLBoxDetection.setObjectName("openGLBoxDetection")
        self.tabControl.addTab(self.tabPageDetection, "")
        self.setCentralWidget(self.centralwidget)
        self.menu = QtWidgets.QMenuBar(self)
        self.menu.setGeometry(QtCore.QRect(0, 0, screenWidth, 23))
        self.menu.setObjectName("menu")
        self.menuStripMain = QtWidgets.QMenu(self.menu)
        self.menuStripMain.setObjectName("menuStripMain")
        self.setMenuBar(self.menu)
        self.toolStripMenuItemLaunch = QtWidgets.QAction(self)
        self.toolStripMenuItemLaunch.setObjectName("toolStripMenuItemLaunch")
        self.toolStripMenuItemSetSourceAddress = QtWidgets.QAction(self)
        self.toolStripMenuItemSetSourceAddress.setObjectName("toolStripMenuItemSetSourceAddress")
        self.toolStripMenuItemSetCatchMode = QtWidgets.QAction(self)
        self.toolStripMenuItemSetCatchMode.setObjectName("toolStripMenuItemSetCatchMode")
        self.toolStripMenuItemSetOutputFile = QtWidgets.QAction(self)
        self.toolStripMenuItemSetOutputFile.setObjectName("toolStripMenuItemSetOutputFile")
        self.toolStripMenuItemSetDropout = QtWidgets.QAction(self)
        self.toolStripMenuItemSetDropout.setObjectName("toolStripMenuItemSetDropout")
        self.toolStripMenuItemSetMemory = QtWidgets.QAction(self)
        self.toolStripMenuItemSetMemory.setObjectName("toolStripMenuItemSetMemory")
        self.toolStripMenuItemPopWindow = QtWidgets.QAction(self)
        self.toolStripMenuItemPopWindow.setObjectName("toolStripMenuItemPopWindow")
        self.toolStripMenuItemPopWindow.setVisible(False)
        self.toolStripMenuItemAbout = QtWidgets.QAction(self)
        self.toolStripMenuItemAbout.setObjectName("toolStripMenuItemAbout")
        self.toolStripMenuItemExit = QtWidgets.QAction(self)
        self.toolStripMenuItemExit.setObjectName("toolStripMenuItemExit")
        self.menuStripMain.addAction(self.toolStripMenuItemLaunch)
        self.menuStripMain.addAction(self.toolStripMenuItemSetSourceAddress)
        self.menuStripMain.addAction(self.toolStripMenuItemSetOutputFile)
        self.menuStripMain.addAction(self.toolStripMenuItemSetCatchMode)
        self.menuStripMain.addAction(self.toolStripMenuItemSetDropout)
        self.menuStripMain.addAction(self.toolStripMenuItemSetMemory)
        self.menuStripMain.addAction(self.toolStripMenuItemPopWindow)
        self.menuStripMain.addAction(self.toolStripMenuItemAbout)
        self.menuStripMain.addAction(self.toolStripMenuItemExit)
        self.menu.addAction(self.menuStripMain.menuAction())

        self.tabControl.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(self)

    def initTexts(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "厂内车牌检测系统"))
        self.tabControl.setTabText(self.tabControl.indexOf(self.tabPageRtsp), _translate("MainWindow", "实时监控"))
        self.tabControl.setTabText(self.tabControl.indexOf(self.tabPageDetection), _translate("MainWindow", "检测抓拍"))
        self.menuStripMain.setTitle(_translate("MainWindow", "主菜单"))
        self.toolStripMenuItemLaunch.setText(_translate("MainWindow", "开始检测"))
        self.toolStripMenuItemSetSourceAddress.setText(_translate("MainWindow", "设置摄像头地址"))
        self.toolStripMenuItemSetCatchMode.setText(_translate("MainWindow", "切换抓拍模式"))
        self.toolStripMenuItemSetOutputFile.setText(_translate("MainWindow", "设置输出文件"))
        self.toolStripMenuItemSetDropout.setText(_translate("MainWindow", "性能设置"))
        self.toolStripMenuItemSetMemory.setText(_translate("MainWindow", "设置最大内存限制"))
        self.toolStripMenuItemPopWindow.setText(_translate("MainWindow", "弹出显示窗口"))
        self.toolStripMenuItemAbout.setText(_translate("MainWindow", "关于"))
        self.toolStripMenuItemExit.setText(_translate("MainWindow", "退出"))
    # endregion


if __name__ == '__main__':
    app = QApplication(sys.argv)
    args = getArgumentParser().parse_args()
    mainWindow = Ui_MainWindow(args)  # 没有初始化不用传参数
    mainWindow.show()
    sys.exit(app.exec_())
