# pyqt5相关UI设置
import os, sys, requests, pickle
import numpy as np
from PyQt5.QtCore import QFile, QTextStream, QSize
from PyQt5.QtWidgets import QWidget, QApplication, QVBoxLayout, \
    QHBoxLayout, QPushButton, QLabel, QFileDialog, QDesktopWidget
from PyQt5.Qt import QPixmap, QPoint, Qt, QPainter, QIcon


def is_empty(value):
    # 判断值是否为空
    return True if value is "" else False


def make_dir(path):
    # 判断文件是否存在
    if os.path.isdir(path):
        os.mkdir(path)


class ImageBox(QWidget):
    def __init__(self):
        super(ImageBox, self).__init__()
        self.img = None
        self.scaled_img = None
        self.point = QPoint(0, 0)
        self.start_pos = None
        self.end_pos = None
        self.left_click = False
        self.scale = 1
        self.SR_flag = False

    def init_ui(self):
        self.setWindowTitle("ImageBox")

    def set_image(self, img_path):
        """
        open image file
        :param img_path: image file path
        :return:
        """
        self.img = QPixmap(img_path)
        if self.SR_flag:
            self.scaled_img = self.img.scaled(self.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
            self.SR_flag = False
        else:
            self.scaled_img = self.img.scaled(self.img.size(), Qt.KeepAspectRatio, Qt.FastTransformation)

    def paintEvent(self, e):
        if self.scaled_img:
            painter = QPainter()
            painter.begin(self)
            painter.scale(self.scale, self.scale)
            painter.drawPixmap(self.point, self.scaled_img)
            painter.end()

    def mouseMoveEvent(self, e):
        if self.left_click:
            self.end_pos = e.pos() - self.start_pos
            self.point = self.point + self.end_pos
            self.start_pos = e.pos()
            self.repaint()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.left_click = True
            self.start_pos = e.pos()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.left_click = False



