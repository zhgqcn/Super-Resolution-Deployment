import sys,math, logging, re, os, cv2, time
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import uic
from ui.MainWin import *
from utils.connectServer import SRApi
from utils.generateLog import QTextEditLogger
from utils.pyqtUI import *


class MainWin(QWidget):
    def __init__(self):
        super(MainWin, self).__init__()

        self.ui = uic.loadUi('./ui/MainWin.ui', self)
        self.ui.showMaximized()
        self.ip_address = ''
        self.is_input = False
        self.image_path = None
        self.image_save = None
        self.is_sr = False
        self.init_ui()
        self.save = './result/'
        self.box = ImageBox()
        self.box.setMaximumSize(900, 900)
        self.box.resize(900, 900)
        self.horizontalLayout_3.addWidget(self.box)
        self.box.set_image(None)

    def init_ui(self):
        # 加载并显示UI
        self.ui.setWindowTitle('医学多光子图像超分辨系统')

        # 设置图标
        self.setWindowIcon(QIcon('./ui/icon/sr.png'))
        self.UploadPushButton.setIcon(QIcon('./ui/icon/upload_icon.png'))
        self.SRPushButton.setIcon(QIcon('./ui/icon/start_icon_begin.png'))
        self.ZoomPushButton.setIcon(QIcon('./ui/icon/zoom.png'))
        self.ZoomBigPushButton.setIcon(QIcon('./ui/icon/zoom_in.png'))
        self.ZoomSmallPushButton.setIcon(QIcon('./ui/icon/zoom_out.png'))
        self.SavePushButton.setIcon(QIcon('./ui/icon/save_icon.png'))
        self.ServerPushButton.setIcon(QIcon('./ui/icon/commit.png'))
        self.SR2PushButton.setIcon(QIcon('./ui/icon/x2.png'))
        self.SR4PushButton.setIcon(QIcon('./ui/icon/x4.png'))
        self.NoSRPushButton.setIcon(QIcon('./ui/icon/x1.png'))
        self.UploadLabel.setPixmap(QPixmap('./ui/icon/upload_label_icon.png'))

        # 设置状态栏输出
        log_text_box = QTextEditLogger(self.plainTextEdit)
        log_text_box.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(log_text_box)
        logging.getLogger().setLevel(logging.DEBUG)

        # 设置信号与槽的关系
        self.ServerPushButton.clicked.connect(self.get_server)
        self.UploadPushButton.clicked.connect(self.upload_image)
        self.SRPushButton.clicked.connect(self.sr_image)
        self.NoSRPushButton.clicked.connect(lambda: self.change_display('X1'))
        self.SR2PushButton.clicked.connect(lambda: self.change_display('X2'))
        self.SR4PushButton.clicked.connect(lambda: self.change_display('X4'))
        self.ZoomPushButton.clicked.connect(self.load_zoom)
        self.ZoomBigPushButton.clicked.connect(self.large_click)
        self.ZoomSmallPushButton.clicked.connect(self.small_click)
        self.SavePushButton.clicked.connect(self.save_image)

    def get_server(self):
        self.ip_address = self.IpLineEdit.text()
        regex = re.compile(
                r'^(?:http|ftp)s?://'  # http:// or https://
                r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
                r'localhost|'  # localhost...
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
                r'(?::\d+)?'  # optional port
                r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        if re.match(regex, self.ip_address) is not None:
            logging.info("\n 地址保存成功！")
        else:
            logging.info("\n 无效地址, 请重新输入！")
            self.ip_address = ''

    def upload_image(self):
        self.image_path, _ = QFileDialog.getOpenFileName(self, 'Open file', '.', "Image files (*.jpg *.tif *.png)")

        if not self.image_path:
            logging.info("\n 图片上传失败")
            self.is_input = False
            self.UploadLabel.setPixmap(QPixmap('./ui/icon/upload_label_icon.png'))
        else:
            logging.info("\n 图片上传成功, 点击开始即可超分")
            self.is_input = True
            self.UploadLabel.setPixmap(self.resize_img(QPixmap(self.image_path), 'upload'))
            self.box.set_image(None)
            self.box.point = QPoint(0, 0)
            self.box.repaint()

    def sr_image(self):
        if self.ip_address == '':
            logging.info('\n 请先输入服务器地址！\n')
            return
        if self.is_input is False:
            logging.info('\n 请先上传图片！\n')
            return

        logging.info('\n 超分辨计算中，请稍等！\n')
        if SRApi(self.ip_address, self.image_path, self.save).main():
            self.image_save = self.save + 'SR_X4/' + os.path.basename(self.image_path) + '.png'
            self.box.SR_flag = True
            self.box.set_image(self.image_save)
            self.box.point = QPoint(0, 0)
            self.box.repaint()
            logging.info('\n 已显示4倍超分辨结果\n')
            self.is_sr = True
        else:
            logging.info('\n 服务器没有响应！\n')

    def change_display(self, opt = 'X2'):
        # self.is_sr = True
        if self.is_sr is False:
            logging.info("\n 请先超分辨图片再查看！\n")
            return
        if opt == 'X2':
            self.image_save = self.save + 'SR_X2/' + os.path.basename(self.image_path) + '.png'
            logging.info('\n 切换到 2 倍结果\n')
        elif opt == 'X4':
            self.image_save = self.save + 'SR_X4/' + os.path.basename(self.image_path) + '.png'
            logging.info('\n 切换到 4 倍结果\n')
        else:
            self.image_save = self.image_path
            logging.info('\n 查看原图 \n')

        self.box.SR_flag = True
        self.box.scale = 1
        self.box.set_image(self.image_save)
        self.box.point = QPoint(0, 0)
        self.box.repaint()

    def resize_img(self, img, opt = 'upload'):
        if opt == 'upload':
            size = 364
        else:
            size = 900
        if img.width() >= size or img.height() >= size:
            img = img.scaled(size, size, Qt.KeepAspectRatio, Qt.FastTransformation)
        return img

    def load_zoom(self):
        self.box.set_image(self.image_save)
        self.repaint()
        if 'X2' in str(self.image_save):
            logging.info('\n 已加载 2倍 超分辨图像 \n')
        elif 'X4' in str(self.image_save):
            logging.info('\n 已加载 4倍 超分辨图像 \n')
        else:
            logging.info('\n 已加载原图 \n')

    def large_click(self):
        logging.info('\n 已放大 \n')
        if self.box.scale < 3:
            self.box.scale += 0.1
            self.box.adjustSize()
            self.update()

    def small_click(self):
        logging.info('\n 已缩小 \n')
        if self.box.scale > 0.1:
            self.box.scale -= 0.2
            self.box.adjustSize()
            self.update()

    def save_image(self):
        if self.image_save == self.image_path:
            logging.info('\n 请保存超分后的图像 \n')
            return False
        filename = QFileDialog.getSaveFileName(self, 'save image', '/', '*.tif *.png *.jpg *.jpeg')
        img = cv2.imread(self.image_save)
        try:
            cv2.imwrite(filename[0], img)
            if 'X2' in str(self.image_save):
                logging.info('\n 已保存 2倍 超分辨图像 \n')
            else:
                logging.info('\n 已保存 4倍 超分辨图像 \n')
        except:
            logging.info('\n 取消保存 \n')
            return False


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = MainWin()
    sys.exit(app.exec_())