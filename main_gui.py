import sys
import cv2
import numpy as np
import time
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from YOLO.python.darknet import *

import torch
import torch.backends.cudnn as cudnn
from RFBNet.data import BaseTransform, VOC_300, VOC_512
from RFBNet.layers.functions import Detect, PriorBox
from RFBNet.models.RFB_Net_vgg import build_net
from collections import OrderedDict


class ObjectDetection(QtWidgets.QMainWindow):
    def __init__(self):
        super(ObjectDetection, self).__init__()
        self.setWindowTitle('Object Detection Demo Program')
        self.algorithm_select()
        self.init_main_window()

    def init_main_window(self):
        self.resize(1800, 900)
        self.setWindowIcon(QtGui.QIcon(r'icon.png'))
        self.file_item = QtWidgets.QAction('Open image', self)
        self.file_item.triggered.connect(self.select_file)

        self.label = DragLabel('Please drag image here\nor\nPress Ctrl+O to select', self)
        self.label.addAction(self.file_item)
        self.setCentralWidget(self.label)
        self.init_menu()
        self.init_status_bar()

    def init_menu(self):
        menubar = self.menuBar()
        open_item = QtWidgets.QAction('Open', self)
        open_item.setShortcut('Ctrl+O')
        open_item.triggered.connect(self.select_file)

        exit_item = QtWidgets.QAction('Exit', self)
        exit_item.setShortcut('Ctrl+Q')
        exit_item.triggered.connect(QtWidgets.qApp.quit)

        file_menu = menubar.addMenu('File')
        file_menu.addAction(open_item)
        file_menu.addAction(exit_item)

        rfbnet_300_item = QtWidgets.QAction('RFBNet 300', self)
        rfbnet_300_item.triggered.connect(self.init_model)
        rfbnet_512_item = QtWidgets.QAction('RFBNet 512', self)
        rfbnet_512_item.triggered.connect(self.init_model)
        yolo_v1_item = QtWidgets.QAction('Yolo v1', self)
        yolo_v1_item.triggered.connect(self.init_model)
        yolo_v2_item = QtWidgets.QAction('Yolo v2', self)
        yolo_v2_item.triggered.connect(self.init_model)
        yolo_v3_item = QtWidgets.QAction('Yolo v3', self)
        yolo_v3_item.triggered.connect(self.init_model)
        algorithm_menu = menubar.addMenu('Change algorithm')
        algorithm_menu.addAction(rfbnet_300_item)
        algorithm_menu.addAction(rfbnet_512_item)
        algorithm_menu.addAction(yolo_v1_item)
        algorithm_menu.addAction(yolo_v2_item)
        algorithm_menu.addAction(yolo_v3_item)

    def init_status_bar(self):
        self.detectTimeLabel = QLabel('Ready')
        self.filePathLabel = QLabel()
        self.statusBar().addWidget(self.detectTimeLabel)
        self.statusBar().addPermanentWidget(self.filePathLabel)

    def algorithm_select(self):
        gridLayout = QGridLayout()
        infoPanel = QLabel('Please choose the algorithm:')

        self.dialog = QDialog()
        self.dialog.setWindowTitle('Algorithm Select')
        self.dialog.setFont(QtGui.QFont('Ubuntu Mono', 14))

        gridBox1 = QGridLayout()
        rfb_label = QLabel('RFBNet: ')
        rfbnet_300 = QPushButton('300 x 300')
        rfbnet_300.clicked.connect(self.init_model)
        rfbnet_512 = QPushButton('512 x 512')
        rfbnet_512.clicked.connect(self.init_model)
        gridBox1.addWidget(rfb_label, 0, 0)
        gridBox1.addWidget(rfbnet_300, 0, 2)
        gridBox1.addWidget(rfbnet_512, 0, 3)

        gridBox2 = QGridLayout()
        yolo_label = QLabel('YOLO: ')
        yolo_v1 = QPushButton('v1')
        yolo_v1.clicked.connect(self.init_model)
        yolo_v2 = QPushButton('v2')
        yolo_v2.clicked.connect(self.init_model)
        yolo_v3 = QPushButton('v3')
        yolo_v3.clicked.connect(self.init_model)
        gridBox2.addWidget(QLabel(), 0, 0)
        gridBox2.addWidget(yolo_label, 0, 1)
        gridBox2.addWidget(QLabel(), 0, 2)
        gridBox2.addWidget(QLabel(), 0, 3)
        gridBox2.addWidget(QLabel(), 0, 4)
        gridBox2.addWidget(yolo_v1, 0, 5)
        gridBox2.addWidget(yolo_v2, 0, 6)
        gridBox2.addWidget(yolo_v3, 0, 7)

        hbox = QHBoxLayout()
        cancelButton = QPushButton('Cancel')
        cancelButton.clicked.connect(self.cancel)
        hbox.addStretch(1)
        hbox.addWidget(cancelButton)

        gridLayout.addWidget(infoPanel, 0, 0)
        gridLayout.addWidget(QLabel(), 1, 0)
        gridLayout.addLayout(gridBox1, 2, 0)
        gridLayout.addLayout(gridBox2, 3, 0)
        gridLayout.addLayout(hbox, 4, 0)
        self.dialog.setLayout(gridLayout)

        self.dialog.setWindowModality(Qt.ApplicationModal)
        self.dialog.exec_()

    def cancel(self):
        exit(0)

    def init_model(self):
        sender = self.sender().text()
        if hasattr(self, 'label'):
            self.label.leftLabel.clear()
            self.label.rightLabel.clear()
            self.detectTimeLabel.setText('Ready')
            self.filePathLabel.clear()
            self.label.setText('Please drag image here\nor\nPress Ctrl+O to select')
        if sender == '300 x 300' or sender == 'RFBNet 300':
            self.algorithm = 'rfbnet'
            self.init_rfbnet('300')
            self.setWindowTitle('Object Detection Demo Program -- RFBNet 300')
        elif sender == '512 x 512' or sender == 'RFBNet 512':
            self.algorithm = 'rfbnet'
            self.init_rfbnet('512')
            self.setWindowTitle('Object Detection Demo Program -- RFBNet 512')
        elif sender == 'v1' or sender == 'Yolo v1':
            self.algorithm = 'yolo'
            self.setWindowTitle('Object Detection Demo Program -- YOLO v1')
            print('v1')
            # self.net = load_net('YOLO/NWPU/yolov1-voc.cfg'.encode('utf-8'),
            #                     'weights/yolov1_NWPU.weights'.encode('utf-8'), 0)
            # self.meta = load_meta('YOLO/NWPU/voc.data'.encode('utf-8'))
        elif sender == 'v2' or sender == 'Yolo v2':
            self.algorithm = 'yolo'
            self.setWindowTitle('Object Detection Demo Program -- YOLO v2')
            self.net = load_net('YOLO/NWPU/yolov2-voc.cfg'.encode('utf-8'),
                                'weights/yolov2_NWPU.weights'.encode('utf-8'), 0)
            self.meta = load_meta('YOLO/NWPU/voc.data'.encode('utf-8'))
        elif sender == 'v3' or sender == 'Yolo v3':
            self.algorithm = 'yolo'
            self.setWindowTitle('Object Detection Demo Program -- YOLO v3')
            self.net = load_net('YOLO/NWPU/yolov3-voc.cfg'.encode('utf-8'),
                                'weights/yolov3_NWPU.weights'.encode('utf-8'), 0)
            self.meta = load_meta('YOLO/NWPU/voc.data'.encode('utf-8'))
        self.dialog.close()
        print('Finished loading model!')

    def init_rfbnet(self, img_size):
        if img_size == '300':
            self.input_size = 300
            self.cfg = VOC_300
            self.trained_model = 'weights/RFB_vgg_NWPU_300.pth'
        elif img_size == '512':
            self.input_size = 512
            self.cfg = VOC_512
            self.trained_model = 'weights/RFB_vgg_NWPU_512.pth'

        self.priorbox = PriorBox(self.cfg)
        self.cuda = True
        self.numclass = 21
        self.net = build_net('test', self.input_size, self.numclass)  # initialize detector
        state_dict = torch.load(self.trained_model)

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:]
            else:
                name = k
            new_state_dict[name] = v
        self.net.load_state_dict(new_state_dict)
        self.net.eval()
        if self.cuda:
            self.net = self.net.cuda()
            cudnn.benchmark = True
        else:
            self.net = self.net.cpu()

    def select_file(self):
        file_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Select image',
                                                          r'/home/zcy/data/NWPU_VHR-10_dataset/positive_image_set',
                                                          'Image files(*.bmp *.jpg *.pbm *.pgm *.png *.ppm *.xbm *.xpm)'
                                                          ';;All files (*.*)')
        try:
            self.detect_img(file_path[0].strip())
        except Exception as e:
            QtWidgets.QMessageBox.information(self, "Alert", str(e))

    def detect_img(self, file_path):
        print(file_path)
        start_time = time.time()
        img = cv2.imread(file_path)
        if not hasattr(self, 'algorithm'):
            QtWidgets.QMessageBox.information(self, 'Alert', 'Please choose algorithm first.')
            return
        if self.algorithm == 'yolo':
            results = detect(self.net, self.meta, file_path.encode('utf-8'))
            end_time = time.time()
            self.detectTimeLabel.setText('Detect Time: ' + str(end_time - start_time) + ' s')
            self.filePathLabel.setText('File Path: ' + file_path)
            for result in results:
                cv2.rectangle(img, (round(result[2][0] - result[2][2] / 2),
                                    round(result[2][1] - result[2][3] / 2),
                                    round(result[2][2]),
                                    round(result[2][3])), (255, 0, 0), 2)
                font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                cv2.putText(img, str(result[0], encoding='utf-8'), (round(result[2][0] - result[2][2] / 2 - 5),
                                                                    round(result[2][1] - result[2][3] / 2 - 5)), font,
                            1, (255, 0, 0), 2)
            cv2.imwrite('test_result.png', img)
        elif self.algorithm == 'rfbnet':
            classes = ['__background__', 'aeroplane', 'ship', 'storage_tank', 'baseball_diamond',
                       'tennis_court', 'basketball_court', 'ground_track_field',
                       'harbor', 'bridge', 'vehicle', '', '', '', '', '', '', '', '', '', '']
            if img is None:
                QtWidgets.QMessageBox.information(self, 'Alert', 'Please select images')
                return
            scale = torch.Tensor([img.shape[1], img.shape[0],
                                  img.shape[1], img.shape[0]])
            detector = Detect(self.numclass, 0, self.cfg)
            transform = BaseTransform(self.net.size, (123, 117, 104), (2, 0, 1))
            with torch.no_grad():
                x = transform(img).unsqueeze(0)
                if self.cuda:
                    x = x.cuda()
                    scale = scale.cuda()
            out = self.net(x)
            with torch.no_grad():
                priors = self.priorbox.forward()
                if self.cuda:
                    priors = priors.cuda()
            boxes, scores = detector.forward(out, priors)
            boxes = boxes[0]
            scores = scores[0]
            boxes *= scale
            boxes = boxes.cpu().numpy()
            scores = scores.cpu().numpy()

            for j in range(1, self.numclass):
                inds = np.where(scores[:, j] > 0.1)[0]  # conf > 0.6
                if inds is None:
                    continue
                c_bboxes = boxes[inds]
                c_scores = scores[inds, j]
                c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)
                keep = self.rfbnet_nms_py(c_dets, 0.6)
                c_dets = c_dets[keep, :]
                c_bboxes = c_dets[:, :4]
                font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                for bbox in c_bboxes:
                    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
                    cv2.putText(img, classes[j], (int(bbox[0] - 5), int(bbox[1]) - 5), font, 1, (255, 0, 0), 2)
            cv2.imwrite('test_result.png', img)

            end_time = time.time()
            self.detectTimeLabel.setText('Detect Time: ' + str(end_time - start_time) + ' s')
            self.filePathLabel.setText('File Path: ' + file_path)
        else:
            QtWidgets.QMessageBox.information(self, 'Alert', 'Please choose algorithm first.')
            return

        detect_result = QtGui.QPixmap('test_result.png')
        original_img = QtGui.QPixmap(file_path)
        self.label.setText("")
        self.label.leftLabel.setPixmap(original_img)
        self.label.rightLabel.setPixmap(detect_result)
        self.setFocus()

    def rfbnet_nms_py(self, dets, thresh):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        ndets = dets.shape[0]
        suppressed = np.zeros((ndets), dtype=np.int)
        keep = []
        for _i in range(ndets):
            i = order[_i]
            if suppressed[i] == 1:
                continue
            keep.append(i)
            ix1 = x1[i]
            iy1 = y1[i]
            ix2 = x2[i]
            iy2 = y2[i]
            iarea = areas[i]
            for _j in range(_i + 1, ndets):
                j = order[_j]
                if suppressed[j] == 1:
                    continue
                xx1 = max(ix1, x1[j])
                yy1 = max(iy1, y1[j])
                xx2 = min(ix2, x2[j])
                yy2 = min(iy2, y2[j])
                w = max(0.0, xx2 - xx1 + 1)
                h = max(0.0, yy2 - yy1 + 1)
                inter = w * h
                ovr = inter / (iarea + areas[j] - inter)
                if ovr >= thresh:
                    suppressed[j] = 1
        return keep


class DragLabel(QLabel):
    def __init__(self, text, parent):
        super().__init__(text, parent)
        self.parent = parent
        self.setAcceptDrops(True)
        self.setFont(QtGui.QFont('Ubuntu Mono', 30))
        self.setAlignment(Qt.AlignCenter)

        self.leftLabel = QLabel()
        self.rightLabel = QLabel()

        inputLabelLayout = QHBoxLayout()
        inputLabelLayout.addWidget(self.leftLabel)
        inputLabelLayout.addWidget(self.rightLabel)

        self.setLayout(inputLabelLayout)

    def dragEnterEvent(self, QDragEnterEvent):
        QDragEnterEvent.accept()

    def dropEvent(self, QDropEvent):
        self.parent.detect_img(QDropEvent.mimeData().text()[7:].strip())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    objectDetection = ObjectDetection()
    objectDetection.show()
    app.exit(app.exec_())
