import sys
import os
import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel,
    QFileDialog, QHBoxLayout, QVBoxLayout
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

# config
from utils.config import opt
# dataset
from data.dataset import Dataset
from data.voc_dataset import VOC_BBOX_LABEL_NAMES
from data.util import VOC_COLOR_LIST
# model
from model import FPNFasterRCNNVGG16
# utils
from utils import array_tool as at
import xml.etree.ElementTree as ET

class Demo(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Demo")
        self.resize(1600, 800)

        # 初始變數
        self.dataset = None
        self.index = 0
        self.net = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.voc_folder = None

        # GUI 元素
        self.gt_label = QLabel("Ground Truth")
        self.gt_label.setAlignment(Qt.AlignHCenter)
        self.gt_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        self.gt_img = QLabel()
        self.gt_img.setScaledContents(True)
        self.gt_img.setMaximumSize(800, 800)

        self.test_label = QLabel("Test")
        self.test_label.setAlignment(Qt.AlignHCenter)
        self.test_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        self.test_img = QLabel()
        self.test_img.setScaledContents(True)
        self.test_img.setMaximumSize(800, 800)

        self.load_data_btn = QPushButton("Load Data")
        self.load_model_btn = QPushButton("Load Model")
        self.prev_btn = QPushButton("Previous")
        self.next_btn = QPushButton("Next")

        # 調整按鈕大小
        for btn in [self.load_model_btn, self.load_data_btn, self.prev_btn, self.next_btn]:
            btn.setFixedHeight(50)
            btn.setFixedWidth(150)

        # Layout
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.gt_label)
        left_layout.addWidget(self.gt_img)

        middle_layout = QVBoxLayout()
        middle_layout.addWidget(self.test_label)
        middle_layout.addWidget(self.test_img)

        right_layout = QVBoxLayout()
        right_layout.addStretch()
        right_layout.addWidget(self.load_model_btn)
        right_layout.addWidget(self.load_data_btn)
        right_layout.addWidget(self.prev_btn)
        right_layout.addWidget(self.next_btn)
        right_layout.addStretch()

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(middle_layout, 1)
        main_layout.addLayout(right_layout, 0)
        self.setLayout(main_layout)

        # Signals
        self.load_data_btn.clicked.connect(self.load_data)
        self.load_model_btn.clicked.connect(self.load_model)
        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn.clicked.connect(self.next_image)

    def load_data(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Test Data Folder", "./data")
        if folder:
            voc_folder = os.path.join(folder, "VOCdevkit", "VOC2007")
            if not os.path.exists(voc_folder):
                print("Invalid VOC folder structure")
                return
            self.voc_folder = voc_folder
            self.dataset = Dataset(opt, mode='vis')
            self.index = 0
            self.update_images()
            print(f"Loaded dataset from {folder}")

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Model File", "./save/exp")
        if path:
            if not os.path.exists(path):
                print("Invalid model file path")
                return
            self.net = FPNFasterRCNNVGG16(n_fg_class=10).to(self.device)
            self.net.load_state_dict(torch.load(path))
            self.net.eval()
            print(f"Loaded model: {path}")

    def prev_image(self):
        if self.dataset and self.index > 0:
            self.index -= 1
            self.update_images()

    def next_image(self):
        if self.dataset and self.index < len(self.dataset) - 1:
            self.index += 1
            self.update_images()

    def update_images(self):
        if self.dataset is None or self.net is None:
            return

        # Dataset 輸出
        ori_img, trans_img, scale, ori_size = self.dataset[self.index]

        # Tensor batch
        trans_img = torch.from_numpy(trans_img).unsqueeze(0).float().to(self.device)
        scale = float(scale)
        original_size = list(ori_size)

        # Ground Truth
        anno_dir = os.path.join(self.voc_folder, "Annotations")
        img_dir = os.path.join(self.voc_folder, "JPEGImages")

        image_files = sorted(os.listdir(img_dir))
        img_name = image_files[self.index]
        xml_name = os.path.splitext(img_name)[0] + ".xml"
        xml_file = os.path.join(anno_dir, xml_name)

        gt_img = ori_img.copy()
        gt_img = np.transpose(gt_img, (1,2,0)).astype(np.uint8)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR)

        if os.path.exists(xml_file):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for obj in root.findall("object"):
                class_name = obj.find("name").text
                bndbox = obj.find("bndbox")
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)
                color = VOC_COLOR_LIST[VOC_BBOX_LABEL_NAMES.index(class_name)]
                cv2.rectangle(gt_img, (xmin, ymin), (xmax, ymax), color, 2)
                label = f"{class_name}: 1.0"
                text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                pl = (xmin, max(0, ymin - text_size[1]))
                cv2.rectangle(gt_img, (pl[0]-1, pl[1]-baseline), (pl[0]+text_size[0], pl[1]+text_size[1]), color, -1)
                cv2.putText(gt_img, label, (pl[0], pl[1]+baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        # Test 推論
        with torch.no_grad():
            pred_bboxes, pred_labels, pred_scores = self.net(trans_img, None, None, scale, original_size)

        test_img = ori_img.copy()
        test_img = np.transpose(test_img, (1,2,0)).astype(np.uint8)
        test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)

        for bbox, label, score in zip(pred_bboxes, pred_labels, pred_scores):
            if score < 0.5:
                continue
            class_name = VOC_BBOX_LABEL_NAMES[label]
            bbox = bbox.astype(np.int32)
            ymin, xmin, ymax, xmax = bbox
            color = VOC_COLOR_LIST[VOC_BBOX_LABEL_NAMES.index(class_name)]
            cv2.rectangle(test_img, (xmin, ymin), (xmax, ymax), color, 2)
            text = f"{class_name}: {round(score,2):.2f}"  # 顯示到 0.xx
            text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            pl = (xmin, ymin - text_size[1])
            cv2.rectangle(test_img, (pl[0]-1, pl[1]-baseline), (pl[0]+text_size[0], pl[1]+text_size[1]), color, -1)
            cv2.putText(test_img, text, (pl[0], pl[1]+baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        # 顯示
        self.set_pixmap(self.gt_img, gt_img)
        self.set_pixmap(self.test_img, test_img)

    def set_pixmap(self, label, img):
        height, width, ch = img.shape
        bytes_per_line = ch * width
        qimg = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qimg)
        label.setPixmap(pixmap)
        label.setScaledContents(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    demo = Demo()
    demo.show()
    sys.exit(app.exec_())
