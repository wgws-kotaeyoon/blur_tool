import sys
#import pyexiv2
import PyQt5
import imutils
import cv2
import glob
import numpy as np
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image , ImageOps
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import uic, QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox




calUI = './cal.ui'
k = 0
count = 0
class MainDialog(QDialog):

    def __init__(self):
        QDialog.__init__(self,None)
        self.msg = QMessageBox()
        self.msg.setWindowTitle("EXIT")
        self.msg.setText("End of File , Open New Folder")
        uic.loadUi(calUI, self)
        self.OPEN.clicked.connect(lambda: self.image_list())
        #self.listWidget.itemDoubleClicked.connect(lambda: self.loadImage(0))
        self.listWidget.itemDoubleClicked.connect(lambda: self.dragImage())
        self.SAVE.clicked.connect(lambda: self.image_save())

        #self.NEXT.clicked.connect(lambda: self.loadImage(1))
        self.AUTO.clicked.connect(lambda: self.auto_progress())
        self.DRAG.clicked.connect(lambda: self.dragImage())
        #QtCore.QMetaObject.connectSlotsByName(QDialog)

    def image_list(self):
        self.listWidget.clear()
        self.AUTO.setEnabled(True)
        dir_Name = QFileDialog.getExistingDirectory() #폴더명 가져오기
        if dir_Name:
            self.filepath = dir_Name
            filenames = os.listdir(dir_Name)
            self.count = 0
            filenames = [file for file in filenames if file.endswith((".png", ".PNG", ".JPG", ".jpg"))] #이미지 파일 필터
            for filename in filenames:
                self.listWidget.insertItem(self.count, filename) #위젯에 추가
                self.count+=1
        else:
            self.msg.setText("Please try again. Image list not found")
            self.msg.exec_()
            self.image_list()

    def detect_and_predict_mask(self,frame, faceNet, maskNet):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
            (104.0, 177.0, 123.0))

        faceNet.setInput(blob)
        detections = faceNet.forward()

        faces = []
        locs = []
        preds = []

        for i in range(0, detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                faces.append(face)
                locs.append((startX, startY, endX, endY))

        if len(faces) > 0:
            faces = np.array(faces, dtype="float32")
            preds = maskNet.predict(faces, batch_size=32)

        return (locs, preds)


    #이미지 불러오기
    def loadImage(self,i):
        try:
            #index = self.listWidget.currentRow()
            #global k
            # if i == 1:
            #     k += 1
            #     index += k
            # elif i == 0:
            #     k = 0
            if i == self.count:
                return
            self.filename = self.listWidget.item(i).text()
            #k += 1
            #self.filename = self.listWidget.item(index).text()
            self.current_image.setText(self.filename)
            path = os.path.join(self.filepath, self.filename).replace('/', '\\')
            img_array = np.fromfile(path, np.uint8)
            self.image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            self.ex_img = Image.open(path)

            self.setPhoto(self.image)
            ksize = 30 # 블러 강도

            prototxtPath = "face_detector\\deploy.prototxt"
            weightsPath = "face_detector\\res10_300x300_ssd_iter_140000.caffemodel"
            faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
            maskNet = load_model("mask_detector.model")
            #이미지 사이즈
            frame = imutils.resize(self.image, width=1960, height= 1080)
            
            (locs, preds) = self.detect_and_predict_mask(frame, faceNet, maskNet)
            #블러처리 되지 않은 파일 메모장 저장
            if not locs:
                print("No Blur Image")
                File = open("No_Blur_List.txt", "a")
                File.write(path.split("\\")[-1] + "\n")
                File.close()

            for (box, pred) in zip(locs, preds):
                #박스처리
                (startX, startY, endX, endY) = box
                w = endX - startX
                h = (endY-150) - (startY+50)
                roi = frame[startY+50:startY+50 + h, startX:startX + w]  # 관심영역 지정
                roi = cv2.blur(roi, (ksize, ksize))  # 블러(모자이크) 처리
                frame[startY+50:startY+50 + h, startX:startX + w] = roi 

            self.image = frame
            self.setPhoto(self.image)
            cv2.destroyAllWindows()
                ###clear def
                # if k == self.count:
                #     self.msg.exec_()
                #     self.AUTO.setAutoRepeat(False)
                #     self.AUTO.setDisabled(True)
                #     ##self.label.clear()
                #     ##self.current_image.setText("")
                #     self.listWidget.clear()
                #     k = 0

        except:
            self.msg.setText("Open Click!!")
            self.msg.exec_()

    #이미지 추출
    def setPhoto(self, image):
        self.tmp = image
        image = imutils.resize(image, width=640)
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(image))
        self.image_save()

    #이미지 저장
    def image_save(self):

        if os.path.exists(os.getcwd() + "\\save\\"):
            print('저장되었습니다.')
            save_filename = self.filename[0:-4] + '_blurred.jpg'
            path = os.getcwd() + "\\save\\" + save_filename
            cv2.imwrite(path, self.image)

            output_img = Image.open(path)
            exif_dict = self.ex_img.info.get('exif')
            
            if exif_dict:
                EXIF = self.ex_img.info['exif']
                output_img = output_img.transpose(Image.ROTATE_90)
                output_img.save(path, exif=EXIF,dpi=(300,300))
            else: ## no exif / memo filename to txt ?
                output_img.save(path,dpi=(300,300))


        else:
            os.makedirs(os.getcwd() + "\\save\\")
            print('저장되었습니다.')
            save_filename = self.filename[0:-4] + '_blur.jpg'
            path = os.getcwd() + "\\save\\" + save_filename
            cv2.imwrite(path, self.image)

    #auto
    def auto_progress(self):
        global k
        self.AUTO.setAutoRepeat(True)
        self.AUTO.animateClick()
        self.loadImage(k)
        k += 1
        if k == self.count:
            # self.AUTO.setAutoRepeat(False)
            # self.AUTO.setDisabled(True)
            # ##self.label.clear()
            # ##self.current_image.setText("")
            self.AUTO.setEnabled(False)
            QTimer.singleShot(1000,lambda: self.AUTO.setDisabled(False))
            self.msg.exec_()
            self.listWidget.clear()
            k = 0

    #드레그 모드
    def dragImage(self):
        try:
            index = self.listWidget.currentRow()
            self.filename = self.listWidget.item(index).text()
            self.current_image.setText(self.filename)
            path = os.path.join(self.filepath, self.filename).replace('/', '\\')
            img_array = np.fromfile(path, np.uint8)
            self.image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            # self.image = cv2.imread(path) #이미지 로드
            #self.image = cv2.resize(self.image , (800,1200), interpolation = cv2.INTER_LINEAR)

            self.ex_img = Image.open(path)
            self.setPhoto(self.image)
            ksize = 30 # 블러 강도
            while True:
                cv2.namedWindow("title",cv2.WINDOW_NORMAL)
                cv2.resizeWindow("title",800,1200)
                x, y, w, h = cv2.selectROI("title", self.image, fromCenter=False, showCrosshair=True)  # 관심영역 선택
                if w > 0 and h > 0:  # 폭과 높이가 음수이면 드래그 방향이 옳음
                    roi = self.image[y:y + h, x:x + w]  # 관심영역 지정
                    roi = cv2.blur(roi, (ksize, ksize))  # 블러(모자이크) 처리
                    self.image[y:y + h, x:x + w] = roi  # 원본 이미지에 적용
                else:
                    break
            self.setPhoto(self.image)
            cv2.destroyAllWindows()
        except:
            self.msg.setText("Open Click!!")
            self.msg.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_dialog = MainDialog()
    main_dialog.show()
    app.exec_()
