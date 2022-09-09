from PyQt5 import QtGui, uic, QtWidgets
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
import socket
from imutils.video import VideoStream
import imagezmq
import threading

class VideoStreamSubscriber:
    def __init__(self, hostname, port):
        self.hostname = hostname
        self.port = port
        self._stop = False
        self._data_ready = threading.Event()
        self._thread = threading.Thread(target=self._run, args=())
        self._thread.daemon = True
        self._thread.start()

    def receive(self, timeout=15.0):
        flag = self._data_ready.wait(timeout=timeout)
        if not flag:
            raise TimeoutError(
                "Timeout while reading from subscriber tcp://{}:{}".format(self.hostname, self.port))
        self._data_ready.clear()
        return self._data

    def _run(self):
        receiver = imagezmq.ImageHub("tcp://{}:{}".format(self.hostname, self.port), REQ_REP=False)
        receiver.zmq_socket.set_hwm(1)
        #receiver.zmq_socket.setsockopt(imagezmq.zmq.CONFLATE, 1)
        receiver.zmq_socket.setsockopt(imagezmq.zmq.RCVHWM,10)
        #sender.zmq_socket.setsockopt(imagezmq.zmq.SNDHWM,10)
        while not self._stop:
            self._data = receiver.recv_jpg()
            self._data_ready.set()
        receiver.close()

    def close(self):
        self._stop = True

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, ip_device, port):
        super().__init__()
        self._run_flag = True
        self.ip_device = ip_device
        self.port = port

    def run(self):
        # capture from web cam
        hostname =  self.ip_device  # Use to receive from localhost
        port = self.port
        receiver = VideoStreamSubscriber(hostname, port)
        while self._run_flag:
            msg, frame = receiver.receive()
            cv_img = cv2.imdecode(np.frombuffer(frame, dtype='uint8'), -1)
            self.change_pixmap_signal.emit(cv_img)
        # shut down capture system

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


class App(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('gait_win.ui', self)
        self.disply_width = 150
        self.display_height = 450

        
        self.button_jetson4 = self.findChild(QtWidgets.QPushButton, 'jetson4_button_true')
        self.button_jetson4.setVisible(False)
        self.button_NoJetson4 = self.findChild(QtWidgets.QPushButton, 'jetson4_button_false')
        self.button_NoJetson4.setVisible(True)
        self.button_NoJetson4.clicked.connect(self.jetson4_off_msg)
        self.button_jetson4.clicked.connect(self.jetson4_on_msg)

        self.button_jetson3 = self.findChild(QtWidgets.QPushButton, 'jetson3_button_true')
        self.button_jetson3.setVisible(False)
        self.button_NoJetson3 = self.findChild(QtWidgets.QPushButton, 'jetson3_button_false')
        self.button_NoJetson3.setVisible(True)
        self.button_NoJetson3.clicked.connect(self.jetson3_off_msg)
        self.button_jetson3.clicked.connect(self.jetson3_on_msg)
        
        self.button_jetson9 = self.findChild(QtWidgets.QPushButton, 'jetson9_button_true')
        self.button_jetson9.setVisible(False)
        self.button_NoJetson9 = self.findChild(QtWidgets.QPushButton, 'jetson9_button_false')
        self.button_NoJetson9.setVisible(True)
        self.button_NoJetson9.clicked.connect(self.jetson9_off_msg)
        self.button_jetson9.clicked.connect(self.jetson9_on_msg)

        self.button_jetson5 = self.findChild(QtWidgets.QPushButton, 'jetson5_button_true')
        self.button_jetson5.setVisible(False)
        self.button_NoJetson5 = self.findChild(QtWidgets.QPushButton, 'jetson5_button_false')
        self.button_NoJetson5.setVisible(True)
        self.button_NoJetson5.clicked.connect(self.jetson5_off_msg)
        self.button_jetson5.clicked.connect(self.jetson5_on_msg)
        
        self.button_jetson6 = self.findChild(QtWidgets.QPushButton, 'jetson6_button_true')
        self.button_jetson6.setVisible(False)
        self.button_NoJetson6 = self.findChild(QtWidgets.QPushButton, 'jetson6_button_false')
        self.button_NoJetson6.setVisible(True)
        self.button_NoJetson6.clicked.connect(self.jetson6_off_msg)
        self.button_jetson6.clicked.connect(self.jetson6_on_msg)
        
        
        self.image_label = self.findChild(QtWidgets.QLabel, 'device1')
        self.image_label2 = self.findChild(QtWidgets.QLabel, 'device2')
        self.image_label3 = self.findChild(QtWidgets.QLabel, 'device3')
        self.image_label4 = self.findChild(QtWidgets.QLabel, 'device4')
        self.image_label5 = self.findChild(QtWidgets.QLabel, 'device5')
        
        

        # create the video capture thread
        
        
        # Jetson 4
        #self.thread = VideoThread("192.168.1.207", 5555)
        
        self.thread = VideoThread("192.168.0.108", 5555)
        
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image1)
        # start the thread
        self.thread.start()
        
        
        # create the video capture thread
        # Jetson 3
        self.thread2 = VideoThread("192.168.0.160", 5555)
        # connect its signal to the update_image slot
        self.thread2.change_pixmap_signal.connect(self.update_image2)
        # start the thread
        self.thread2.start()
    
        # create the video capture thread
        # Jetson 9
        self.thread3 = VideoThread("192.168.0.123", 5555)
        # connect its signal to the update_image slot
        self.thread3.change_pixmap_signal.connect(self.update_image3)
        # start the thread
        self.thread3.start()

        # create the video capture thread
        # Jetson 5
        self.thread4 = VideoThread("192.168.0.139", 5555)
        # connect its signal to the update_image slot
        self.thread4.change_pixmap_signal.connect(self.update_image4)
        # start the thread
        self.thread4.start()
        
        # create the video capture thread
        # Jetson 6
        #self.thread5 = VideoThread("192.168.0.130", 5555)
        self.thread5 = VideoThread("192.168.1.208", 5555)
        # connect its signal to the update_image slot
        self.thread5.change_pixmap_signal.connect(self.update_image5)
        # start the thread
        self.thread5.start()

        

    
    def jetson4_off_msg(self):
        self.button_jetson4.setVisible(True)
        self.button_NoJetson4.setVisible(False)
        self.thread.hide()
        
    def jetson4_on_msg(self):
        self.button_jetson4.setVisible(False)
        self.button_NoJetson4.setVisible(True)
        self.thread.show()
        

    def jetson3_off_msg(self):
        self.button_jetson3.setVisible(True)
        self.button_NoJetson3.setVisible(False)
        self.image_label2.hide()
        
    def jetson3_on_msg(self):
        self.button_jetson3.setVisible(False)
        self.button_NoJetson3.setVisible(True)
        self.image_label2.show()


    def jetson9_off_msg(self):
        self.button_jetson9.setVisible(True)
        self.button_NoJetson9.setVisible(False)
        self.image_label3.hide()
        
    def jetson9_on_msg(self):
        self.button_jetson9.setVisible(False)
        self.button_NoJetson9.setVisible(True)
        self.image_label3.show()    

    def jetson5_off_msg(self):
        self.button_jetson5.setVisible(True)
        self.button_NoJetson5.setVisible(False)
        self.image_label4.hide()
        
    def jetson5_on_msg(self):
        self.button_jetson5.setVisible(False)
        self.button_NoJetson5.setVisible(True)
        self.image_label4.show()

    

    def jetson6_off_msg(self):
        self.button_jetson6.setVisible(True)
        self.button_NoJetson6.setVisible(False)
        self.image_label5.hide()
        
    def jetson6_on_msg(self):
        self.button_jetson6.setVisible(False)
        self.button_NoJetson6.setVisible(True)
        self.image_label5.show()  

    

    def closeEvent(self, event):
        self.thread.stop()
        self.thread2.stop()
        self.thread3.stop()
        self.thread4.stop()
        self.thread5.stop()
        
        
        event.accept()

    
    @pyqtSlot(np.ndarray)
    def update_image1(self, cv_img):
        #Updates the image_label with a new opencv image
        qt_img = self.convert_cv_qt(cv_img)
        self.device1.setPixmap(qt_img)
    
    @pyqtSlot(np.ndarray)
    def update_image2(self, cv_img):
        #Updates the image_label with a new opencv image
        qt_img = self.convert_cv_qt(cv_img)
        self.device2.setPixmap(qt_img) 
    
    @pyqtSlot(np.ndarray)
    def update_image3(self, cv_img):
        #Updates the image_label with a new opencv image
        qt_img = self.convert_cv_qt(cv_img)
        self.device3.setPixmap(qt_img)

    @pyqtSlot(np.ndarray)
    def update_image4(self, cv_img):
        #Updates the image_label with a new opencv image
        cv_resize = cv2.resize(cv_img,(300,900))
        qt_img = self.convert_cv_qt(cv_resize)
        self.device4.setPixmap(qt_img)
    
    @pyqtSlot(np.ndarray)
    def update_image5(self, cv_img):
        #Updates the image_label with a new opencv image
        qt_img = self.convert_cv_qt(cv_img)
        self.device5.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
if __name__=="__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())
