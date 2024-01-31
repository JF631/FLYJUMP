from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QWidget

from .Ui_DroneControlDialog import Ui_Dialog
from utils.droneControl import DroneControl

class DroneControlDialog(QDialog):
    def __init__(self, parent: QWidget | None = ...) -> None:
        super().__init__(parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.connect_signals_and_slots()
        self.drone_connection = DroneControl()
    
    def connect_signals_and_slots(self):
        self.ui.connect_button.clicked.connect(self.connect)
        self.ui.pushButton_3.clicked.connect(self.disarm_drone)

    def show(self):
        self.exec_()

    def connect(self):
        print("connected")
        self.drone_connection.connect_to_drone()
        self.drone_connection.arm()
    
    def disarm_drone(self):
        self.drone_connection.disarm()
