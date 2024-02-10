from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import  QDialog, QWidget
from PyQt5.QtGui import QIcon, QKeyEvent

import threading

from .Ui_DroneControlDialog import Ui_Dialog
from utils.droneControl import DroneControl, DroneConnection
from utils.controlsignals import DroneSignals
from utils.filehandler import FileHandler

class DroneControlDialog(QDialog):
    def __init__(self, parent: QWidget | None = ...) -> None:
        super().__init__(parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.setupUi()
        self.drone_signals = DroneSignals()
        self.connect_signals_and_slots()
        self.drone_connection = DroneConnection(self.drone_signals)
        self.drone_control = DroneControl(self.drone_connection,
                                          self.drone_signals)
        
    def setupUi(self):
        icon_path = FileHandler.get_icon_path()
        self.ui.up_btn.setIcon(QIcon(icon_path + '/arrow_up.png'))
        self.ui.down_btn.setIcon(QIcon(icon_path + '/arrow_down.png'))
        self.ui.right_btn.setIcon(QIcon(icon_path + '/arrow_right.png'))
        self.ui.left_btn.setIcon(QIcon(icon_path + '/arrow_left.png'))
    
    def connect_signals_and_slots(self):
        self.finished.connect(self.on_dialog_closed)
        self.ui.clear_msg_button.clicked.connect(self.clear_check_messages)
        self.ui.connect_button.clicked.connect(self.connect)
        self.ui.takeoff_button.clicked.connect(self.perform_takeoff)
        self.ui.land_button.clicked.connect(self.perform_landing)
        self.ui.rtl_button.clicked.connect(self.init_home_return)
        self.ui.left_btn.clicked.connect(self.fly_left)
        self.ui.right_btn.clicked.connect(self.fly_right)
        self.ui.up_btn.clicked.connect(self.fly_forward)
        self.ui.down_btn.clicked.connect(self.fly_backwards)
        self.drone_signals.status_text.connect(self.update_error_label)
        self.drone_signals.connection_changed.connect(
            self.change_connection_status)
        self.drone_signals.vehicle_gps_status.connect(self.update_params)

    @pyqtSlot(str)
    def update_error_label(self, text):
        current_msg = self.ui.label_arm_checks.text()
        text = current_msg + "\n" + text
        self.ui.label_arm_checks.setText(text)

    @pyqtSlot(dict)
    def update_params(self, vehicle_status):
        alt = vehicle_status.get('relative_alt') #in m
        gnd_speed = vehicle_status.get('velocity') # in m/s
        sat_count = vehicle_status.get('satelites')
        fix_type = vehicle_status.get('fix_type')
        self.ui.label_height.setText(str(alt))
        self.ui.label_velocity.setText(str(gnd_speed))
        self.ui.label_satelites.setText(str(sat_count))
        self.ui.label_fix_type.setText(str(fix_type))
    
    @pyqtSlot(bool)
    def change_connection_status(self, connetcted: bool):
        btn_text = 'Connection Lost'
        if connetcted:
            btn_text = 'Connected'
        self.ui.connect_button.setText(btn_text)
    
    @pyqtSlot()
    def clear_check_messages(self):
        self.ui.label_arm_checks.clear()
    
    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key == Qt.Key_W:
            self.fly_forward()
        elif key == Qt.Key_S:
            self.fly_backwards()
        elif key == Qt.Key_A:
            self.fly_left()
        elif key == Qt.Key_D:
            self.fly_right()
        else:
            super().keyPressEvent(event)

    def show(self):
        self.exec_()

    def connect(self):
        self.drone_connection.start()
        arm_check_thread = threading.Thread(name='armingChecks',
                         target=self.drone_control.run_arming_checks)
        arm_check_thread.start()

    def perform_takeoff(self):
        self.drone_control.takeoff(height=5)
    
    def disarm_drone(self):
        self.drone_control.run_arming_checks()
    
    def arm_drone(self):
        self.drone_control.arm()

    def init_home_return(self):
        self.drone_control.return_home()
    
    def perform_landing(self):
        self.drone_control.land()
    
    def fly_forward(self):
        self.drone_control.fly_forward(velocity=1)
    
    def fly_backwards(self):
        self.drone_control.fly_backwards()
    
    def fly_right(self):
        self.drone_control.fly_right()
    
    def fly_left(self):
        self.drone_control.fly_left()
    
    def on_dialog_closed(self):
        self.drone_connection.terminate()
        self.drone_connection.close()
