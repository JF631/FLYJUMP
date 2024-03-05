import threading
import socket

import cv2
from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QIcon, QKeyEvent
from PyQt5.QtSerialPort import QSerialPortInfo
from PyQt5.QtWidgets import QDialog, QMessageBox, QVBoxLayout, QWidget

from ljanalyzer.frame import Frame
from ljanalyzer.video import VideoSignals
from utils.controlsignals import DroneSignals, SharedBool
from utils.droneControl import DroneConnection, DroneControl
from utils.filehandler import FileHandler

from .Ui_DroneControlDialog import Ui_Dialog
from .VideoWidget import VideoWidget


class LiveStreamHandler(QThread):
    finished = pyqtSignal()

    def __init__(
        self, parent: QObject | None = ..., signals: VideoSignals = None
    ) -> None:
        self.signals = signals
        self.cap = None
        self.ctrl_socket: socket.socket = None
        self.analysis_running = SharedBool()
        super(LiveStreamHandler, self).__init__(parent)

    def init_connection(self):
        self.ctrl_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.ctrl_socket.connect(('10.42.0.1', 8080))
        except TimeoutError as te:
            print(f"connection timed out with: {te}")
            self.ctrl_socket = None
            return
        except Exception as e:
            print(f"connection failed with: {e}")
            self.ctrl_socket = None
            return

    def _start_stream(self):
        if not self.ctrl_socket:
            return
        self.ctrl_socket.sendall('stream'.encode())
        self.__receive_stream()

    def record_video(self):
        if not self.ctrl_socket:
            return
        self.ctrl_socket.sendall('start_recording'.encode())
        self.analysis_running.set()

    def receive_data(self):
        if not self.ctrl_socket:
            return
        file_size = self.ctrl_socket.recv(4)
        file_size = int.from_bytes(file_size, byteorder='big')
        received = b''
        while len(received) < file_size:
            chunk = self.ctrl_socket.recv(1024)
            received += chunk
        with open('test123.mp4', 'wb') as infile:
            infile.write(received)
        print('everything received')


    def stop_recording(self):
        if not self.ctrl_socket:
            return
        self.ctrl_socket.sendall('stop_recording'.encode())
        self.analysis_running.reset()
        self.receive_data()
        # recv = threading.Thread(target=self.receive_data)
        # recv.start()

    def __receive_stream(self):
        print("receiving...")
        ip_address = "10.42.0.1"
        port = 8081
        video_url = f"tcp://{ip_address}:{port}"
        self.cap = cv2.VideoCapture(video_url)
        if not self.cap.isOpened():
            print("Error: Unable to open video stream")
            return
        while self.cap.isOpened() and not self.analysis_running.get():
            try:
                ret, frame = self.cap.read()
                # print(frame)
                if not ret:
                    print("Error: Unable to read frame from video stream")
                    break
                frame = Frame(frame)
                self.signals.update_frame.emit(frame)
            except Exception as e:
                print("error while reading video stream: {}".format(e))
                self.cap.release()
        self.cap.release()
        self.analysis_running.reset()
        print("live stream cleaned up")

    def terminate(self) -> None:
        print("trying to clean up")
        if self.cap:
            self.cap.release()
        return super().terminate()

    def run(self) -> None:
        self.init_connection()
        self._start_stream()
        self.finished.emit()


class DroneControlDialog(QDialog):
    def __init__(self, parent: QWidget | None = ...) -> None:
        super().__init__(parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.drone_signals = DroneSignals()
        self.video_signals = VideoSignals()
        self.live_stream_handler = LiveStreamHandler(self, signals=self.video_signals)
        self.setupUi()
        self.connect_signals_and_slots()
        self.scan_serial_ports()
        self.drone_connection = DroneConnection(self.drone_signals)
        self.drone_control = DroneControl(self.drone_connection, self.drone_signals)

    def setupUi(self):
        self.video_widget = VideoWidget(
            self.video_signals, parent=self.ui.live_stream_widget
        )
        video_area = QVBoxLayout(self.ui.live_stream_widget)
        video_area.addWidget(self.video_widget)
        self.ui.live_stream_widget.setLayout(video_area)
        icon_path = FileHandler.get_icon_path()
        self.ui.up_btn.setIcon(QIcon(icon_path + "/arrow_up.png"))
        self.ui.down_btn.setIcon(QIcon(icon_path + "/arrow_down.png"))
        self.ui.right_btn.setIcon(QIcon(icon_path + "/arrow_right.png"))
        self.ui.left_btn.setIcon(QIcon(icon_path + "/arrow_left.png"))

    def scan_serial_ports(self):
        available_ports = QSerialPortInfo.availablePorts()
        current_descr = None
        for port in available_ports:
            current_descr = port.description()
            self.ui.com_combobox.addItem(port.portName())
            if "USB" in current_descr:
                self.ui.com_combobox.setCurrentText(port.portName())

    def connect_signals_and_slots(self):
        self.finished.connect(self.on_dialog_closed)
        self.ui.clear_msg_button.clicked.connect(self.clear_check_messages)
        self.ui.connect_button.clicked.connect(self.connect)
        self.ui.livestream_btn.clicked.connect(self.show_livestream)
        self.ui.start_analysis_btn.clicked.connect(self.start_measurement)
        self.ui.stop_analysis_btn.clicked.connect(self.stop_measurement)
        self.ui.takeoff_button.clicked.connect(self.perform_takeoff)
        self.ui.land_button.clicked.connect(self.perform_landing)
        self.ui.rtl_button.clicked.connect(self.init_home_return)
        self.ui.left_btn.clicked.connect(self.fly_left)
        self.ui.right_btn.clicked.connect(self.fly_right)
        self.ui.up_btn.clicked.connect(self.fly_forward)
        self.ui.down_btn.clicked.connect(self.fly_backwards)
        self.drone_signals.status_text.connect(self.update_error_label)
        self.drone_signals.vehicle_battery_status.connect(self.update_battery_info)
        self.drone_signals.connection_changed.connect(self.change_connection_status)
        self.drone_signals.vehicle_gps_status.connect(self.update_params)

    @pyqtSlot(str)
    def update_error_label(self, text):
        current_msg = self.ui.label_arm_checks.text()
        text = current_msg + "\n" + text
        self.ui.label_arm_checks.setText(text)

    @pyqtSlot(dict)
    def update_params(self, vehicle_status):
        alt = vehicle_status.get("relative_alt")  # in m
        gnd_speed = vehicle_status.get("velocity")  # in m/s
        sat_count = vehicle_status.get("satelites")
        fix_type = vehicle_status.get("fix_type")
        self.ui.label_height.setText(str(alt))
        self.ui.label_velocity.setText(str(gnd_speed))
        self.ui.label_satelites.setText(str(sat_count))
        self.ui.label_fix_type.setText(str(fix_type))

    @pyqtSlot(dict)
    def update_battery_info(self, battery_status):
        voltage = battery_status.get("voltages")[0] / 1e3
        consumed_current = battery_status.get("current_consumed")
        self.ui.label_voltage.setText(str(voltage))
        self.ui.label_consumed_current.setText(str(consumed_current))

    @pyqtSlot(bool)
    def change_connection_status(self, connetcted: bool):
        btn_text = "Connection Lost"
        if connetcted:
            btn_text = "Connected"
        self.ui.connect_button.setText(btn_text)

    @pyqtSlot()
    def clear_check_messages(self):
        self.ui.label_arm_checks.clear()

    def get_port_from_name(self, name: str):
        available_ports = QSerialPortInfo.availablePorts()
        for port in available_ports:
            if port.portName() == name:
                return port
        return None

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
        elif key == Qt.Key_I:
            self.climb()
            print("climb")
        elif key == Qt.Key_K:
            self.descent()
            print("descent")
        else:
            super().keyPressEvent(event)

    def show(self):
        self.exec_()

    def show_livestream(self):
        self.live_stream_handler.start()

    def start_measurement(self):
        self.live_stream_handler.record_video()

    def stop_measurement(self):
        self.live_stream_handler.stop_recording()

    def connect(self):
        self.live_stream_handler.start()
        selected_port_name = self.ui.com_combobox.currentText()
        selected_port = self.get_port_from_name(selected_port_name)
        if not selected_port:
            return
        if selected_port.isBusy():
            QMessageBox.critical(
                None,
                "Serial port error",
                f"""Serial port {selected_port_name} is currently busy.\
                        Thus, it cannot be opened (again)\
                        Maybe a connection is still active?""",
                QMessageBox.Ok,
            )
            return
        if not "USB" in selected_port.description():
            choice = QMessageBox.warning(
                None,
                "Serial port warning",
                f"""Serial port {selected_port_name} is not a USB\
                        port! Click Retry to continue or Cancel to select\
                        another port""",
                QMessageBox.Cancel | QMessageBox.Retry,
            )
            if choice == QMessageBox.Cancel:
                return
        if not self.drone_connection.init_serial_port(selected_port_name):
            return
        self.drone_connection.start()
        arm_check_thread = threading.Thread(
            name="armingChecks", target=self.drone_control.run_arming_checks
        )
        arm_check_thread.start()

    def perform_takeoff(self):
        self.drone_control.arm()
        self.drone_control.takeoff(height=9)

    def disarm_drone(self):
        self.drone_control.run_arming_checks()

    def arm_drone(self):
        self.drone_control.arm()

    def climb(self):
        self.drone_control.climb(velocity=0.5)

    def descent(self):
        self.drone_control.sink(velocity=0.5)

    def init_home_return(self):
        self.drone_control.return_home()

    def perform_landing(self):
        self.drone_control.land()

    def fly_forward(self):
        self.drone_control.fly_forward(velocity=0.5)

    def fly_backwards(self):
        self.drone_control.fly_backwards(velocity=0.5)

    def fly_right(self):
        self.drone_control.fly_right(velocity=0.5)

    def fly_left(self):
        self.drone_control.fly_left(velocity=0.5)

    def on_dialog_closed(self):
        self.drone_connection.terminate()
        self.live_stream_handler.terminate()
        self.drone_connection.close()
