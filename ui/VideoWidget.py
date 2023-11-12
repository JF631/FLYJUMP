from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PyQt5.QtCore import pyqtSlot, QCoreApplication, Qt

from ljanalyzer.frame import Frame
from ljanalyzer.video import VideoSignals

class VideoWidget(QWidget):
    def __init__(self, video_signals: VideoSignals = None, parent: QWidget | None = ...) -> None:
        super().__init__(parent)
        self.video_label = QLabel(parent)
        self.video_label.show()
        self.video_label.setSizePolicy(QSizePolicy.Expanding,
                                       QSizePolicy.Expanding)
        layout  = QVBoxLayout()
        layout.addWidget(self.video_label)
        if video_signals:
            self.connect_signals(video_signals)

    def connect_signals(self, video_signals: VideoSignals):
        self.video_signals = video_signals
        self.video_signals.update_frame.connect(self.update)
        # self.video_signals.finished.connect(self.clear)

    @pyqtSlot(Frame)
    def update(self, frame: Frame):
        height, width, channels = frame.dims
        bytes_per_line = channels * width
        q_image = QImage(frame.to_rgb(), width, height, bytes_per_line,
                         QImage.Format_RGB888)
        q_pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(q_pixmap.scaled(
            self.video_label.size(),aspectRatioMode=Qt.KeepAspectRatio))

    def resizeEvent(self, event):
        self.update_label_size()

    def update_label_size(self):
        self.video_label.resize(self.size())

    @pyqtSlot()
    def clear(self):
        layout = self.parentWidget().layout()
        if layout:
            layout.removeWidget(self)
            self.video_signals.update_frame.disconnect(self.update)
            self.video_signals.finished.disconnect(self.clear)
            self.deleteLater()

    def move_to_main_thread(obj):
        obj.moveToThread(QCoreApplication.instance().thread())