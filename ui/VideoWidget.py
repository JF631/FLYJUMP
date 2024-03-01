from PyQt5.QtCore import QCoreApplication, Qt, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from ljanalyzer.frame import Frame
from ljanalyzer.video import VideoSignals


class VideoWidget(QWidget):
    def __init__(
        self, video_signals: VideoSignals = None, parent: QWidget | None = ...
    ) -> None:
        super().__init__(parent)
        self.video_label = QLabel(parent)
        self.video_label.show()
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setScaledContents(False)
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        self.current_pixmap = None
        self.current_frame = None
        if video_signals:
            self.connect_signals(video_signals)

    def connect_signals(self, video_signals: VideoSignals):
        self.video_signals = video_signals
        self.video_signals.update_frame.connect(self.update)
        self.video_signals.error.connect(self.clear)

    @pyqtSlot(Frame)
    def update(self, frame: Frame):
        self.current_frame = frame
        self.set_image()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_label_size()

    def set_image(self):
        height, width, channels = self.current_frame.dims
        bytes_per_line = channels * width
        q_image = QImage(
            self.current_frame.to_rgb(),
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888,
        )
        self.current_pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(
            self.current_pixmap.scaled(
                self.video_label.size(), aspectRatioMode=Qt.KeepAspectRatio
            )
        )

    def update_label_size(self):
        size = self.size()
        self.video_label.resize(size)
        if self.current_pixmap:
            self.set_image()

    def disconnect_signals(self):
        self.video_signals.update_frame.disconnect(self.update)
        self.video_signals.error.disconnect(self.clear)

    @pyqtSlot()
    def clear(self):
        self.video_signals.update_frame.disconnect(self.update)
        self.video_signals.error.disconnect(self.clear)
        layout = self.parentWidget().layout()
        if layout:
            layout.removeWidget(self)
            self.deleteLater()

    def move_to_main_thread(obj):
        obj.moveToThread(QCoreApplication.instance().thread())
