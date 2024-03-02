from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget

from ljanalyzer.video import VideoSignals


class VideoProgessArea(QWidget):
    def __init__(self, parent: QWidget | None = ...) -> None:
        super().__init__(parent)
        # self.setMinimumSize(50 , 50)
        # self.setMaximumSize(200, 200)
        self.initUI()

    def initUI(self):
        self.setLayout(QVBoxLayout())
        self.layout().setAlignment(Qt.AlignTop | Qt.AlignRight)

    def add_widget(self, widget: QWidget):
        self.layout().addWidget(widget)

    def clear(self):
        while self.layout().count():
            curr_item = self.layout().takeAt(0)
            curr_widget = curr_item.widget()
            if curr_widget:
                curr_widget.setParent(None)
                curr_widget.deleteLater()


class VideoProgressBar(QWidget):
    def __init__(self, identifier: str, signals: VideoSignals) -> None:
        super().__init__()
        self.identifier = identifier
        self.initUI()
        signals.progress.connect(self.update_progressbar)
        signals.finished.connect(self.delete_progress_bar)

    def initUI(self):
        layout = QVBoxLayout()
        self.bar_label = QLabel(self.identifier)
        self.progressbar = QProgressBar()
        self.progressbar.setMinimum(0)
        self.progressbar.setMaximum(100)
        layout.addWidget(self.bar_label)
        layout.addWidget(self.progressbar)
        self.setLayout(layout)

    @pyqtSlot(int)
    def update_progressbar(self, value: int):
        self.progressbar.setValue(value)

    @pyqtSlot()
    def delete_progress_bar(self):
        parent_layout = self.parentWidget().layout()
        parent_layout.removeWidget(self)
        self.deleteLater()
