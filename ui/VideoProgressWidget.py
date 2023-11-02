import typing
from PyQt5 import QtCore
from PyQt5.QtWidgets import QWidget, QLabel, QProgressBar, QVBoxLayout
from ljanalyzer.video import VideoSignals

class VideoProgressWidget(QWidget):
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
        
    def update_progressbar(self, value: int):
        print("value: %d%%" % value)
        self.progressbar.setValue(value)
        
    def delete_progress_bar(self):
        self.destroy()