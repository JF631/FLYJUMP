'''
This file contains the Mainwindow that is shown on Application start up.

Ui files were created using Qt Designer and then translated to python file 
using pyuic5.

Ui Files for this Window:
- MainWindow.ui
- Ui_MainWindow.py


Author: Jakob Faust (software_jaf@mx442.de)
Date: 2023-10-28
'''

from PyQt5.QtWidgets import QMainWindow, QFileDialog, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import QThreadPool, Qt

from .Ui_MainWindow import Ui_MainWindow
from .VideoProgressWidget import VideoProgressWidget
from ljanalyzer.video import Video
from utils import controlsignals

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setupUi()
        self.ui.action_choose_video_file.triggered.connect(
            self.choose_file_dialog)
        self.progress_widgets = {}
        self.thread_pool = QThreadPool.globalInstance()
        self.thread_pool.setMaxThreadCount(self.thread_pool.maxThreadCount() / 2)
        self.control_signals = controlsignals.ControlSignals()
        
    def setupUi(self):
        self.main_layout = self.centralWidget().layout()
        self.progressbar_layout = QHBoxLayout()
        self.progressbar_layout.setAlignment(Qt.AlignBottom | Qt.AlignRight)
        self.main_layout.addChildLayout(self.progressbar_layout)

    def choose_file_dialog(self):
        '''
        opens file chooser dialog to let user select a video file
        '''
        dialog_options = QFileDialog.Options()
        dialog_options |= QFileDialog.ReadOnly
        file_names, _ = QFileDialog.getOpenFileNames(
            self, 'Choose a Video File', '',
            'Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)',
            options=dialog_options)
        if not file_names:
            return
        for file_name in file_names:
            video_task = Video(file_name, self.control_signals)
            progress_widget = VideoProgressWidget(video_task.get_path(),
                                                  video_task.signals)
            video_task.signals.progress.connect(self.show_progress)
            self.progressbar_layout.addWidget(progress_widget)
            self.thread_pool.start(video_task)
        # self.ui.centralwidget.(self.main_layout)
            
    def show_progress(self, progress):
        pass
        # print("%d%% done" % progress)
        
    def closeEvent(self, event) -> None:
        self.thread_pool.clear()
        self.control_signals.terminate.emit()
        self.thread_pool.waitForDone()
        event.accept()  