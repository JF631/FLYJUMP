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

from PyQt5.QtWidgets import QMainWindow, QFileDialog
from PyQt5.QtCore import QThreadPool, pyqtSlot

from .Ui_MainWindow import Ui_MainWindow
from .VideoProgressWidget import VideoProgressBar, VideoProgessArea
from ljanalyzer.video import Video
from utils.controlsignals import ControlSignals, SharedBool

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.control_signals = ControlSignals()
        self.abort_flag = SharedBool()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setupUi()
        self.ui.action_choose_video_file.triggered.connect(
            self.choose_file_dialog)
        self.thread_pool = QThreadPool.globalInstance()
        # limit used cpu cores to half of available cores.
        # This is because usually one video analysis uses two cores.  
        self.thread_pool.setMaxThreadCount(
            int(self.thread_pool.maxThreadCount() / 2)
        )

    def setupUi(self):
        self.progressbar_area = VideoProgessArea(self)
        self.centralWidget().layout().addWidget(self.progressbar_area)

    @pyqtSlot()
    def choose_file_dialog(self):
        '''
        opens file chooser dialog to let user select a video file
        '''
        dialog_options = QFileDialog.Options()
        dialog_options |= QFileDialog.ReadOnly
        file_names, _ = QFileDialog.getOpenFileNames(
            self, 'Choose a Video File', '',
            'Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)',
            options=dialog_options
        )
        if not file_names:
            return
        for file_name in file_names:
            video_task = Video(file_name, self.abort_flag)
            progress_widget = VideoProgressBar(video_task.get_filename(),
                                               video_task.signals)
            self.progressbar_area.add_widget(progress_widget)
            self.thread_pool.start(video_task)

    def closeEvent(self, event) -> None:
        self.abort_flag.set()
        self.thread_pool.clear()
        self.thread_pool.waitForDone()
        event.accept()
