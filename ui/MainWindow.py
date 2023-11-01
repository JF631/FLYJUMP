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

import multiprocessing
from PyQt5 import QtGui

from PyQt5.QtWidgets import QMainWindow, QFileDialog

from .Ui_MainWindow import Ui_MainWindow
from ljanalyzer.video import Video

def start_video_analysis(file_name: str):
        vd = Video(file_name)
        vd.analyze()
class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.action_choose_video_file.triggered.connect(
            self.choose_file_dialog)
        self.running_processes = []

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
            video_process = multiprocessing.Process(target=start_video_analysis, args=(file_name,))
            video_process.start()
            self.running_processes.append(video_process)
        
        
    def closeEvent(self, event) -> None:
        for process in self.running_processes:
            process.terminate()
            process.join()
        event.accept()  