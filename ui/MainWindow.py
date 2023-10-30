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

from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox

from .Ui_MainWindow import Ui_MainWindow
from src.video import Video

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.action_choose_video_file.triggered.connect(
            self.choose_file_dialog)

    def choose_file_dialog(self):
        '''
        opens file chooser dialog to let user select a video file
        '''
        dialog_options = QFileDialog.Options()
        dialog_options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(
            self, 'Choose a Video File', '',
            'Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)',
            options=dialog_options)
        if not file_name:
            if len(file_name) > 0:
                QMessageBox.critical(self, "File not found", 
                                    f"The file {file_name} could not be found",
                                    QMessageBox.Ok)
            return
        vd = Video(file_name)
        vd.analyze()