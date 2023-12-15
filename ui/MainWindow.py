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
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QVBoxLayout
from PyQt5.QtCore import QThreadPool, pyqtSlot

from .Ui_MainWindow import Ui_MainWindow
from .VideoProgressWidget import VideoProgressBar, VideoProgessArea
from .PlotWidget import MultiPlot
from .VideoWidget import VideoWidget
from ljanalyzer.video import Video
from utils.controlsignals import ControlSignals, SharedBool
from utils.filehandler import FileHandler

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.control_signals = ControlSignals()
        self.abort_flag = SharedBool()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.video_widget = None
        self.setupUi()
        self.ui.action_choose_video_file.triggered.connect(
            self.choose_file_dialog)
        self.thread_pool = QThreadPool.globalInstance()
        FileHandler.create_general_structure()
        '''
        limit used cpu cores to half of available cores.
        This is because usually one video analysis uses two cores.
        '''
        self.thread_pool.setMaxThreadCount(
            int(self.thread_pool.maxThreadCount() / 2)
        )

    def setupUi(self):
        self.progressbar_area = VideoProgessArea(self)
        result_area_layout = QVBoxLayout(self.ui.result_area)
        result_area_layout.addWidget(self.progressbar_area)
        video_area_layout = QVBoxLayout(self.ui.main_video)
        self.ui.main_video.setLayout(video_area_layout)
        self.video_widget = VideoWidget(parent=self.ui.main_video)
        video_area_layout.addWidget(self.video_widget)

    def __start_video_analaysis(self, file_names):
        if not file_names:
            return
        if not self.video_widget:
            return
        show_video = True
        if len(file_names) > 1:
            show_video = False
        plot_descr = {"Height": ["left foot", "right foot", "hip"],
                      "Angle":["right knee", "left knee"]}
        for file_name in file_names:
            video_task = Video(file_name, self.abort_flag)
            progress_widget = VideoProgressBar(video_task.get_filename(),
                                               video_task.signals)
            if show_video:
                multi_plot = MultiPlot(signals=video_task.signals,
                                       num_plots=2, curves=plot_descr,
                                       parent=self.ui.result_area)
                self.video_widget.connect_signals(video_task.signals)
                self.progressbar_area.add_widget(multi_plot)
            self.progressbar_area.add_widget(progress_widget)
            self.thread_pool.start(video_task)

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
        self.__start_video_analaysis(file_names)

    def closeEvent(self, event) -> None:
        self.abort_flag.set()
        self.thread_pool.clear()
        self.thread_pool.waitForDone()
        event.accept()
