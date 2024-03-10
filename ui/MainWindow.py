"""
This file contains the Mainwindow that is shown on Application start up.

Ui files were created using Qt Designer and then translated to python files 
using pyuic5.

Ui Files for this Window:
- MainWindow.ui
- Ui_MainWindow.py


Author: Jakob Faust (software_jaf@mx442.de)
Date: 2023-10-28
"""

from PyQt5.QtCore import Qt, QThreadPool, pyqtSlot
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QFileDialog, QLabel, QMainWindow, QMessageBox,
                             QVBoxLayout)

from ljanalyzer.eval import Filter, EvalType
from ljanalyzer.video import Video
from utils.controlsignals import ControlSignals, SharedBool
from utils.filehandler import FileHandler, ParameterFile

from .DroneControlDialog import DroneControlDialog
from .PlotWidget import MatplotCanvas, MultiPlot
from .Ui_MainWindow import Ui_MainWindow
from .VideoProgressWidget import VideoProgessArea, VideoProgressBar
from .VideoWidget import VideoWidget


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.control_signals = ControlSignals()
        self.abort_flag = SharedBool()
        self.ui = Ui_MainWindow()
        self.setFocusPolicy(Qt.StrongFocus)
        self.ui.setupUi(self)
        self.video_widget = None
        self.setupUi()
        self.connect_signals_and_slots()
        self.thread_pool = QThreadPool.globalInstance()
        self.current_video = None
        self.__analysis_files = None
        FileHandler.create_general_structure()
        """
        limit used cpu cores to half of available cores.
        This is because usually one video analysis uses two cores.
        """
        self.thread_pool.setMaxThreadCount(int(self.thread_pool.maxThreadCount() / 2))

    def connect_signals_and_slots(self):
        self.ui.action_choose_video_file.triggered.connect(self.choose_video_dialog)
        self.ui.action_load_analysis.triggered.connect(self.choose_analysis_dialog)
        self.ui.action_control_drone.triggered.connect(self.show_drone_control)
        self.ui.start_analysis_btn.pressed.connect(self.start_analysis)
        self.ui.export_frame_btn.clicked.connect(self.export_frame)
        self.ui.filtered_out_checkBox.stateChanged.connect(self.set_save_filter_output)

    def hide_analysis_params(self):
        self.ui.start_analysis_btn.hide()
        self.ui.filtered_out_checkBox.hide()
        self.ui.filter_combo.hide()
        self.ui.type_combo.hide()
        self.ui.export_frame_btn.hide()
        self.ui.overlay_checkBox.hide()

    def show_analysis_params(self):
        self.ui.start_analysis_btn.show()
        self.ui.filtered_out_checkBox.show()
        self.ui.filter_combo.show()
        self.ui.type_combo.show()
        self.ui.overlay_checkBox.show()

    def setupUi(self):
        # main video widget
        self.video_widget = VideoWidget(parent=self.ui.main_video)
        video_area = QVBoxLayout(self.ui.main_video)
        self.logo_label = QLabel(self.ui.main_video)
        logo = QPixmap(FileHandler.get_icon_path() + "/logo.png")
        logo = logo.scaled(512, 512, Qt.KeepAspectRatio)
        self.logo_label.setPixmap(logo)
        self.logo_label.setAlignment(Qt.AlignCenter)
        video_area.addWidget(self.logo_label)
        video_area.addWidget(self.video_widget)
        self.video_widget.hide()
        self.hide_analysis_params()
        self.ui.main_video.setLayout(video_area)
        # matplot widgets
        self.result_area = QVBoxLayout(self.ui.result_area)
        self.matplot_widget = None
        self.matplot_angle = None
        # analysis progressbar
        self.progressbar_area = VideoProgessArea(self.ui.result_area)
        self.result_area.addWidget(self.progressbar_area)
        self.ui.result_area.setLayout(self.result_area)
        # combobox for filters
        for filter in Filter:
            self.ui.filter_combo.addItem(filter.name)
        for eval_type in EvalType:
            self.ui.type_combo.addItem(eval_type.name)

    def __add_matplot_ui(self):
        self.matplot_widget = MatplotCanvas(
            self.ui.result_area,
            x_label="t[frames]",
            y_label="height[norm. pixel]",
            control_signals=self.control_signals,
        )
        self.matplot_angle = MatplotCanvas(
            self.ui.result_area,
            x_label='t[frames]', y_label='angle[degree]',
            control_signals=self.control_signals)
        self.result_area.addWidget(self.matplot_widget, stretch=3)
        self.result_area.addWidget(self.matplot_angle, stretch=3)

    def show_drone_control(self):
        control_dialog = DroneControlDialog(self)
        control_dialog.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        control_dialog.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        control_dialog.exec_()

    def __start_video_analaysis(self, file_names, filter: Filter = None,
                                eval_type: EvalType = None,
                                save_filter_output: bool = False,
                                analysis_overlay: bool = True):
        """
        Starts one or multiple video analysis Threads.
        One thread per passed file name is created.

        Parameters
        ----------
        file_names : list
            List of video (.mp4) files that should be analyzed.
        """
        if not file_names:
            return
        if not self.video_widget:
            return
        if self.logo_label:
            self.logo_label.hide()
            self.logo_label = None
        self.video_widget.show()
        self.progressbar_area.clear()
        if self.matplot_widget:
            self.matplot_widget.hide()
        if self.matplot_angle:
            self.matplot_angle.hide()
        if self.current_video:
            self.current_video.stop()
            self.current_video = None
        show_video = True
        if len(file_names) > 1:
            show_video = False
        plot_descr = {
            "Height": ["left foot", "right foot", "hip"],
            "Angle": ["right knee", "left knee"],
        }
        for file_name in file_names:
            video_task = Video(file_name, self.abort_flag)
            video_task.set_filter(filter)
            video_task.set_eval_type(eval_type)
            video_task.set_filter_output(save_filter_output)
            video_task.set_analysis_overlay(analysis_overlay)
            progress_widget = VideoProgressBar(
                video_task.get_filename(), video_task.signals
            )
            if show_video:
                video_task.signals.finished.connect(self.analysation_finished)
                video_task.signals.error.connect(self.handle_error)
                multi_plot = MultiPlot(
                    signals=video_task.signals,
                    num_plots=2,
                    curves=plot_descr,
                    parent=self.ui.result_area,
                )
                self.video_widget.connect_signals(video_task.signals)
                self.progressbar_area.add_widget(multi_plot)
            self.progressbar_area.add_widget(progress_widget)
            self.thread_pool.start(video_task)

    def __show_analysis_result(self, file_names):
        """
        visualizes parameter file analysis results.
        Currently in one matplot window.

        Parameters
        ----------
        file_names : list
            list of hdf5 Parameter Files.
            Only the first file in the list is visualized.
        """
        if not file_names:
            return
        if self.logo_label:
            self.logo_label.hide()
            self.logo_label = None
        self.video_widget.show()
        if not self.matplot_angle or not self.matplot_widget:
            self.__add_matplot_ui()
        self.ui.export_frame_btn.show()
        param_file = ParameterFile(file_names[0])
        param_file.load()
        takeoff_frame = param_file.get_takeoff_frame()
        first_frame = 0
        if takeoff_frame is None:
            message = QMessageBox.critical(
                None,
                "Error",
                """No takeoff frame found. Do still you want to
                show the other results?""",
                QMessageBox.Yes | QMessageBox.Cancel,
            )
            if message == QMessageBox.Cancel:
                return
        else:
            first_frame = takeoff_frame[0]
        if self.current_video:
            self.video_widget.disconnect_signals()
            self.current_video.stop()
            self.current_video = None
        self.matplot_widget.clear()
        self.matplot_angle.clear()
        self.matplot_angle.show()
        self.matplot_widget.show()
        self.progressbar_area.clear()
        self.matplot_widget.plot2D(param_file.get_left_foot_height(),
                                   label="left foot")
        self.matplot_widget.plot2D(
            param_file.get_right_foot_height(), label="right foot"
        )
        self.matplot_widget.plot2D(param_file.get_hip_height(),
                                   label="hip height")
        self.matplot_widget.add_points(takeoff_frame,
                                       label="changing points")
        self.matplot_angle.plot2D(
            param_file.get_left_knee_angle(), label="left knee angle"
        )
        self.matplot_angle.plot2D(
            param_file.get_right_knee_angle(), label="right knee angle"
        )
        self.matplot_angle.add_points(takeoff_frame, label="changing points")
        self.current_video = Video(param_file.get_video_path(),
                                   self.abort_flag)
        self.current_video.set_control_signals(self.control_signals)
        self.video_widget.connect_signals(self.current_video.signals)
        self.ui.takeoff_frame_label.setText(
            str(param_file.get_takeoff_frame()[0]))
        self.ui.takeoff_angle_label.setText(
            str(param_file.get_takeoff_angle()[0]))
        self.current_video.play(first_frame)

    def set_save_filter_output(self, state):
        if not self.control_signals:
            return
        if state == 2:
            self.control_signals.change_filter_output_mode.emit(True)
        else:
            self.control_signals.change_filter_output_mode.emit(False)

    @pyqtSlot()
    def export_frame(self):
        file_name, _ = QFileDialog.getSaveFileName(
            None, "Save current frame", "", "Images (*.jpg)")
        if file_name:
            self.control_signals.export_frame.emit(file_name)

    @pyqtSlot(str)
    def handle_error(self, error: str):
        print(error)
        message = QMessageBox.critical(
            None,
            "Error",
            """The file {} already exists. Do you want to
                override it?""".format(
                error
            ),
            QMessageBox.Yes | QMessageBox.Cancel,
        )
        if message == QMessageBox.Cancel:
            return None

    @pyqtSlot()
    def choose_analysis_dialog(self):
        """
        opens file chooser dialog to let user select an analyzed video file
        """
        dialog_options = QFileDialog.Options()
        dialog_options |= QFileDialog.ReadOnly
        start_path = FileHandler.get_output_path()
        file_names, _ = QFileDialog.getOpenFileNames(
            self,
            "Choose an analysis File",
            start_path,
            "HDF5 Files (*.h5 *.hdf5);;All Files (*)",
            options=dialog_options,
        )
        if not file_names:
            return
        self.ui.video_loaded_label.setText('loaded:' + ', '.join(file_names))
        self.__show_analysis_result(file_names)

    @pyqtSlot()
    def choose_video_dialog(self):
        """
        opens file chooser dialog to let user select a video file
        """
        dialog_options = QFileDialog.Options()
        dialog_options |= QFileDialog.ReadOnly
        file_names, _ = QFileDialog.getOpenFileNames(
            self,
            "Choose a Video File",
            "",
            "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)",
            options=dialog_options,
        )
        self.__analysis_files = file_names
        if len(file_names) <= 0:
            return
        self.ui.video_loaded_label.setText(", ".join(file_names))
        self.show_analysis_params()

    @pyqtSlot()
    def start_analysis(self):
        if not self.__analysis_files:
            return
        self.__start_video_analaysis(
            self.__analysis_files, Filter[self.ui.filter_combo.currentText()],
            EvalType[self.ui.type_combo.currentText()],
            self.ui.filtered_out_checkBox.isChecked(),
            self.ui.overlay_checkBox.isChecked()
        )

    @pyqtSlot(str)
    def analysation_finished(self, anlysis_path):
        """
        Slot that is always called when a video analysis is finished.

        Parameters
        ----------
        analysis_path : str
            Path of the analyzed .hdf5 Parameter file
        """
        print(f"analysis path: {anlysis_path}")
        self.video_widget.disconnect_signals()
        self.__show_analysis_result([anlysis_path])

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            if self.current_video:
                self.current_video.toggle()
        if event.key() == Qt.Key_Left:
            if self.current_video:
                self.current_video.rewind()
        if event.key() == Qt.Key_Right:
            if self.current_video:
                self.current_video.forward()

    def closeEvent(self, event) -> None:
        self.abort_flag.set()
        self.thread_pool.clear()
        self.thread_pool.waitForDone()
        event.accept()
