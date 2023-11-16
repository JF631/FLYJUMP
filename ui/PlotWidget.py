import typing
from PyQt5 import QtCore
from PyQt5.QtGui import QPen
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import Qt, pyqtSlot
from qwt import QwtPlot, QwtPlotCurve, QwtLegend
import numpy as np

from ljanalyzer.video import VideoSignals


class MultiPlot(QWidget):
    def __init__(self, signals:VideoSignals, num_plots:int = 1, 
                 parent: QWidget | None = ...) -> None:
        super().__init__(parent)
        self.plot_widgets = []
        self.video_signals = None
        if signals:
            self.connect_signals(signals)
        self.initUI(num_plots)

    def add_widget(self, widget: QWidget):
        self.layout().addWidget(widget)

    def create_subplot(self):
        plt_widget = Plot(3, ["left foot", "right foot", "hip"], self.parent())
        self.plot_widgets.append(plt_widget)
        self.add_widget(plt_widget)

    def initUI(self, num_plots:int):
        self.setLayout(QVBoxLayout())
        self.layout().setAlignment(Qt.AlignTop | Qt.AlignRight)
        if not self.video_signals:
            return
        for _ in range(num_plots):
            self.create_subplot()

    def connect_signals(self, signals: VideoSignals):
        self.video_signals = signals
        self.video_signals.update_frame_parameters.connect(self.set_data)

    @pyqtSlot(np.ndarray)
    def set_data(self, data: np.ndarray):
        if np.any(data < 0.0) or np.any(data > 1.0):
            return
        self.plot_widgets[0].set_data(data[1])
        

class Plot(QWidget):
    '''
    class that offers convenient Qwt plot creation.
    Multiple curves per plot are supported.
    
    Parameters
    ----------
    num_curves : int
        number of data curves that should be emplaced in the plot
    titles : list
        holds title for each curve.
        Must have the same length as num_curves
    parent : QWidget
        layout parent that holds the widget 
    '''
    def __init__(self,  num_curves: int = 1, titles: list = ["curve1"], 
                 parent: QWidget | None = ...) -> None:
        super().__init__(parent)
        assert(len(titles) == num_curves)
        self.plot = QwtPlot(self)
        self.legend = QwtLegend()
        self.max_rows = 128
        self.current_row = 0
        self.data = np.empty((self.max_rows, num_curves), dtype='f4')
        self.plot.insertLegend(self.legend, QwtPlot.BottomLegend)
        self.curves: QwtPlotCurve = []
        for title in titles:
            curve = QwtPlotCurve(title)
            self.curves.append(curve)
            curve.attach(self.plot)
        
        layout = QVBoxLayout()
        layout.addWidget(self.plot)
        self.setLayout(layout)

    def set_data(self, data: np.ndarray):
        '''
        updates all curves in current plot.
        
        Parameters
        -----------
        data : np.ndarray
            flat array of shape (1, num_curves).
            Each colum must hold the new data for one curve. 
        '''
        columns = self.data.shape[1]
        assert(data.shape[0] == columns)
        if self.current_row + 1 >= self.data.shape[0]:
            self.max_rows *= 2
            self.data.resize((self.max_rows, columns))
        self.data[self.current_row] = data
        self.current_row += 1
        
        for i, curve in enumerate(self.curves):
            curve.setData(np.arange(self.current_row), self.data[:,i])
            
        self.plot.setAxisAutoScale(QwtPlot.xBottom)
        self.plot.setAxisAutoScale(QwtPlot.yLeft)
        self.plot.replot()

class PlotWidget(QWidget):
    def __init__(self, signals : VideoSignals, 
                 parent: QWidget | None = ...) -> None:
        super().__init__(parent)
        signals.update_frame_parameters.connect(self.set_data)
        # Create a QwtPlot widget
        self.plot = QwtPlot(self)
        self.plot_legend = QwtLegend()
        self.plot.insertLegend(self.plot_legend, QwtPlot.BottomLegend)
        self.l_foot_y = np.array([])
        self.r_foot_y = np.array([])
        self.l_foot_curve = QwtPlotCurve("Left foot")
        self.r_foot_curve = QwtPlotCurve("Right foot")
        self.r_foot_curve.setTitle("right foot")
        l_foot_pen = QPen(Qt.red)
        self.l_foot_curve.setPen(l_foot_pen)

        # Set up the layout
        layout = QVBoxLayout()
        layout.addWidget(self.plot)
        self.setLayout(layout)

        # Customize the plot and curve as needed
        # For example, you can set axis titles:
        self.plot.setAxisTitle(QwtPlot.xBottom, "X Axis")
        self.plot.setAxisTitle(QwtPlot.yLeft, "Y Axis")
        
        self.l_foot_curve.attach(self.plot)
        self.r_foot_curve.attach(self.plot)

    @pyqtSlot(np.ndarray)
    def set_data(self, data: np.ndarray):
        if np.any(data < 0.0) or np.any(data > 1.0):
            return
        self.l_foot_y = np.append(self.l_foot_y, data[1,0])
        self.r_foot_y = np.append(self.r_foot_y, data[1,1])
        self.l_foot_curve.setData(np.arange(len(self.l_foot_y)), self.l_foot_y)
        self.r_foot_curve.setData(np.arange(len(self.r_foot_y)), self.r_foot_y)

        # Automatically adjust the axis scales based on the data
        self.plot.setAxisAutoScale(QwtPlot.xBottom)
        self.plot.setAxisAutoScale(QwtPlot.yLeft)
        self.plot.replot()
