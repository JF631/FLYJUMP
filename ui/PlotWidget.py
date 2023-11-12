from PyQt5.QtGui import QPen
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import Qt, pyqtSlot
from qwt import QwtPlot, QwtPlotCurve, QwtLegend
import numpy as np

from ljanalyzer.video import VideoSignals

class PlotWidget(QWidget):
    def __init__(self, signals : VideoSignals, parent: QWidget | None = ...) -> None:
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
