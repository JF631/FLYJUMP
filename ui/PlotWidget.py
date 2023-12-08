from PyQt5.QtGui import QPen
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import Qt, pyqtSlot
from qwt import QwtPlot, QwtPlotCurve, QwtLegend
import numpy as np

from ljanalyzer.video import VideoSignals


class MultiPlot(QWidget):
    '''
    class that offers convenient Multi Qwt plot creation.
    Multiple plots per widget are supported.
    Each plot can again hold multiple curves.
    
    Parameters
    ----------
    num_plots : int
        number of plot widgets that should be emplaced in the multiplot widget
    plot_descr : dict
        holds discripition for each plot.
        Must be build like {"plot_ind":["curve_title0", "curve_titile1", ...]}.
        plot_ind is expected to be in range [0, num_plots).
        CAUTION: A name for each curve in a plot must be given!
    parent : QWidget
        layout parent that holds the widget 
    '''
    def __init__(self, signals:VideoSignals, num_plots:int = 1,
                 curves : dict = {}, parent: QWidget | None = ...) -> None:
        super().__init__(parent)
        self.plot_widgets = []
        self.video_signals = None
        self.curves = curves
        if signals:
            self.connect_signals(signals)
        self.initUI()

    def add_widget(self, widget: QWidget):
        self.layout().addWidget(widget)

    def create_subplot(self, plot_title, num_curves, curve_titles):
        plt_widget = Plot(plot_title, num_curves, curve_titles, self.parent())
        self.plot_widgets.append(plt_widget)
        self.add_widget(plt_widget)

    def initUI(self):
        self.setLayout(QVBoxLayout())
        self.layout().setAlignment(Qt.AlignTop | Qt.AlignRight)
        if not self.video_signals:
            return
        for plot_title, curve_titles in self.curves.items():
            self.create_subplot(plot_title, len(curve_titles), curve_titles)

    def connect_signals(self, signals: VideoSignals):
        self.video_signals = signals
        self.video_signals.update_frame_parameters.connect(self.set_data)

    @pyqtSlot(np.ndarray)
    def set_data(self, data: np.ndarray):
        # if np.any(data < 0.0):
        #     return
        curve_offset = 0
        for plot in self.plot_widgets:
            if np.any(data < 0.0):
                return
            plot_curves = plot.num_curves
            plot.set_data(data[:, curve_offset:curve_offset + plot_curves])
            curve_offset += plot_curves
        # self.plot_widgets[0].set_data(data)
        

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
    def __init__(self, title : str = "", num_curves: int = 1, 
                 curve_titles: list = ["curve1"],
                 parent: QWidget | None = ...) -> None:
        super().__init__(parent)
        assert(len(curve_titles) == num_curves)
        self.plot = QwtPlot(self)
        self.legend = QwtLegend()
        self.max_rows = 128
        self.current_row = 0
        self.num_curves = num_curves
        self.data = np.empty((self.max_rows, num_curves), dtype='f4')
        self.plot.insertLegend(self.legend, QwtPlot.BottomLegend)
        self.plot.setTitle(title)
        self.curves: QwtPlotCurve = []
        for title in curve_titles:
            curve = QwtPlotCurve(title)
            self.curves.append(curve)
            curve.attach(self.plot)
        
        layout = QVBoxLayout()
        layout.addWidget(self.plot)
        self.setLayout(layout)

    def set_data(self, data: np.ndarray):
        '''
        updates all curves in current plot.
        The x values are expected to be in time domain and are automatically 
        set.
        
        Parameters
        -----------
        data : np.ndarray
            matrix of shape (values_to_add, num_curves).
            Each colum thereby must hold new data for one curve. 
        '''
        columns = self.data.shape[1]
        num_new_values = data.shape[0]
        if self.current_row + num_new_values >= self.data.shape[0]:
            self.max_rows *= 2
            self.data.resize((self.max_rows, columns))
        self.data[[self.current_row, self.current_row + num_new_values]] = data
        self.current_row += num_new_values
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
