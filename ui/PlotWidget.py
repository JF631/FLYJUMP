import matplotlib
import numpy as np
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import QVBoxLayout, QWidget
from qwt import QwtLegend, QwtPlot, QwtPlotCurve

matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from ljanalyzer.video import VideoSignals
from utils.controlsignals import ControlSignals


class MultiPlot(QWidget):
    """
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
    """

    def __init__(
        self,
        signals: VideoSignals,
        num_plots: int = 1,
        curves: dict = {},
        parent: QWidget | None = ...,
    ) -> None:
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
            plot.set_data(data[:, curve_offset : curve_offset + plot_curves])
            curve_offset += plot_curves


class MatplotCanvas(FigureCanvasQTAgg):
    """
    Class that allows to embed a simple matplotlib plot inside the GUI.
    It currently supports plotting simple 2D data via plot2D().
    """

    def __init__(
        self,
        parent=None,
        width=0.1,
        height=3,
        dpi=100,
        x_label=None,
        y_label=None,
        control_signals: ControlSignals = None,
    ):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        # fig.patch.set_alpha(0)
        # self.axes.set_alpha(0)
        self.x_values: np.ndarray = None
        self.y_values: np.ndarray = None
        self.x_label: str = x_label
        self.y_label: str = y_label
        self.selected_frame = None
        self.hover_frame: int = 0
        self.control_signals = control_signals
        if self.control_signals:
            self.control_signals.jump_to_frame.connect(
                self.update_current_frame_indicator
            )
        super(MatplotCanvas, self).__init__(fig)
        self.mpl_connect("button_press_event", self.on_click)
        self.mpl_connect("motion_notify_event", self.on_hover)

        self.annotation = self.axes.annotate(
            "",
            xy=(0, 0),
            xytext=(20, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
        )

    def delete(self):
        '''
        Removes the canvas from the GUI.
        '''
        self.setParent(None)
        self.deleteLater()

    def plot2D(self, *values, label=None):
        """
        creates a simple 2d plot.

        Parameters
        ----------
        values : tuple
            list like values to plot.
            provide x and y values. if no x values are provided, they will
            simply be fitted to match the y value length.
        label : str
            label for the current line that should be plotted.
        """
        num_values = len(values)
        if num_values > 2 or num_values < 1:
            return
        if num_values == 1:
            self.y_values = np.array(values[0])
            self.x_values = np.arange(len(self.y_values))
        if num_values == 2:
            self.x_values, self.y_values = values
            self.x_values = np.array(self.x_values)
            self.y_values = np.array(self.y_values)
        if not self.x_values.shape == self.y_values.shape:
            return
        self.axes.plot(self.x_values, self.y_values, label=label)
        self.axes.legend()
        self.axes.set_xlabel(self.x_label)
        self.axes.set_ylabel(self.y_label)
        self.draw()

    def on_click(self, event):
        """
        handles clicks on the plot.
        emits jump_to_frame(int) control signal.
        """
        if event.inaxes == self.axes:
            x_clicked = int(event.xdata)
            if self.control_signals:
                self.control_signals.jump_to_frame.emit(x_clicked)

    def on_hover(self, event):
        '''
        handles mouse events when hovering over the plot.
        '''
        if event.inaxes == self.axes:
            x, _ = max(0, int(event.xdata)), max(0, int(event.ydata))
            self.show_hover_hint(x)

    def show_hover_hint(self, frame_num: int):
        """
        updates hover indicator.

        Parameters
        ----------
        frame_num : int
            frame that should be indicated.
        """
        if self.y_values is None:
            return
        if frame_num > len(self.y_values) - 1:
            return
        x_value = frame_num
        y_value = self.y_values[x_value]
        if self.hover_frame:
            self.hover_frame.remove()
        self.hover_frame = self.axes.plot(
            x_value, y_value, "o", color="gray", markersize=4
        )[0]
        self.draw()

    def update_current_frame_indicator(self, frame: int):
        """
        updates indicator that corresponds to the currently shown video frame.
        updates are triggered via jump_to_frame control signal.

        Parameters
        ----------
        frame : int
            Current frame that should be indicated.
        """
        if self.y_values is None:
            return
        if frame > len(self.y_values) - 1:
            return
        x_value = frame
        y_value = self.y_values[x_value]
        if self.selected_frame:
            self.selected_frame.remove()
        self.selected_frame = self.axes.plot(x_value, y_value, "ro", markersize=8)[0]
        self.draw()

    def clear(self):
        """
        removes all currently plotted data from plot.
        """
        self.axes.clear()

    def add_points(self, x_values, label=None):
        """
        adds red dots to 2D plot at given x positions.
        plot2D() must have been invoked before

        Parameters
        ----------
        x_values : list
            x values at which the dots should be scattered.
        label : str
            the data label that should be shown in matplotlib legend.
        """
        if x_values is None:
            return
        x_values = np.array(x_values)
        y_values = self.y_values[x_values]
        if not y_values.shape == x_values.shape:
            return
        self.axes.scatter(x_values, y_values, color="red", marker="o", label=label)
        for x, y in zip(x_values, y_values):
            self.axes.annotate("({}, {:.2f})".format(x, y), (x, y))
        self.axes.legend()
        self.draw()


class Plot(QWidget):
    """
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
    """

    def __init__(
        self,
        title: str = "",
        num_curves: int = 1,
        curve_titles: list = ["curve1"],
        parent: QWidget | None = ...,
    ) -> None:
        super().__init__(parent)
        assert len(curve_titles) == num_curves
        self.plot = QwtPlot(self)
        self.legend = QwtLegend()
        self.max_rows = 128
        self.current_row = 0
        self.num_curves = num_curves
        self.data = np.empty((self.max_rows, num_curves), dtype="f4")
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
        """
        updates all curves in current plot.
        The x values are expected to be in time domain and are automatically
        set.

        Parameters
        -----------
        data : np.ndarray
            matrix of shape (values_to_add, num_curves).
            Each colum thereby must hold new data for one curve.
        """
        columns = self.data.shape[1]
        num_new_values = data.shape[0]
        if self.current_row + num_new_values >= self.data.shape[0]:
            self.max_rows *= 2
            self.data.resize((self.max_rows, columns))
        self.data[[self.current_row, self.current_row + num_new_values]] = data
        self.current_row += num_new_values
        for i, curve in enumerate(self.curves):
            curve.setData(np.arange(self.current_row), self.data[:, i])
        self.plot.setAxisAutoScale(QwtPlot.xBottom)
        self.plot.setAxisAutoScale(QwtPlot.yLeft)
        self.plot.replot()


class PlotWidget(QWidget):
    def __init__(self, signals: VideoSignals, parent: QWidget | None = ...) -> None:
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
        layout = QVBoxLayout()
        layout.addWidget(self.plot)
        self.setLayout(layout)
        self.plot.setAxisTitle(QwtPlot.xBottom, "X Axis")
        self.plot.setAxisTitle(QwtPlot.yLeft, "Y Axis")
        self.l_foot_curve.attach(self.plot)
        self.r_foot_curve.attach(self.plot)

    @pyqtSlot(np.ndarray)
    def set_data(self, data: np.ndarray):
        if np.any(data < 0.0) or np.any(data > 1.0):
            return
        self.l_foot_y = np.append(self.l_foot_y, data[1, 0])
        self.r_foot_y = np.append(self.r_foot_y, data[1, 1])
        self.l_foot_curve.setData(np.arange(len(self.l_foot_y)), self.l_foot_y)
        self.r_foot_curve.setData(np.arange(len(self.r_foot_y)), self.r_foot_y)
        self.plot.setAxisAutoScale(QwtPlot.xBottom)
        self.plot.setAxisAutoScale(QwtPlot.yLeft)
        self.plot.replot()
