from PyQt5.QtCore import QObject, pyqtSignal

class ControlSignals(QObject):
    terminate = pyqtSignal()