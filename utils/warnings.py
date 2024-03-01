"""
Module that provides several custom warnings.

Author: Jakob Faust (software_jaf@mx442.de)
Date: 2023-10-124
"""

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout


class PotentialRaceConditionWarning(ResourceWarning):
    """
    warning is shown if the software detects a risk of a possible race
    condition
    """

    pass


class WarningDialog(QDialog):
    def __init__(self, signals=None) -> None:
        super().__init__()
        self.setWindowTitle("Warning")
        QBtn = QDialogButtonBox.StandardButton.Ok
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        layout = QVBoxLayout()
        self.label = QLabel("Something went wrong!")
        layout.addWidget(self.label)
        layout.addWidget(self.buttonBox)
        self.setLayout(layout)
        if signals:
            signals.error.connect(self.warning_dialog)

    def show_warning(self, message):
        warning_text = f"""Warning: {message}"""
        self.label.setText(warning_text)
        self.exec()

    @pyqtSlot(str)
    def warning_dialog(self, message):
        # self.moveToThread(QApplication.instance().thread())
        warning_text = f"""I detected some potential error: {message}"""
        self.label.setText(warning_text)
        self.exec_()
