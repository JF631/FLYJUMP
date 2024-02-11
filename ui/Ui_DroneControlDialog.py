# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\DroneControlDialog.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(499, 483)
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout(Dialog)
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.splitter_2 = QtWidgets.QSplitter(Dialog)
        self.splitter_2.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_2.setObjectName("splitter_2")
        self.widget = QtWidgets.QWidget(self.splitter_2)
        self.widget.setObjectName("widget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.up_btn = QtWidgets.QPushButton(self.widget)
        self.up_btn.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(".\\icons/arrow_up.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.up_btn.setIcon(icon)
        self.up_btn.setObjectName("up_btn")
        self.verticalLayout_2.addWidget(self.up_btn)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.left_btn = QtWidgets.QPushButton(self.widget)
        self.left_btn.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(".\\icons/arrow_left.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.left_btn.setIcon(icon1)
        self.left_btn.setObjectName("left_btn")
        self.horizontalLayout.addWidget(self.left_btn)
        self.right_btn = QtWidgets.QPushButton(self.widget)
        self.right_btn.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(".\\icons/arrow_right.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.right_btn.setIcon(icon2)
        self.right_btn.setObjectName("right_btn")
        self.horizontalLayout.addWidget(self.right_btn)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.down_btn = QtWidgets.QPushButton(self.widget)
        self.down_btn.setText("")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(".\\icons/arrow_down.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.down_btn.setIcon(icon3)
        self.down_btn.setObjectName("down_btn")
        self.verticalLayout_2.addWidget(self.down_btn)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.rtl_button = QtWidgets.QPushButton(self.widget)
        self.rtl_button.setObjectName("rtl_button")
        self.horizontalLayout_2.addWidget(self.rtl_button)
        self.land_button = QtWidgets.QPushButton(self.widget)
        self.land_button.setObjectName("land_button")
        self.horizontalLayout_2.addWidget(self.land_button)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.takeoff_button = QtWidgets.QPushButton(self.widget)
        self.takeoff_button.setObjectName("takeoff_button")
        self.verticalLayout_2.addWidget(self.takeoff_button)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.com_combobox = QtWidgets.QComboBox(self.widget)
        self.com_combobox.setToolTip("")
        self.com_combobox.setEditable(False)
        self.com_combobox.setCurrentText("")
        self.com_combobox.setObjectName("com_combobox")
        self.horizontalLayout_4.addWidget(self.com_combobox)
        self.connect_button = QtWidgets.QPushButton(self.widget)
        self.connect_button.setObjectName("connect_button")
        self.horizontalLayout_4.addWidget(self.connect_button)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem1)
        self.splitter = QtWidgets.QSplitter(self.splitter_2)
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setObjectName("splitter")
        self.layoutWidget = QtWidgets.QWidget(self.splitter)
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_clear_messages = QtWidgets.QLabel(self.layoutWidget)
        self.label_clear_messages.setObjectName("label_clear_messages")
        self.verticalLayout_3.addWidget(self.label_clear_messages)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.clear_msg_button = QtWidgets.QPushButton(self.layoutWidget)
        self.clear_msg_button.setObjectName("clear_msg_button")
        self.horizontalLayout_3.addWidget(self.clear_msg_button)
        spacerItem2 = QtWidgets.QSpacerItem(18, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem2)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        self.scrollArea = QtWidgets.QScrollArea(self.layoutWidget)
        self.scrollArea.setMinimumSize(QtCore.QSize(100, 50))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 189, 111))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_arm_checks = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_arm_checks.setText("")
        self.label_arm_checks.setObjectName("label_arm_checks")
        self.verticalLayout.addWidget(self.label_arm_checks)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.verticalLayout_3.addWidget(self.scrollArea)
        self.gps_groupbox = QtWidgets.QGroupBox(self.splitter)
        self.gps_groupbox.setMinimumSize(QtCore.QSize(100, 30))
        self.gps_groupbox.setObjectName("gps_groupbox")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.gps_groupbox)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_height = QtWidgets.QLabel(self.gps_groupbox)
        self.label_height.setObjectName("label_height")
        self.horizontalLayout_5.addWidget(self.label_height)
        self.label_height_m = QtWidgets.QLabel(self.gps_groupbox)
        self.label_height_m.setObjectName("label_height_m")
        self.horizontalLayout_5.addWidget(self.label_height_m)
        self.verticalLayout_4.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_velocity = QtWidgets.QLabel(self.gps_groupbox)
        self.label_velocity.setObjectName("label_velocity")
        self.horizontalLayout_6.addWidget(self.label_velocity)
        self.label_velocity_ms = QtWidgets.QLabel(self.gps_groupbox)
        self.label_velocity_ms.setObjectName("label_velocity_ms")
        self.horizontalLayout_6.addWidget(self.label_velocity_ms)
        self.verticalLayout_4.addLayout(self.horizontalLayout_6)
        self.line = QtWidgets.QFrame(self.gps_groupbox)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout_4.addWidget(self.line)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_satelites = QtWidgets.QLabel(self.gps_groupbox)
        self.label_satelites.setObjectName("label_satelites")
        self.horizontalLayout_7.addWidget(self.label_satelites)
        self.label_satelites_count = QtWidgets.QLabel(self.gps_groupbox)
        self.label_satelites_count.setObjectName("label_satelites_count")
        self.horizontalLayout_7.addWidget(self.label_satelites_count)
        self.verticalLayout_4.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label_fix_type = QtWidgets.QLabel(self.gps_groupbox)
        self.label_fix_type.setObjectName("label_fix_type")
        self.horizontalLayout_8.addWidget(self.label_fix_type)
        self.verticalLayout_4.addLayout(self.horizontalLayout_8)
        self.battery_groupbox = QtWidgets.QGroupBox(self.splitter)
        self.battery_groupbox.setMinimumSize(QtCore.QSize(100, 30))
        self.battery_groupbox.setObjectName("battery_groupbox")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.battery_groupbox)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.label_voltage = QtWidgets.QLabel(self.battery_groupbox)
        self.label_voltage.setObjectName("label_voltage")
        self.horizontalLayout_9.addWidget(self.label_voltage)
        self.label_voltage_v = QtWidgets.QLabel(self.battery_groupbox)
        self.label_voltage_v.setObjectName("label_voltage_v")
        self.horizontalLayout_9.addWidget(self.label_voltage_v)
        self.verticalLayout_5.addLayout(self.horizontalLayout_9)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.label_consumed_current = QtWidgets.QLabel(self.battery_groupbox)
        self.label_consumed_current.setObjectName("label_consumed_current")
        self.horizontalLayout_10.addWidget(self.label_consumed_current)
        self.label_consumed_current_mah = QtWidgets.QLabel(self.battery_groupbox)
        self.label_consumed_current_mah.setObjectName("label_consumed_current_mah")
        self.horizontalLayout_10.addWidget(self.label_consumed_current_mah)
        self.verticalLayout_5.addLayout(self.horizontalLayout_10)
        self.horizontalLayout_11.addWidget(self.splitter_2)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.up_btn.setToolTip(_translate("Dialog", "<html><head/><body><p><br/></p></body></html>"))
        self.left_btn.setToolTip(_translate("Dialog", "<html><head/><body><p><br/></p></body></html>"))
        self.right_btn.setToolTip(_translate("Dialog", "<html><head/><body><p><br/></p></body></html>"))
        self.down_btn.setToolTip(_translate("Dialog", "<html><head/><body><p><br/></p></body></html>"))
        self.rtl_button.setText(_translate("Dialog", "Return Home"))
        self.land_button.setText(_translate("Dialog", "Land"))
        self.takeoff_button.setText(_translate("Dialog", "Takeoff"))
        self.connect_button.setText(_translate("Dialog", "Connect"))
        self.label_clear_messages.setText(_translate("Dialog", "Arming checks"))
        self.clear_msg_button.setText(_translate("Dialog", "Clear"))
        self.gps_groupbox.setTitle(_translate("Dialog", "Info"))
        self.label_height.setText(_translate("Dialog", "height"))
        self.label_height_m.setText(_translate("Dialog", "[m]"))
        self.label_velocity.setText(_translate("Dialog", "velocity"))
        self.label_velocity_ms.setText(_translate("Dialog", "[m/s]"))
        self.label_satelites.setText(_translate("Dialog", "satelites"))
        self.label_satelites_count.setText(_translate("Dialog", "[count]"))
        self.label_fix_type.setText(_translate("Dialog", "fix type"))
        self.battery_groupbox.setTitle(_translate("Dialog", "Battery"))
        self.label_voltage.setText(_translate("Dialog", "voltage"))
        self.label_voltage_v.setText(_translate("Dialog", "[V]"))
        self.label_consumed_current.setText(_translate("Dialog", "consumed"))
        self.label_consumed_current_mah.setText(_translate("Dialog", "[mAh]"))
