# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'segmentation.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Segmentation(object):
    def setupUi(self, Segmentation):
        Segmentation.setObjectName("Segmentation")
        Segmentation.resize(991, 694)
        self.mpl_ct = MplSliceWidget(Segmentation)
        self.mpl_ct.setGeometry(QtCore.QRect(10, 9, 550, 570))
        self.mpl_ct.setObjectName("mpl_ct")
        self.sliceScrollbar = QtWidgets.QScrollBar(Segmentation)
        self.sliceScrollbar.setGeometry(QtCore.QRect(600, 30, 16, 471))
        self.sliceScrollbar.setMinimum(1)
        self.sliceScrollbar.setMaximum(120)
        self.sliceScrollbar.setProperty("value", 60)
        self.sliceScrollbar.setOrientation(QtCore.Qt.Vertical)
        self.sliceScrollbar.setObjectName("sliceScrollbar")
        self.displayButton = QtWidgets.QPushButton(Segmentation)
        self.displayButton.setGeometry(QtCore.QRect(429, 620, 116, 32))
        self.displayButton.setObjectName("displayButton")
        self.savenameLabel = QtWidgets.QLabel(Segmentation)
        self.savenameLabel.setGeometry(QtCore.QRect(520, 660, 441, 20))
        self.savenameLabel.setText("")
        self.savenameLabel.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.savenameLabel.setObjectName("savenameLabel")
        self.layoutWidget = QtWidgets.QWidget(Segmentation)
        self.layoutWidget.setGeometry(QtCore.QRect(620, 230, 61, 47))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.sliceLabel = QtWidgets.QLabel(self.layoutWidget)
        self.sliceLabel.setObjectName("sliceLabel")
        self.verticalLayout_2.addWidget(self.sliceLabel)
        self.slice = QtWidgets.QLineEdit(self.layoutWidget)
        self.slice.setObjectName("slice")
        self.verticalLayout_2.addWidget(self.slice)
        self.layoutWidget1 = QtWidgets.QWidget(Segmentation)
        self.layoutWidget1.setGeometry(QtCore.QRect(250, 581, 172, 82))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.layoutWidget1)
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.pixellimsLabel = QtWidgets.QLabel(self.layoutWidget1)
        self.pixellimsLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.pixellimsLabel.setObjectName("pixellimsLabel")
        self.verticalLayout_7.addWidget(self.pixellimsLabel)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.vminLabel = QtWidgets.QLabel(self.layoutWidget1)
        self.vminLabel.setObjectName("vminLabel")
        self.verticalLayout_3.addWidget(self.vminLabel)
        self.vmaxLabel = QtWidgets.QLabel(self.layoutWidget1)
        self.vmaxLabel.setObjectName("vmaxLabel")
        self.verticalLayout_3.addWidget(self.vmaxLabel)
        self.horizontalLayout.addLayout(self.verticalLayout_3)
        self.verticalLayout_1 = QtWidgets.QVBoxLayout()
        self.verticalLayout_1.setObjectName("verticalLayout_1")
        self.vmin = QtWidgets.QLineEdit(self.layoutWidget1)
        self.vmin.setObjectName("vmin")
        self.verticalLayout_1.addWidget(self.vmin)
        self.vmax = QtWidgets.QLineEdit(self.layoutWidget1)
        self.vmax.setObjectName("vmax")
        self.verticalLayout_1.addWidget(self.vmax)
        self.horizontalLayout.addLayout(self.verticalLayout_1)
        self.verticalLayout_7.addLayout(self.horizontalLayout)
        self.statusLabel = QtWidgets.QLabel(Segmentation)
        self.statusLabel.setGeometry(QtCore.QRect(700, 500, 261, 20))
        self.statusLabel.setText("")
        self.statusLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.statusLabel.setObjectName("statusLabel")
        self.layoutWidget2 = QtWidgets.QWidget(Segmentation)
        self.layoutWidget2.setGeometry(QtCore.QRect(760, 530, 140, 71))
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.layoutWidget2)
        self.verticalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.saveButton = QtWidgets.QPushButton(self.layoutWidget2)
        self.saveButton.setObjectName("saveButton")
        self.verticalLayout_8.addWidget(self.saveButton)
        self.closeButton = QtWidgets.QPushButton(self.layoutWidget2)
        self.closeButton.setObjectName("closeButton")
        self.verticalLayout_8.addWidget(self.closeButton)
        self.showCubeBox = QtWidgets.QCheckBox(Segmentation)
        self.showCubeBox.setGeometry(QtCore.QRect(60, 590, 151, 20))
        self.showCubeBox.setObjectName("showCubeBox")
        self.widget = QtWidgets.QWidget(Segmentation)
        self.widget.setGeometry(QtCore.QRect(659, 90, 273, 71))
        self.widget.setObjectName("widget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.stNameLine = QtWidgets.QLineEdit(self.widget)
        self.stNameLine.setObjectName("stNameLine")
        self.horizontalLayout_2.addWidget(self.stNameLine)
        self.createStButton = QtWidgets.QPushButton(self.widget)
        self.createStButton.setObjectName("createStButton")
        self.horizontalLayout_2.addWidget(self.createStButton)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setObjectName("label")
        self.horizontalLayout_3.addWidget(self.label)
        self.StBox = QtWidgets.QComboBox(self.widget)
        self.StBox.setObjectName("StBox")
        self.horizontalLayout_3.addWidget(self.StBox)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.widget1 = QtWidgets.QWidget(Segmentation)
        self.widget1.setGeometry(QtCore.QRect(710, 310, 131, 66))
        self.widget1.setObjectName("widget1")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.widget1)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.undoButton = QtWidgets.QPushButton(self.widget1)
        self.undoButton.setObjectName("undoButton")
        self.verticalLayout_4.addWidget(self.undoButton)
        self.clearButton = QtWidgets.QPushButton(self.widget1)
        self.clearButton.setObjectName("clearButton")
        self.verticalLayout_4.addWidget(self.clearButton)

        self.retranslateUi(Segmentation)
        QtCore.QMetaObject.connectSlotsByName(Segmentation)

    def retranslateUi(self, Segmentation):
        _translate = QtCore.QCoreApplication.translate
        Segmentation.setWindowTitle(_translate("Segmentation", "Segmentation"))
        self.displayButton.setText(_translate("Segmentation", "Display"))
        self.sliceLabel.setText(_translate("Segmentation", "z-slice:"))
        self.slice.setText(_translate("Segmentation", "60"))
        self.pixellimsLabel.setText(_translate("Segmentation", "Pixel Limits"))
        self.vminLabel.setText(_translate("Segmentation", "Min:"))
        self.vmaxLabel.setText(_translate("Segmentation", "Max:"))
        self.vmin.setText(_translate("Segmentation", "-400"))
        self.vmax.setText(_translate("Segmentation", "400"))
        self.saveButton.setText(_translate("Segmentation", "Save dcm"))
        self.closeButton.setText(_translate("Segmentation", "Close"))
        self.showCubeBox.setText(_translate("Segmentation", "Show Edit Cubes"))
        self.createStButton.setText(_translate("Segmentation", "Create Structure"))
        self.label.setText(_translate("Segmentation", "Current Structure:"))
        self.undoButton.setText(_translate("Segmentation", "Undo Point"))
        self.clearButton.setText(_translate("Segmentation", "Clear Points"))
from mplslicewidget import MplSliceWidget
