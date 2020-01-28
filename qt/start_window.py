# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'start_window.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(676, 242)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(229, 30, 232, 64))
        self.widget.setObjectName("widget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label1 = QtWidgets.QLabel(self.widget)
        self.label1.setAlignment(QtCore.Qt.AlignCenter)
        self.label1.setObjectName("label1")
        self.verticalLayout_2.addWidget(self.label1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.filetypeBox = QtWidgets.QComboBox(self.widget)
        self.filetypeBox.setObjectName("filetypeBox")
        self.verticalLayout.addWidget(self.filetypeBox)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.locatefileButtom = QtWidgets.QPushButton(self.widget)
        self.locatefileButtom.setObjectName("locatefileButtom")
        self.horizontalLayout.addWidget(self.locatefileButtom)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.widget1 = QtWidgets.QWidget(self.centralwidget)
        self.widget1.setGeometry(QtCore.QRect(279, 164, 106, 66))
        self.widget1.setObjectName("widget1")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.widget1)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.loadButton = QtWidgets.QPushButton(self.widget1)
        self.loadButton.setObjectName("loadButton")
        self.verticalLayout_3.addWidget(self.loadButton)
        self.closeButton = QtWidgets.QPushButton(self.widget1)
        self.closeButton.setObjectName("closeButton")
        self.verticalLayout_3.addWidget(self.closeButton)
        self.fileLabel = QtWidgets.QLabel(self.centralwidget)
        self.fileLabel.setGeometry(QtCore.QRect(10, 105, 651, 51))
        self.fileLabel.setText("")
        self.fileLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.fileLabel.setObjectName("fileLabel")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label1.setText(_translate("MainWindow", "Select file type:"))
        self.locatefileButtom.setText(_translate("MainWindow", "Locate Scan"))
        self.loadButton.setText(_translate("MainWindow", "Load Scan"))
        self.closeButton.setText(_translate("MainWindow", "Close"))
