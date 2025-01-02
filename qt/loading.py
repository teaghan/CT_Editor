# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'loading.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Loading(object):
    def setupUi(self, Loading):
        Loading.setObjectName("Loading")
        Loading.resize(812, 145)
        self.window_text = QtWidgets.QLabel(Loading)
        self.window_text.setGeometry(QtCore.QRect(60, 50, 691, 20))
        self.window_text.setAlignment(QtCore.Qt.AlignCenter)
        self.window_text.setObjectName("window_text")

        self.retranslateUi(Loading)
        QtCore.QMetaObject.connectSlotsByName(Loading)

    def retranslateUi(self, Loading):
        _translate = QtCore.QCoreApplication.translate
        Loading.setWindowTitle(_translate("Loading", "Loading"))
        self.window_text.setText(_translate("Loading", "Loading..."))
