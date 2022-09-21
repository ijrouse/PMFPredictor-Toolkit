# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'dashboard.ui'
##
## Created by: Qt User Interface Compiler version 6.3.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QFrame, QGridLayout,
    QLabel, QLineEdit, QMainWindow, QMenu,
    QMenuBar, QPushButton, QSizePolicy, QStatusBar,
    QTextEdit, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(884, 932)
        icon = QIcon()
        icon.addFile(u"pmfpredictorgui/pmfp.ico", QSize(), QIcon.Normal, QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.actionClose = QAction(MainWindow)
        self.actionClose.setObjectName(u"actionClose")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.textEdit = QTextEdit(self.centralwidget)
        self.textEdit.setObjectName(u"textEdit")
        self.textEdit.setGeometry(QRect(20, 360, 841, 501))
        self.textEdit.setReadOnly(True)
        self.label_7 = QLabel(self.centralwidget)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setGeometry(QRect(560, 40, 291, 41))
        self.label_8 = QLabel(self.centralwidget)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setGeometry(QRect(540, 120, 121, 21))
        self.label_9 = QLabel(self.centralwidget)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setGeometry(QRect(540, 200, 101, 21))
        self.text_newNPName = QLineEdit(self.centralwidget)
        self.text_newNPName.setObjectName(u"text_newNPName")
        self.text_newNPName.setGeometry(QRect(670, 200, 181, 21))
        self.button_buildNP = QPushButton(self.centralwidget)
        self.button_buildNP.setObjectName(u"button_buildNP")
        self.button_buildNP.setGeometry(QRect(640, 240, 89, 25))
        self.text_newNPLocation = QLineEdit(self.centralwidget)
        self.text_newNPLocation.setObjectName(u"text_newNPLocation")
        self.text_newNPLocation.setGeometry(QRect(670, 120, 181, 21))
        self.button_FindNPDownloadFolder = QPushButton(self.centralwidget)
        self.button_FindNPDownloadFolder.setObjectName(u"button_FindNPDownloadFolder")
        self.button_FindNPDownloadFolder.setGeometry(QRect(670, 150, 89, 25))
        self.label_10 = QLabel(self.centralwidget)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setGeometry(QRect(120, 40, 291, 41))
        self.widget = QWidget(self.centralwidget)
        self.widget.setObjectName(u"widget")
        self.widget.setGeometry(QRect(40, 120, 450, 217))
        self.gridLayout = QGridLayout(self.widget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(self.widget)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.b_scanChem = QCheckBox(self.widget)
        self.b_scanChem.setObjectName(u"b_scanChem")
        self.b_scanChem.setChecked(True)

        self.gridLayout.addWidget(self.b_scanChem, 0, 1, 1, 1)

        self.button_runAC = QPushButton(self.widget)
        self.button_runAC.setObjectName(u"button_runAC")

        self.gridLayout.addWidget(self.button_runAC, 0, 2, 1, 1)

        self.line = QFrame(self.widget)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.gridLayout.addWidget(self.line, 1, 0, 1, 3)

        self.label_2 = QLabel(self.widget)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 2, 0, 2, 1)

        self.button_runGenChemPot = QPushButton(self.widget)
        self.button_runGenChemPot.setObjectName(u"button_runGenChemPot")

        self.gridLayout.addWidget(self.button_runGenChemPot, 2, 2, 2, 1)

        self.b_refreshChemPot = QCheckBox(self.widget)
        self.b_refreshChemPot.setObjectName(u"b_refreshChemPot")

        self.gridLayout.addWidget(self.b_refreshChemPot, 3, 1, 1, 1)

        self.label_3 = QLabel(self.widget)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout.addWidget(self.label_3, 4, 0, 1, 1)

        self.b_refreshSurfPot = QCheckBox(self.widget)
        self.b_refreshSurfPot.setObjectName(u"b_refreshSurfPot")

        self.gridLayout.addWidget(self.b_refreshSurfPot, 4, 1, 1, 1)

        self.button_runGenSurfPot = QPushButton(self.widget)
        self.button_runGenSurfPot.setObjectName(u"button_runGenSurfPot")

        self.gridLayout.addWidget(self.button_runGenSurfPot, 4, 2, 1, 1)

        self.line_2 = QFrame(self.widget)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.HLine)
        self.line_2.setFrameShadow(QFrame.Sunken)

        self.gridLayout.addWidget(self.line_2, 5, 0, 1, 3)

        self.label_4 = QLabel(self.widget)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout.addWidget(self.label_4, 6, 0, 2, 1)

        self.button_runHGEChem = QPushButton(self.widget)
        self.button_runHGEChem.setObjectName(u"button_runHGEChem")

        self.gridLayout.addWidget(self.button_runHGEChem, 6, 2, 2, 1)

        self.b_refreshHGChem = QCheckBox(self.widget)
        self.b_refreshHGChem.setObjectName(u"b_refreshHGChem")
        self.b_refreshHGChem.setChecked(True)

        self.gridLayout.addWidget(self.b_refreshHGChem, 7, 1, 1, 1)

        self.label_5 = QLabel(self.widget)
        self.label_5.setObjectName(u"label_5")

        self.gridLayout.addWidget(self.label_5, 8, 0, 1, 1)

        self.b_refreshHGSurf = QCheckBox(self.widget)
        self.b_refreshHGSurf.setObjectName(u"b_refreshHGSurf")
        self.b_refreshHGSurf.setChecked(True)

        self.gridLayout.addWidget(self.b_refreshHGSurf, 8, 1, 1, 1)

        self.button_runHGESurf = QPushButton(self.widget)
        self.button_runHGESurf.setObjectName(u"button_runHGESurf")

        self.gridLayout.addWidget(self.button_runHGESurf, 8, 2, 1, 1)

        self.line_3 = QFrame(self.widget)
        self.line_3.setObjectName(u"line_3")
        self.line_3.setFrameShape(QFrame.HLine)
        self.line_3.setFrameShadow(QFrame.Sunken)

        self.gridLayout.addWidget(self.line_3, 9, 0, 1, 3)

        self.label_6 = QLabel(self.widget)
        self.label_6.setObjectName(u"label_6")

        self.gridLayout.addWidget(self.label_6, 10, 0, 1, 1)

        self.button_runPredict = QPushButton(self.widget)
        self.button_runPredict.setObjectName(u"button_runPredict")

        self.gridLayout.addWidget(self.button_runPredict, 10, 2, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 884, 22))
        self.menuMenu = QMenu(self.menubar)
        self.menuMenu.setObjectName(u"menuMenu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuMenu.menuAction())
        self.menuMenu.addAction(self.actionClose)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"PMFPredictor Dashboard", None))
        self.actionClose.setText(QCoreApplication.translate("MainWindow", u"Close", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-size:20pt; font-weight:600;\">NP from Charmm-GUI</span></p></body></html>", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"Download folder", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"New NP name", None))
        self.text_newNPName.setText(QCoreApplication.translate("MainWindow", u"newnp-001", None))
        self.button_buildNP.setText(QCoreApplication.translate("MainWindow", u"Build NP", None))
        self.text_newNPLocation.setText(QCoreApplication.translate("MainWindow", u"newnp-001", None))
        self.button_FindNPDownloadFolder.setText(QCoreApplication.translate("MainWindow", u"Open", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-size:20pt; font-weight:600;\">PMF Predictor Scripts</span></p></body></html>", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Generate missing chemical structures", None))
        self.b_scanChem.setText(QCoreApplication.translate("MainWindow", u"Scan", None))
        self.button_runAC.setText(QCoreApplication.translate("MainWindow", u"Run", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Generate chemical potentials", None))
        self.button_runGenChemPot.setText(QCoreApplication.translate("MainWindow", u"Run", None))
        self.b_refreshChemPot.setText(QCoreApplication.translate("MainWindow", u"Refresh?", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Generate surface potentials", None))
        self.b_refreshSurfPot.setText(QCoreApplication.translate("MainWindow", u"Refresh?", None))
        self.button_runGenSurfPot.setText(QCoreApplication.translate("MainWindow", u"Run", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Generate chemical potential expansions", None))
        self.button_runHGEChem.setText(QCoreApplication.translate("MainWindow", u"Run", None))
        self.b_refreshHGChem.setText(QCoreApplication.translate("MainWindow", u"Refresh?", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Generate surface potential expansions", None))
        self.b_refreshHGSurf.setText(QCoreApplication.translate("MainWindow", u"Refresh?", None))
        self.button_runHGESurf.setText(QCoreApplication.translate("MainWindow", u"Run", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"Generate PMFs", None))
        self.button_runPredict.setText(QCoreApplication.translate("MainWindow", u"Run", None))
        self.menuMenu.setTitle(QCoreApplication.translate("MainWindow", u"Menu", None))
    # retranslateUi

