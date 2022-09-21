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
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFrame,
    QGridLayout, QLabel, QLineEdit, QMainWindow,
    QMenu, QMenuBar, QPushButton, QSizePolicy,
    QSpinBox, QStatusBar, QTabWidget, QTextEdit,
    QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(922, 932)
        icon = QIcon()
        icon.addFile(u"pmfp.ico", QSize(), QIcon.Normal, QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.actionClose = QAction(MainWindow)
        self.actionClose.setObjectName(u"actionClose")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.textEdit = QTextEdit(self.centralwidget)
        self.textEdit.setObjectName(u"textEdit")
        self.textEdit.setGeometry(QRect(20, 360, 881, 501))
        self.textEdit.setReadOnly(True)
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setGeometry(QRect(20, 10, 881, 331))
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.layoutWidget = QWidget(self.tab)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.layoutWidget.setGeometry(QRect(10, 60, 450, 217))
        self.gridLayout = QGridLayout(self.layoutWidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(self.layoutWidget)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.b_scanChem = QCheckBox(self.layoutWidget)
        self.b_scanChem.setObjectName(u"b_scanChem")
        self.b_scanChem.setCheckable(True)
        self.b_scanChem.setChecked(True)

        self.gridLayout.addWidget(self.b_scanChem, 0, 1, 1, 1)

        self.button_runAC = QPushButton(self.layoutWidget)
        self.button_runAC.setObjectName(u"button_runAC")

        self.gridLayout.addWidget(self.button_runAC, 0, 2, 1, 1)

        self.line = QFrame(self.layoutWidget)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.gridLayout.addWidget(self.line, 1, 0, 1, 3)

        self.label_2 = QLabel(self.layoutWidget)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 2, 0, 2, 1)

        self.button_runGenChemPot = QPushButton(self.layoutWidget)
        self.button_runGenChemPot.setObjectName(u"button_runGenChemPot")

        self.gridLayout.addWidget(self.button_runGenChemPot, 2, 2, 2, 1)

        self.b_refreshChemPot = QCheckBox(self.layoutWidget)
        self.b_refreshChemPot.setObjectName(u"b_refreshChemPot")

        self.gridLayout.addWidget(self.b_refreshChemPot, 3, 1, 1, 1)

        self.label_3 = QLabel(self.layoutWidget)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout.addWidget(self.label_3, 4, 0, 1, 1)

        self.b_refreshSurfPot = QCheckBox(self.layoutWidget)
        self.b_refreshSurfPot.setObjectName(u"b_refreshSurfPot")

        self.gridLayout.addWidget(self.b_refreshSurfPot, 4, 1, 1, 1)

        self.button_runGenSurfPot = QPushButton(self.layoutWidget)
        self.button_runGenSurfPot.setObjectName(u"button_runGenSurfPot")

        self.gridLayout.addWidget(self.button_runGenSurfPot, 4, 2, 1, 1)

        self.line_2 = QFrame(self.layoutWidget)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.HLine)
        self.line_2.setFrameShadow(QFrame.Sunken)

        self.gridLayout.addWidget(self.line_2, 5, 0, 1, 3)

        self.label_4 = QLabel(self.layoutWidget)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout.addWidget(self.label_4, 6, 0, 2, 1)

        self.button_runHGEChem = QPushButton(self.layoutWidget)
        self.button_runHGEChem.setObjectName(u"button_runHGEChem")

        self.gridLayout.addWidget(self.button_runHGEChem, 6, 2, 2, 1)

        self.b_refreshHGChem = QCheckBox(self.layoutWidget)
        self.b_refreshHGChem.setObjectName(u"b_refreshHGChem")
        self.b_refreshHGChem.setChecked(True)

        self.gridLayout.addWidget(self.b_refreshHGChem, 7, 1, 1, 1)

        self.label_5 = QLabel(self.layoutWidget)
        self.label_5.setObjectName(u"label_5")

        self.gridLayout.addWidget(self.label_5, 8, 0, 1, 1)

        self.b_refreshHGSurf = QCheckBox(self.layoutWidget)
        self.b_refreshHGSurf.setObjectName(u"b_refreshHGSurf")
        self.b_refreshHGSurf.setChecked(True)

        self.gridLayout.addWidget(self.b_refreshHGSurf, 8, 1, 1, 1)

        self.button_runHGESurf = QPushButton(self.layoutWidget)
        self.button_runHGESurf.setObjectName(u"button_runHGESurf")

        self.gridLayout.addWidget(self.button_runHGESurf, 8, 2, 1, 1)

        self.line_3 = QFrame(self.layoutWidget)
        self.line_3.setObjectName(u"line_3")
        self.line_3.setFrameShape(QFrame.HLine)
        self.line_3.setFrameShadow(QFrame.Sunken)

        self.gridLayout.addWidget(self.line_3, 9, 0, 1, 3)

        self.label_6 = QLabel(self.layoutWidget)
        self.label_6.setObjectName(u"label_6")

        self.gridLayout.addWidget(self.label_6, 10, 0, 1, 1)

        self.button_runPredict = QPushButton(self.layoutWidget)
        self.button_runPredict.setObjectName(u"button_runPredict")

        self.gridLayout.addWidget(self.button_runPredict, 10, 2, 1, 1)

        self.label_10 = QLabel(self.tab)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setGeometry(QRect(30, 10, 291, 41))
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.button_FindNPDownloadFolder = QPushButton(self.tab_2)
        self.button_FindNPDownloadFolder.setObjectName(u"button_FindNPDownloadFolder")
        self.button_FindNPDownloadFolder.setGeometry(QRect(160, 100, 89, 25))
        self.text_newNPLocation = QLineEdit(self.tab_2)
        self.text_newNPLocation.setObjectName(u"text_newNPLocation")
        self.text_newNPLocation.setGeometry(QRect(160, 70, 181, 21))
        self.label_7 = QLabel(self.tab_2)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setGeometry(QRect(50, 20, 351, 41))
        self.text_newNPName = QLineEdit(self.tab_2)
        self.text_newNPName.setObjectName(u"text_newNPName")
        self.text_newNPName.setGeometry(QRect(160, 130, 191, 21))
        self.label_8 = QLabel(self.tab_2)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setGeometry(QRect(30, 70, 121, 21))
        self.label_9 = QLabel(self.tab_2)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setGeometry(QRect(30, 130, 101, 21))
        self.button_buildNP = QPushButton(self.tab_2)
        self.button_buildNP.setObjectName(u"button_buildNP")
        self.button_buildNP.setGeometry(QRect(160, 260, 89, 31))
        self.label_11 = QLabel(self.tab_2)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setGeometry(QRect(30, 160, 67, 17))
        self.label_12 = QLabel(self.tab_2)
        self.label_12.setObjectName(u"label_12")
        self.label_12.setGeometry(QRect(30, 190, 67, 17))
        self.label_13 = QLabel(self.tab_2)
        self.label_13.setObjectName(u"label_13")
        self.label_13.setGeometry(QRect(30, 220, 67, 17))
        self.cb_npShape = QComboBox(self.tab_2)
        self.cb_npShape.addItem("")
        self.cb_npShape.addItem("")
        self.cb_npShape.setObjectName(u"cb_npShape")
        self.cb_npShape.setGeometry(QRect(160, 160, 191, 25))
        self.cb_npSource = QComboBox(self.tab_2)
        self.cb_npSource.addItem("")
        self.cb_npSource.addItem("")
        self.cb_npSource.addItem("")
        self.cb_npSource.setObjectName(u"cb_npSource")
        self.cb_npSource.setGeometry(QRect(160, 190, 191, 25))
        self.cb_npSSD = QComboBox(self.tab_2)
        self.cb_npSSD.addItem("")
        self.cb_npSSD.addItem("")
        self.cb_npSSD.addItem("")
        self.cb_npSSD.addItem("")
        self.cb_npSSD.setObjectName(u"cb_npSSD")
        self.cb_npSSD.setGeometry(QRect(160, 220, 191, 25))
        self.label_14 = QLabel(self.tab_2)
        self.label_14.setObjectName(u"label_14")
        self.label_14.setGeometry(QRect(510, 20, 291, 41))
        self.text_newChemID = QLineEdit(self.tab_2)
        self.text_newChemID.setObjectName(u"text_newChemID")
        self.text_newChemID.setGeometry(QRect(680, 70, 113, 25))
        self.text_newChemSmiles = QLineEdit(self.tab_2)
        self.text_newChemSmiles.setObjectName(u"text_newChemSmiles")
        self.text_newChemSmiles.setGeometry(QRect(680, 110, 113, 25))
        self.text_newChemDesc = QLineEdit(self.tab_2)
        self.text_newChemDesc.setObjectName(u"text_newChemDesc")
        self.text_newChemDesc.setGeometry(QRect(660, 190, 181, 25))
        self.label_15 = QLabel(self.tab_2)
        self.label_15.setObjectName(u"label_15")
        self.label_15.setGeometry(QRect(530, 70, 111, 21))
        self.label_16 = QLabel(self.tab_2)
        self.label_16.setObjectName(u"label_16")
        self.label_16.setGeometry(QRect(530, 110, 121, 21))
        self.label_17 = QLabel(self.tab_2)
        self.label_17.setObjectName(u"label_17")
        self.label_17.setGeometry(QRect(530, 150, 111, 21))
        self.spinBox_newChemCharge = QSpinBox(self.tab_2)
        self.spinBox_newChemCharge.setObjectName(u"spinBox_newChemCharge")
        self.spinBox_newChemCharge.setGeometry(QRect(670, 150, 44, 26))
        self.spinBox_newChemCharge.setMinimum(-5)
        self.spinBox_newChemCharge.setMaximum(5)
        self.label_18 = QLabel(self.tab_2)
        self.label_18.setObjectName(u"label_18")
        self.label_18.setGeometry(QRect(530, 190, 111, 21))
        self.button_addChem = QPushButton(self.tab_2)
        self.button_addChem.setObjectName(u"button_addChem")
        self.button_addChem.setGeometry(QRect(650, 250, 151, 31))
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QWidget()
        self.tab_3.setObjectName(u"tab_3")
        self.label_19 = QLabel(self.tab_3)
        self.label_19.setObjectName(u"label_19")
        self.label_19.setGeometry(QRect(20, 20, 851, 51))
        self.label_20 = QLabel(self.tab_3)
        self.label_20.setObjectName(u"label_20")
        self.label_20.setGeometry(QRect(20, 70, 831, 21))
        self.label_21 = QLabel(self.tab_3)
        self.label_21.setObjectName(u"label_21")
        self.label_21.setGeometry(QRect(20, 110, 851, 31))
        self.label_22 = QLabel(self.tab_3)
        self.label_22.setObjectName(u"label_22")
        self.label_22.setGeometry(QRect(20, 150, 831, 31))
        self.label_23 = QLabel(self.tab_3)
        self.label_23.setObjectName(u"label_23")
        self.label_23.setGeometry(QRect(50, 180, 821, 31))
        self.tabWidget.addTab(self.tab_3, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 922, 22))
        self.menuMenu = QMenu(self.menubar)
        self.menuMenu.setObjectName(u"menuMenu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuMenu.menuAction())
        self.menuMenu.addAction(self.actionClose)

        self.retranslateUi(MainWindow)

        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"PMFPredictor Toolkit", None))
        self.actionClose.setText(QCoreApplication.translate("MainWindow", u"Close", None))
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
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-size:20pt; font-weight:600;\">PMF Predictor Scripts</span></p></body></html>", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QCoreApplication.translate("MainWindow", u"Scripts", None))
        self.button_FindNPDownloadFolder.setText(QCoreApplication.translate("MainWindow", u"Open", None))
        self.text_newNPLocation.setText(QCoreApplication.translate("MainWindow", u"newsurf-001", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-size:20pt; font-weight:600;\">Surface from Charmm-GUI</span></p></body></html>", None))
        self.text_newNPName.setText(QCoreApplication.translate("MainWindow", u"newsurf-001", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"Download folder", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"Surface ID", None))
        self.button_buildNP.setText(QCoreApplication.translate("MainWindow", u"Add Surface", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"Shape", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"Source", None))
        self.label_13.setText(QCoreApplication.translate("MainWindow", u"SSD type", None))
        self.cb_npShape.setItemText(0, QCoreApplication.translate("MainWindow", u"Plane", None))
        self.cb_npShape.setItemText(1, QCoreApplication.translate("MainWindow", u"Cylinder", None))

        self.cb_npSource.setItemText(0, QCoreApplication.translate("MainWindow", u"SU (ions)", None))
        self.cb_npSource.setItemText(1, QCoreApplication.translate("MainWindow", u"UCD", None))
        self.cb_npSource.setItemText(2, QCoreApplication.translate("MainWindow", u"SU (no ions)", None))

        self.cb_npSSD.setItemText(0, QCoreApplication.translate("MainWindow", u"Fixed surface", None))
        self.cb_npSSD.setItemText(1, QCoreApplication.translate("MainWindow", u"Min z. dist", None))
        self.cb_npSSD.setItemText(2, QCoreApplication.translate("MainWindow", u"COM - slab width", None))
        self.cb_npSSD.setItemText(3, QCoreApplication.translate("MainWindow", u"Fixed (graphene-like)", None))

        self.label_14.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-size:20pt; font-weight:600;\">Chemical from SMILES</span></p></body></html>", None))
        self.text_newChemID.setText(QCoreApplication.translate("MainWindow", u"NEWCHEM", None))
        self.text_newChemSmiles.setText(QCoreApplication.translate("MainWindow", u"CC", None))
        self.text_newChemDesc.setText(QCoreApplication.translate("MainWindow", u"a new chemical", None))
        self.label_15.setText(QCoreApplication.translate("MainWindow", u"Chemical ID", None))
        self.label_16.setText(QCoreApplication.translate("MainWindow", u"Chemical SMILES", None))
        self.label_17.setText(QCoreApplication.translate("MainWindow", u"Formal charge", None))
        self.label_18.setText(QCoreApplication.translate("MainWindow", u"Description", None))
        self.button_addChem.setText(QCoreApplication.translate("MainWindow", u"Add Chemical", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), QCoreApplication.translate("MainWindow", u"Structures", None))
        self.label_19.setText(QCoreApplication.translate("MainWindow", u"Scripts should usually be run starting from the top and working downwards, as this means everything is calculated in order.", None))
        self.label_20.setText(QCoreApplication.translate("MainWindow", u"Unless you have a very good reason to change them, the check boxes should be left alone.", None))
        self.label_21.setText(QCoreApplication.translate("MainWindow", u"Adding a surface generates the corresponding entry in SurfaceDefinitions and makes a structure file.", None))
        self.label_22.setText(QCoreApplication.translate("MainWindow", u"Adding a chemical adds the entry to ChemicalDefinitions but doesn't make the structure automatically", None))
        self.label_23.setText(QCoreApplication.translate("MainWindow", u"Running the script to generate missing chemical structures batch-processes these, or you can manually add one.", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), QCoreApplication.translate("MainWindow", u"Tips", None))
        self.menuMenu.setTitle(QCoreApplication.translate("MainWindow", u"Menu", None))
    # retranslateUi

