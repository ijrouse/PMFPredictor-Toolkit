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
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFormLayout,
    QFrame, QGridLayout, QLabel, QLineEdit,
    QMainWindow, QMenu, QMenuBar, QPushButton,
    QSizePolicy, QSpinBox, QStatusBar, QTabWidget,
    QTextEdit, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(899, 932)
        icon = QIcon()
        icon.addFile(u"pmfp.ico", QSize(), QIcon.Normal, QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.actionClose = QAction(MainWindow)
        self.actionClose.setObjectName(u"actionClose")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.formLayout = QFormLayout(self.centralwidget)
        self.formLayout.setObjectName(u"formLayout")
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.verticalLayout = QVBoxLayout(self.tab_2)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.gridLayout_5 = QGridLayout()
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.label_7 = QLabel(self.tab_2)
        self.label_7.setObjectName(u"label_7")

        self.gridLayout_5.addWidget(self.label_7, 0, 0, 1, 1)

        self.label_14 = QLabel(self.tab_2)
        self.label_14.setObjectName(u"label_14")

        self.gridLayout_5.addWidget(self.label_14, 0, 1, 1, 1)

        self.gridLayout_2 = QGridLayout()
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.label_8 = QLabel(self.tab_2)
        self.label_8.setObjectName(u"label_8")

        self.gridLayout_2.addWidget(self.label_8, 0, 0, 1, 1)

        self.text_newNPLocation = QLineEdit(self.tab_2)
        self.text_newNPLocation.setObjectName(u"text_newNPLocation")

        self.gridLayout_2.addWidget(self.text_newNPLocation, 0, 1, 1, 1)

        self.button_FindNPDownloadFolder = QPushButton(self.tab_2)
        self.button_FindNPDownloadFolder.setObjectName(u"button_FindNPDownloadFolder")

        self.gridLayout_2.addWidget(self.button_FindNPDownloadFolder, 1, 1, 1, 1)

        self.label_9 = QLabel(self.tab_2)
        self.label_9.setObjectName(u"label_9")

        self.gridLayout_2.addWidget(self.label_9, 2, 0, 1, 1)

        self.text_newNPName = QLineEdit(self.tab_2)
        self.text_newNPName.setObjectName(u"text_newNPName")

        self.gridLayout_2.addWidget(self.text_newNPName, 2, 1, 1, 1)

        self.label_11 = QLabel(self.tab_2)
        self.label_11.setObjectName(u"label_11")

        self.gridLayout_2.addWidget(self.label_11, 3, 0, 1, 1)

        self.cb_npShape = QComboBox(self.tab_2)
        self.cb_npShape.addItem("")
        self.cb_npShape.addItem("")
        self.cb_npShape.setObjectName(u"cb_npShape")

        self.gridLayout_2.addWidget(self.cb_npShape, 3, 1, 1, 1)

        self.label_12 = QLabel(self.tab_2)
        self.label_12.setObjectName(u"label_12")

        self.gridLayout_2.addWidget(self.label_12, 4, 0, 1, 1)

        self.cb_npSource = QComboBox(self.tab_2)
        self.cb_npSource.addItem("")
        self.cb_npSource.addItem("")
        self.cb_npSource.addItem("")
        self.cb_npSource.addItem("")
        self.cb_npSource.setObjectName(u"cb_npSource")

        self.gridLayout_2.addWidget(self.cb_npSource, 4, 1, 1, 1)

        self.label_13 = QLabel(self.tab_2)
        self.label_13.setObjectName(u"label_13")

        self.gridLayout_2.addWidget(self.label_13, 5, 0, 1, 1)

        self.cb_npSSD = QComboBox(self.tab_2)
        self.cb_npSSD.addItem("")
        self.cb_npSSD.addItem("")
        self.cb_npSSD.addItem("")
        self.cb_npSSD.addItem("")
        self.cb_npSSD.setObjectName(u"cb_npSSD")

        self.gridLayout_2.addWidget(self.cb_npSSD, 5, 1, 1, 1)


        self.gridLayout_5.addLayout(self.gridLayout_2, 1, 0, 1, 1)

        self.formLayout_2 = QFormLayout()
        self.formLayout_2.setObjectName(u"formLayout_2")
        self.label_15 = QLabel(self.tab_2)
        self.label_15.setObjectName(u"label_15")

        self.formLayout_2.setWidget(0, QFormLayout.LabelRole, self.label_15)

        self.text_newChemID = QLineEdit(self.tab_2)
        self.text_newChemID.setObjectName(u"text_newChemID")

        self.formLayout_2.setWidget(0, QFormLayout.FieldRole, self.text_newChemID)

        self.label_16 = QLabel(self.tab_2)
        self.label_16.setObjectName(u"label_16")

        self.formLayout_2.setWidget(1, QFormLayout.LabelRole, self.label_16)

        self.text_newChemSmiles = QLineEdit(self.tab_2)
        self.text_newChemSmiles.setObjectName(u"text_newChemSmiles")

        self.formLayout_2.setWidget(1, QFormLayout.FieldRole, self.text_newChemSmiles)

        self.label_17 = QLabel(self.tab_2)
        self.label_17.setObjectName(u"label_17")

        self.formLayout_2.setWidget(2, QFormLayout.LabelRole, self.label_17)

        self.spinBox_newChemCharge = QSpinBox(self.tab_2)
        self.spinBox_newChemCharge.setObjectName(u"spinBox_newChemCharge")
        self.spinBox_newChemCharge.setMinimum(-5)
        self.spinBox_newChemCharge.setMaximum(5)

        self.formLayout_2.setWidget(2, QFormLayout.FieldRole, self.spinBox_newChemCharge)

        self.label_18 = QLabel(self.tab_2)
        self.label_18.setObjectName(u"label_18")

        self.formLayout_2.setWidget(3, QFormLayout.LabelRole, self.label_18)

        self.text_newChemDesc = QLineEdit(self.tab_2)
        self.text_newChemDesc.setObjectName(u"text_newChemDesc")

        self.formLayout_2.setWidget(3, QFormLayout.FieldRole, self.text_newChemDesc)


        self.gridLayout_5.addLayout(self.formLayout_2, 1, 1, 1, 1)

        self.button_buildNP = QPushButton(self.tab_2)
        self.button_buildNP.setObjectName(u"button_buildNP")

        self.gridLayout_5.addWidget(self.button_buildNP, 2, 0, 1, 1)

        self.button_addChem = QPushButton(self.tab_2)
        self.button_addChem.setObjectName(u"button_addChem")

        self.gridLayout_5.addWidget(self.button_addChem, 2, 1, 1, 1)


        self.verticalLayout.addLayout(self.gridLayout_5)

        self.tabWidget.addTab(self.tab_2, "")
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.gridLayout_3 = QGridLayout(self.tab)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.button_runGenChemPot = QPushButton(self.tab)
        self.button_runGenChemPot.setObjectName(u"button_runGenChemPot")

        self.gridLayout.addWidget(self.button_runGenChemPot, 2, 2, 2, 1)

        self.label_6 = QLabel(self.tab)
        self.label_6.setObjectName(u"label_6")

        self.gridLayout.addWidget(self.label_6, 10, 0, 1, 1)

        self.b_refreshHGChem = QCheckBox(self.tab)
        self.b_refreshHGChem.setObjectName(u"b_refreshHGChem")
        self.b_refreshHGChem.setChecked(False)

        self.gridLayout.addWidget(self.b_refreshHGChem, 7, 1, 1, 1)

        self.label_3 = QLabel(self.tab)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout.addWidget(self.label_3, 4, 0, 1, 1)

        self.line = QFrame(self.tab)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.gridLayout.addWidget(self.line, 1, 0, 1, 3)

        self.button_runHGESurf = QPushButton(self.tab)
        self.button_runHGESurf.setObjectName(u"button_runHGESurf")

        self.gridLayout.addWidget(self.button_runHGESurf, 8, 2, 1, 1)

        self.b_refreshChemPot = QCheckBox(self.tab)
        self.b_refreshChemPot.setObjectName(u"b_refreshChemPot")

        self.gridLayout.addWidget(self.b_refreshChemPot, 3, 1, 1, 1)

        self.b_scanChem = QCheckBox(self.tab)
        self.b_scanChem.setObjectName(u"b_scanChem")
        self.b_scanChem.setCheckable(True)
        self.b_scanChem.setChecked(True)

        self.gridLayout.addWidget(self.b_scanChem, 0, 1, 1, 1)

        self.button_runAC = QPushButton(self.tab)
        self.button_runAC.setObjectName(u"button_runAC")

        self.gridLayout.addWidget(self.button_runAC, 0, 2, 1, 1)

        self.line_2 = QFrame(self.tab)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.HLine)
        self.line_2.setFrameShadow(QFrame.Sunken)

        self.gridLayout.addWidget(self.line_2, 5, 0, 1, 3)

        self.label_4 = QLabel(self.tab)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout.addWidget(self.label_4, 6, 0, 2, 1)

        self.label_5 = QLabel(self.tab)
        self.label_5.setObjectName(u"label_5")

        self.gridLayout.addWidget(self.label_5, 8, 0, 1, 1)

        self.button_runPredict = QPushButton(self.tab)
        self.button_runPredict.setObjectName(u"button_runPredict")

        self.gridLayout.addWidget(self.button_runPredict, 10, 2, 1, 1)

        self.b_refreshHGSurf = QCheckBox(self.tab)
        self.b_refreshHGSurf.setObjectName(u"b_refreshHGSurf")
        self.b_refreshHGSurf.setChecked(False)

        self.gridLayout.addWidget(self.b_refreshHGSurf, 8, 1, 1, 1)

        self.label = QLabel(self.tab)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.line_3 = QFrame(self.tab)
        self.line_3.setObjectName(u"line_3")
        self.line_3.setFrameShape(QFrame.HLine)
        self.line_3.setFrameShadow(QFrame.Sunken)

        self.gridLayout.addWidget(self.line_3, 9, 0, 1, 3)

        self.button_runGenSurfPot = QPushButton(self.tab)
        self.button_runGenSurfPot.setObjectName(u"button_runGenSurfPot")

        self.gridLayout.addWidget(self.button_runGenSurfPot, 4, 2, 1, 1)

        self.b_refreshSurfPot = QCheckBox(self.tab)
        self.b_refreshSurfPot.setObjectName(u"b_refreshSurfPot")

        self.gridLayout.addWidget(self.b_refreshSurfPot, 4, 1, 1, 1)

        self.button_runHGEChem = QPushButton(self.tab)
        self.button_runHGEChem.setObjectName(u"button_runHGEChem")

        self.gridLayout.addWidget(self.button_runHGEChem, 6, 2, 2, 1)

        self.label_2 = QLabel(self.tab)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 2, 0, 2, 1)

        self.b_standardisePMF = QCheckBox(self.tab)
        self.b_standardisePMF.setObjectName(u"b_standardisePMF")
        self.b_standardisePMF.setChecked(True)

        self.gridLayout.addWidget(self.b_standardisePMF, 10, 1, 1, 1)


        self.gridLayout_3.addLayout(self.gridLayout, 1, 0, 1, 1)

        self.label_10 = QLabel(self.tab)
        self.label_10.setObjectName(u"label_10")

        self.gridLayout_3.addWidget(self.label_10, 0, 0, 1, 1)

        self.tabWidget.addTab(self.tab, "")
        self.tab_3 = QWidget()
        self.tab_3.setObjectName(u"tab_3")
        self.gridLayout_4 = QGridLayout(self.tab_3)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.label_19 = QLabel(self.tab_3)
        self.label_19.setObjectName(u"label_19")

        self.gridLayout_4.addWidget(self.label_19, 0, 0, 1, 1)

        self.label_20 = QLabel(self.tab_3)
        self.label_20.setObjectName(u"label_20")

        self.gridLayout_4.addWidget(self.label_20, 1, 0, 1, 1)

        self.label_21 = QLabel(self.tab_3)
        self.label_21.setObjectName(u"label_21")

        self.gridLayout_4.addWidget(self.label_21, 2, 0, 1, 1)

        self.label_24 = QLabel(self.tab_3)
        self.label_24.setObjectName(u"label_24")

        self.gridLayout_4.addWidget(self.label_24, 3, 0, 1, 1)

        self.label_22 = QLabel(self.tab_3)
        self.label_22.setObjectName(u"label_22")

        self.gridLayout_4.addWidget(self.label_22, 4, 0, 1, 1)

        self.label_23 = QLabel(self.tab_3)
        self.label_23.setObjectName(u"label_23")

        self.gridLayout_4.addWidget(self.label_23, 5, 0, 1, 1)

        self.tabWidget.addTab(self.tab_3, "")

        self.formLayout.setWidget(0, QFormLayout.SpanningRole, self.tabWidget)

        self.textEdit = QTextEdit(self.centralwidget)
        self.textEdit.setObjectName(u"textEdit")
        self.textEdit.setReadOnly(True)

        self.formLayout.setWidget(1, QFormLayout.SpanningRole, self.textEdit)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 899, 22))
        self.menuMenu = QMenu(self.menubar)
        self.menuMenu.setObjectName(u"menuMenu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuMenu.menuAction())
        self.menuMenu.addAction(self.actionClose)

        self.retranslateUi(MainWindow)

        self.tabWidget.setCurrentIndex(1)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"PMFPredictor Toolkit", None))
        self.actionClose.setText(QCoreApplication.translate("MainWindow", u"Close", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-size:20pt; font-weight:600;\">Surface from Charmm-GUI</span></p></body></html>", None))
        self.label_14.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-size:20pt; font-weight:600;\">Chemical from SMILES</span></p></body></html>", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"Download folder", None))
        self.text_newNPLocation.setText(QCoreApplication.translate("MainWindow", u"newsurf-001", None))
        self.button_FindNPDownloadFolder.setText(QCoreApplication.translate("MainWindow", u"Find folder", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"Surface ID", None))
        self.text_newNPName.setText(QCoreApplication.translate("MainWindow", u"newsurf-001", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"Shape", None))
        self.cb_npShape.setItemText(0, QCoreApplication.translate("MainWindow", u"Plane", None))
        self.cb_npShape.setItemText(1, QCoreApplication.translate("MainWindow", u"Cylinder", None))

        self.label_12.setText(QCoreApplication.translate("MainWindow", u"Source", None))
        self.cb_npSource.setItemText(0, QCoreApplication.translate("MainWindow", u"SU (ions)", None))
        self.cb_npSource.setItemText(1, QCoreApplication.translate("MainWindow", u"UCD (110/111)", None))
        self.cb_npSource.setItemText(2, QCoreApplication.translate("MainWindow", u"SU (no ions)", None))
        self.cb_npSource.setItemText(3, QCoreApplication.translate("MainWindow", u"UCD (100)", None))

        self.label_13.setText(QCoreApplication.translate("MainWindow", u"SSD type", None))
        self.cb_npSSD.setItemText(0, QCoreApplication.translate("MainWindow", u"Fixed surface", None))
        self.cb_npSSD.setItemText(1, QCoreApplication.translate("MainWindow", u"Min z. dist", None))
        self.cb_npSSD.setItemText(2, QCoreApplication.translate("MainWindow", u"COM - slab width", None))
        self.cb_npSSD.setItemText(3, QCoreApplication.translate("MainWindow", u"Fixed (graphene-like)", None))

        self.label_15.setText(QCoreApplication.translate("MainWindow", u"Chemical ID", None))
        self.text_newChemID.setText(QCoreApplication.translate("MainWindow", u"NEWCHEM", None))
        self.label_16.setText(QCoreApplication.translate("MainWindow", u"Chemical SMILES", None))
        self.text_newChemSmiles.setText(QCoreApplication.translate("MainWindow", u"CC", None))
        self.label_17.setText(QCoreApplication.translate("MainWindow", u"Formal charge", None))
        self.label_18.setText(QCoreApplication.translate("MainWindow", u"Description", None))
        self.text_newChemDesc.setText(QCoreApplication.translate("MainWindow", u"a new chemical", None))
        self.button_buildNP.setText(QCoreApplication.translate("MainWindow", u"Add Surface", None))
        self.button_addChem.setText(QCoreApplication.translate("MainWindow", u"Add Chemical", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), QCoreApplication.translate("MainWindow", u"Structures", None))
        self.button_runGenChemPot.setText(QCoreApplication.translate("MainWindow", u"Run", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"Generate PMFs", None))
        self.b_refreshHGChem.setText(QCoreApplication.translate("MainWindow", u"Refresh?", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Generate surface potentials", None))
        self.button_runHGESurf.setText(QCoreApplication.translate("MainWindow", u"Run", None))
        self.b_refreshChemPot.setText(QCoreApplication.translate("MainWindow", u"Refresh?", None))
        self.b_scanChem.setText(QCoreApplication.translate("MainWindow", u"Scan", None))
        self.button_runAC.setText(QCoreApplication.translate("MainWindow", u"Run", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Generate chemical potential expansions", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Generate surface potential expansions", None))
        self.button_runPredict.setText(QCoreApplication.translate("MainWindow", u"Run", None))
        self.b_refreshHGSurf.setText(QCoreApplication.translate("MainWindow", u"Refresh?", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Generate missing chemical structures", None))
        self.button_runGenSurfPot.setText(QCoreApplication.translate("MainWindow", u"Run", None))
        self.b_refreshSurfPot.setText(QCoreApplication.translate("MainWindow", u"Refresh?", None))
        self.button_runHGEChem.setText(QCoreApplication.translate("MainWindow", u"Run", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Generate chemical potentials", None))
        self.b_standardisePMF.setText(QCoreApplication.translate("MainWindow", u"Standardise?", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-size:20pt; font-weight:600;\">PMF Predictor Scripts</span></p></body></html>", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QCoreApplication.translate("MainWindow", u"Scripts", None))
        self.label_19.setText(QCoreApplication.translate("MainWindow", u"Scripts should usually be run starting from the top and working downwards, as this means everything is calculated in order.", None))
        self.label_20.setText(QCoreApplication.translate("MainWindow", u"Unless you have a very good reason to change them, the check boxes should be left alone.", None))
        self.label_21.setText(QCoreApplication.translate("MainWindow", u"Adding a surface generates the corresponding entry in SurfaceDefinitions and makes a structure file.", None))
        self.label_24.setText(QCoreApplication.translate("MainWindow", u"--The folder has to be the base directory, not the GROMACS subfolder. Usually charmm-gui-XXXXXXXXXX", None))
        self.label_22.setText(QCoreApplication.translate("MainWindow", u"Adding a chemical adds the entry to ChemicalDefinitions but doesn't make the structure automatically", None))
        self.label_23.setText(QCoreApplication.translate("MainWindow", u"--Running the script to generate missing chemical structures batch-processes these, or you can manually add one.", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), QCoreApplication.translate("MainWindow", u"Tips", None))
        self.menuMenu.setTitle(QCoreApplication.translate("MainWindow", u"Menu", None))
    # retranslateUi

