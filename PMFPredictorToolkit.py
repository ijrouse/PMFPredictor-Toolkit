#!/usr/bin/env python3
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QPushButton, QFileDialog
from PySide6.QtCore import QFile, QProcess
from pmfpredictorgui.pmfpredictordashboard import Ui_MainWindow
from NPtoCSV import charmmguiToCSV

class MainWindow(QMainWindow):

    def enableButtonSet(self):
        #print(self.processHandler.error())
        self.processHandler = None
        self.message("-------------------\n")
        self.message("Script finished\n")
        self.message("-------------------\n")
        for button in self.scriptButtonSet:
            button.setEnabled(True)
    def disableButtonSet(self):
        for button in self.scriptButtonSet:
            button.setEnabled(False)
    def buttonClick(self,buttonLabel,extraArgs = None):
        self.processHandler = QProcess(self)
        argString = ""
        if not extraArgs == None:
            argString = "with option " + str(extraArgs)
        foundScript = 0
        #QMessageBox.about(self,"Message","Button "+buttonLabel+ " successfully clicked" + argString)
        #self.message(buttonLabel)
        if buttonLabel=="runAC":
            foundScript = 1
            argString  = "-s "+str(int(extraArgs))
            argSet = ["ChemicalFromACPYPE.py",argString]
        elif buttonLabel=="runGenChemPot":
            foundScript = 1
            argString  = "-f "+str(int(extraArgs))
            argSet = ["GenerateChemicalPotentials.py",argString]
        elif buttonLabel=="runGenSurfPot":
            foundScript = 1
            argString  = "-f "+str(int(extraArgs))
            argSet = ["GenerateSurfacePotentials.py",argString]
        elif buttonLabel=="runHGEChem":
            foundScript = 1
            argString  = "-f "+str(int(extraArgs))
            argSet = ["HGExpandChemicalPotential.py",argString]
        elif buttonLabel=="runHGESurf":
            foundScript = 1
            argString  = "-f "+str(int(extraArgs))
            argSet = ["HGExpandSurfacePotential.py",argString]
        elif buttonLabel=="runPredict":
            foundScript = 1
            argString  = "-f "+str(int(extraArgs))
            argSet = ["BuildPredictedPMFs.py",""]

        else:
            print("Target not recognised or not implemented")
        if foundScript == 1:
            self.disableButtonSet()
            self.processHandler.start("python3",argSet)
            self.processHandler.readyReadStandardOutput.connect(self.handle_stdout)
            self.processHandler.readyReadStandardError.connect(self.handle_stderr)
            self.processHandler.finished.connect(self.enableButtonSet)
    def message(self,s):
        self.ui.textEdit.append(s)
    def handle_stdout(self):
        data = self.processHandler.readAllStandardOutput()
        stdout = bytes(data).decode("utf8")
        self.message(stdout)
    def handle_stderr(self):
        data = self.processHandler.readAllStandardError()
        stdout = bytes(data).decode("utf8")
        self.message(stdout)
    def getFolderDialog(self):
        self.NPdialog = QFileDialog(self)
        self.NPdialog.setFileMode(QFileDialog.Directory)
        self.NPdialog.setViewMode(QFileDialog.Detail)
        if self.NPdialog.exec():
            self.npFileFolder = self.NPdialog.selectedFiles()
            self.ui.text_newNPLocation.setText(  self.NPdialog.selectedFiles()[0])
            print(self.npFileFolder)
    def buildNPStructure(self):
        targetDir = self.ui.text_newNPLocation.text()
        targetName = self.ui.text_newNPName.text()
        print("Making NP structure file: ", targetDir)
        res = charmmguiToCSV(targetName,targetDir)
        if res == 0:
            self.message("Failed to create NP, please check folder location\n")
        else:
            self.message("Created NP successfully. Remember to update SurfaceDefinitions.csv to define the geometry, source and SSD type.")
    def __init__(self):
        super(MainWindow,self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.button_runAC.clicked.connect(lambda: self.buttonClick("runAC",self.ui.b_scanChem.isChecked()))
        self.ui.button_runGenChemPot.clicked.connect(lambda: self.buttonClick("runGenChemPot",self.ui.b_refreshChemPot.isChecked()))
        self.ui.button_runGenSurfPot.clicked.connect(lambda: self.buttonClick("runGenSurfPot",self.ui.b_refreshSurfPot.isChecked()))
        self.ui.button_runHGEChem.clicked.connect(lambda: self.buttonClick("runHGEChem",self.ui.b_refreshHGChem.isChecked()))
        self.ui.button_runHGESurf.clicked.connect(lambda: self.buttonClick("runHGESurf",self.ui.b_refreshHGSurf.isChecked()))
        self.ui.button_runPredict.clicked.connect(lambda: self.buttonClick("runPredict"))
        self.processHandler = QProcess(self)
        self.scriptButtonSet = [self.ui.button_runAC,self.ui.button_runGenChemPot,self.ui.button_runGenSurfPot,self.ui.button_runHGEChem,self.ui.button_runHGESurf,self.ui.button_runPredict]
        #set up the buttons for NP conversion


        self.npFileFolder = []
        self.ui.button_buildNP.clicked.connect(self.buildNPStructure)
        self.ui.button_FindNPDownloadFolder.clicked.connect(self.getFolderDialog)
        self.ui.actionClose.triggered.connect(self.close)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
