#!/usr/bin/env python3
import sys
import os
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QPushButton, QFileDialog
from PySide6.QtCore import QFile, QProcess
from pmfpredictorgui.pmfpredictordashboard import Ui_MainWindow
from NPtoCSV import charmmguiToCSV

class MainWindow(QMainWindow):
    def updateStatusBar(self,s):
        self.ui.statusbar.showMessage(s)
    def enableButtonSet(self):
        for button in self.scriptButtonSet:
            button.setEnabled(True)
    def finishExternalScript(self):
        self.processHandler = None
        self.message("-------------------\n")
        self.message("Script finished\n")
        self.message("-------------------\n")
        self.enableButtonSet()
        self.updateStatusBar("Ready")
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
            matchParams = "0"
            if int(extraArgs)== 0:
                matchParams = "1"
            
            argString  = "-m "+matchParams
            argSet = ["BuildPredictedPMFs.py",argString]

        else:
            print("Target not recognised or not implemented")
        if foundScript == 1:
            self.disableButtonSet()
            self.updateStatusBar("Running "+argSet[0])
            self.processHandler.readyReadStandardOutput.connect(self.handle_stdout)
            self.processHandler.readyReadStandardError.connect(self.handle_stderr)
            self.processHandler.finished.connect(self.finishExternalScript)
            self.processHandler.start("python3",argSet)
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
        self.disableButtonSet()
        surfDefFile = open("Structures/SurfaceDefinitions.csv","r")
        knownSurfIDs = []
        for line in surfDefFile:
            if line[0]=="#":
                continue
            lineTerms = line.strip().split(",")
            knownSurfIDs.append(lineTerms[0])
        surfDefFile.close()        


        targetDir = self.ui.text_newNPLocation.text()
        targetName = self.ui.text_newNPName.text()

        if targetName in knownSurfIDs:
            self.message("A surface named " +targetName +" already exists, please choose a new name")
            return 0

        shapeMap = { "Plane":"plane", "Cylinder":"cylinder"}
        sourceMap = {"SU (no ions)":"0", "SU (ions)":"1", "UCD (110/111)":"2", "UCD (100)":"3"}
        ssdMap = {"Fixed surface":"0", "Min z. dist":"1", "COM - slab width":"2", "Fixed (graphene-like)":"3"}
        targetShapeString = shapeMap.get( self.ui.cb_npShape.currentText(),"plane")
        targetSourceString = sourceMap.get( self.ui.cb_npSource.currentText(), "1")
        targetSSDString = ssdMap.get(self.ui.cb_npSSD.currentText() , "0") 
        
        print("Making NP structure file: ", targetDir)
        res = charmmguiToCSV(targetName,targetDir)
        if res == 0:
            self.message("Failed to create NP, please check folder location\n")
        else:
            self.message("NP structure converted, appending to SurfaceDefinitions\n")
            newSurfString = targetName+","+targetShapeString+","+targetSourceString+","+targetSSDString
            self.message(newSurfString)
            surfDefFile = open("Structures/SurfaceDefinitions.csv","a")
            surfDefFile.write(newSurfString+"\n")
            surfDefFile.close()
        self.enableButtonSet()
    def addChemStructure(self):
        self.disableButtonSet()
        chemDefFile = open("Structures/ChemicalDefinitions.csv","r")
        knownChemIDs = []
        for line in chemDefFile:
            if line[0]=="#":
                continue
            lineTerms = line.strip().split(",")
            knownChemIDs.append(lineTerms[0])
        chemDefFile.close()
        chemID = self.ui.text_newChemID.text().replace(",","<COMMA>").replace("#","<HASH>")
        chemSMILES = self.ui.text_newChemSmiles.text().replace(",","<COMMA>").replace("#","<HASH>")
        chemCharge = self.ui.spinBox_newChemCharge.value()
        chemDesc = self.ui.text_newChemDesc.text().replace(",","<COMMA>").replace("#","<HASH>")
        chemStringOut = chemID+","+chemSMILES+","+str(chemCharge)+","+chemDesc
        if chemID in knownChemIDs:
            self.message("The chemical "+chemID+" is already registered, please choose a new name")
        else:
            self.message("Appending chemical string: "+chemStringOut)
            chemDefFile = open("Structures/ChemicalDefinitions.csv","a")
            chemDefFile.write(chemStringOut+"\n")
            chemDefFile.close()
            self.message("Chemical added, generate the structure using the script with Scan checked or manually add one")
        self.enableButtonSet()
    def __init__(self):
        super(MainWindow,self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.button_runAC.clicked.connect(lambda: self.buttonClick("runAC",self.ui.b_scanChem.isChecked()))
        self.ui.button_runGenChemPot.clicked.connect(lambda: self.buttonClick("runGenChemPot",self.ui.b_refreshChemPot.isChecked()))
        self.ui.button_runGenSurfPot.clicked.connect(lambda: self.buttonClick("runGenSurfPot",self.ui.b_refreshSurfPot.isChecked()))
        self.ui.button_runHGEChem.clicked.connect(lambda: self.buttonClick("runHGEChem",self.ui.b_refreshHGChem.isChecked()))
        self.ui.button_runHGESurf.clicked.connect(lambda: self.buttonClick("runHGESurf",self.ui.b_refreshHGSurf.isChecked()))
        self.ui.button_runPredict.clicked.connect(lambda: self.buttonClick("runPredict"  , self.ui.b_standardisePMF.isChecked()))
        self.processHandler = QProcess(self)
        self.scriptButtonSet = [self.ui.button_runAC,self.ui.button_runGenChemPot,self.ui.button_runGenSurfPot,self.ui.button_runHGEChem,self.ui.button_runHGESurf,self.ui.button_runPredict]
        #set up the buttons for NP conversion

        self.updateStatusBar("Ready")
        self.npFileFolder = []
        self.ui.button_buildNP.clicked.connect(self.buildNPStructure)
        self.ui.button_addChem.clicked.connect(self.addChemStructure)
        self.ui.button_FindNPDownloadFolder.clicked.connect(self.getFolderDialog)
        self.ui.actionClose.triggered.connect(self.close)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
