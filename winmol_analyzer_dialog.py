# -*- coding: utf-8 -*-
"""
/***************************************************************************
 WINMOLAnalyzer
Dialog
                                 A QGIS plugin
 Plugin to detect stems from UAV
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                             -------------------
        begin                : 2023-11-29
        git sha              : $Format:%H$
        copyright            : (C) 2023 by Hochschule für nachhaltige Entwicklung Eberswalde | mundialis GmbH & Co. KG | terrestris GmbH & Co. KG
        email                : Stefan.Reder@hnee.de
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

import os

from qgis.PyQt import QtWidgets, uic

# This loads your .ui file so that PyQt can populate your plugin with the elements from Qt Designer
FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'winmol_analyzer_dialog_base.ui'))


class WINMOLAnalyzerDialog(QtWidgets.QDialog, FORM_CLASS):
    def __init__(self, parent=None):
        """Constructor."""
        super(WINMOLAnalyzerDialog, self).__init__(parent)
        self.setupUi(self)
        self.set_connections()

    def set_connections(self):
        self.run_button.clicked.connect(self.run_process)
        self.model_comboBox.currentIndexChanged.connect(self.handleModelComboBoxChange)

    def handleModelComboBoxChange(self, index):
        selected_text = self.model_comboBox.currentText()
        if selected_text == "Custom":
            self.tileside_label.setEnabled(True)
            self.image_spinBox.setEnabled(True)
            self.model_lineEdit.setEnabled(True)
            self.model_toolButton.setEnabled(True)
            self.segm_label.setEnabled(True)
            self.tileside_doubleSpinBox.setEnabled(True)
            self.tileside_unit_label.setEnabled(True)
            self.image_label.setEnabled(True)
            self.image_spinBox.setEnabled(True)
            self.image_unit_label.setEnabled(True)
        else:
            self.tileside_label.setEnabled(False)
            self.image_spinBox.setEnabled(False)
            self.model_lineEdit.setEnabled(False)
            self.model_toolButton.setEnabled(False)
            self.segm_label.setEnabled(False)
            self.tileside_doubleSpinBox.setEnabled(False)
            self.tileside_unit_label.setEnabled(False)
            self.image_label.setEnabled(False)
            self.image_spinBox.setEnabled(False)
            self.image_unit_label.setEnabled(False)

    def run_process(self):
        print("run process")
