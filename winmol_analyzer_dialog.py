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
import sys
import subprocess

#from tensorflow import keras


from qgis.core import QgsProject, QgsVectorLayer
from qgis.PyQt import QtWidgets, uic
#from shapely import LineString, Point

# Set up current path.
current_path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(current_path + 'classes'))
sys.path.append(os.path.abspath(current_path + 'utils'))
sys.path.append(os.path.abspath(current_path + 'qgisutil'))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

#from utils import IO
#from utils import Prediction as Pred

from classes.Stem import Stem
from classes.Config import Config
from qgisutil.FeatureFactory import FeatureFactory

from PyQt5.QtWidgets import QFileDialog


# This loads your .ui file so that PyQt can populate your plugin with the
# elements from Qt Designer
FORM_CLASS, _ = uic.loadUiType(
    os.path.join(current_path, 'winmol_analyzer_dialog_base.ui')
)


class WINMOLAnalyzerDialog(QtWidgets.QDialog, FORM_CLASS):
    ff: FeatureFactory

    def __init__(self, parent=None):
        """Constructor."""
        super(WINMOLAnalyzerDialog, self).__init__(parent)
        self.setupUi(self)

        self.ff = FeatureFactory()

        # parameters
        self.img_path = None
        self.model_path = None
        self.stem_dir = None
        self.trees_dir = None
        self.nodes_dir = None

        # Create a Config instance
        self.config = Config()

        self.set_connections()
        self.output_log.setReadOnly(True)

    def set_connections(self):
        self.run_button.clicked.connect(self.run_process)
        self.model_comboBox.currentIndexChanged.connect(
            self.handleModelComboBoxChange)
        self.set_parameters()
        self.set_default_config_parameters()
        self.get_config_parameters_from_gui()
        self.uav_toolButton.clicked.connect(self.uav_file_dialog)
        self.model_toolButton.clicked.connect(self.model_file_dialog)
        self.output_toolButton_stem.clicked.connect(self.file_dialog_stem)
        self.output_toolButton_trees.clicked.connect(self.file_dialog_trees)
        self.output_toolButton_nodes.clicked.connect(self.file_dialog_nodes)
        self.output_checkBox_stem.stateChanged.connect(self.checkbox_changed_stem)
        self.output_checkBox_trees.stateChanged.connect(self.checkbox_changed_trees)
        self.output_checkBox_nodes.stateChanged.connect(self.checkbox_changed_nodes)
        self.close_button.clicked.connect(self.close_application)

    def handleModelComboBoxChange(self, index):
        selected_text = self.model_comboBox.currentText()
        widgets_to_enable = [
            self.tileside_label, self.image_spinBox, self.model_lineEdit,
            self.model_toolButton, self.segm_label, self.tileside_doubleSpinBox,
            self.tileside_unit_label, self.image_label, self.image_spinBox,
            self.image_unit_label
        ]

        for widget in widgets_to_enable:
            widget.setEnabled(selected_text == "Custom")

        if selected_text == "Custom":
            self.apply_style_to_line_edit(self.model_lineEdit, True)
        else:
            self.apply_style_to_line_edit(self.model_lineEdit, False)
    def model_file_dialog(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "Model File (*.hdf5);;All Files (*)", options=options)
        if file_path:
            self.model_lineEdit.setText(file_path)

    def set_parameters(self):
        # Extract command-line arguments
        self.img_path = self.uav_lineEdit.text()
        self.model_path = self.model_lineEdit.text()
        self.stem_dir = self.output_lineEdit_stem.text()
        self.trees_dir = self.output_lineEdit_trees.text()
        self.nodes_dir = self.output_lineEdit_nodes.text()

    def set_default_config_parameters(self):
        # set default values
        self.minlength_doubleSpinBox.setValue(self.config.min_length)
        self.maxdistance_doubleSpinBox.setValue(self.config.max_distance)
        self.tolerance_doubleSpinBox.setValue(self.config.tolerance_angle)
        self.maxtree_doubleSpinBox.setValue(self.config.max_tree_height)
        self.tileside_doubleSpinBox.setValue(self.config.tile_size)
        self.image_spinBox.setValue(self.config.img_width)

    def get_config_parameters_from_gui(self):
        self.config.min_length = self.minlength_doubleSpinBox.value()
        self.config.max_distance = self.maxdistance_doubleSpinBox.value()
        self.config.tolerance_angle = self.tolerance_doubleSpinBox.value()
        self.config.max_tree_height = self.maxtree_doubleSpinBox.value()

    def uav_file_dialog(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select TIFF File", "", "TIFF Files (*.tiff *.tif);;All Files (*)", options=options)
        if file_path:
            self.uav_lineEdit.setText(file_path)

    def file_dialog_stem(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Select Location And Name For Stem Map", "", "All Files (*)", options=options)
        if file_path:
            self.output_lineEdit_stem.setText(file_path)

    def file_dialog_trees(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Select Location And Name For Semantic Stem Map", "", "All Files (*)", options=options)
        if file_path:
            self.output_lineEdit_trees.setText(file_path)

    def file_dialog_nodes(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Select Location And Name For Measuring Nodes", "", "All Files (*)", options=options)
        if file_path: # C:/Users/49176/Documents/WINMOL_Analyzer/test_stem
            self.output_lineEdit_nodes.setText(file_path)

    def checkbox_changed_stem(self, state):
        is_checked = state == 2
        self.output_lineEdit_stem.setEnabled(is_checked)
        self.output_toolButton_stem.setEnabled(is_checked)
        self.apply_style_to_line_edit(self.output_lineEdit_stem, is_checked)

    def checkbox_changed_trees(self, state):
        is_checked = state == 2
        self.output_lineEdit_trees.setEnabled(is_checked)
        self.output_toolButton_trees.setEnabled(is_checked)
        self.apply_style_to_line_edit(self.output_lineEdit_trees, is_checked)

    def checkbox_changed_nodes(self, state):
        is_checked = state == 2
        self.output_lineEdit_nodes.setEnabled(is_checked)
        self.output_toolButton_nodes.setEnabled(is_checked)
        self.apply_style_to_line_edit(self.output_lineEdit_nodes, is_checked)

    def apply_style_to_line_edit(self, line_edit, is_checked):
        # Enable or disable the QLineEdit based on the checkbox state
        line_edit.setEnabled(is_checked)

        # Set the stylesheet to gray out the QLineEdit when it is disabled
        if not is_checked:
            line_edit.setStyleSheet("")
        else:
            line_edit.setStyleSheet("QLineEdit { background-color: rgb(255, 255, 255) }")

    def close_application(self):
        print("Closing application")
        self.close()

    def run_process(self):
        # Path to the Python script
        script_path = r'C:\Users\49176\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\WINMOL_Analyzer\winmol_run.py'

        model_path = self.model_lineEdit.text()
        img_path = self.uav_lineEdit.text()
        stem_dir = os.path.dirname(self.output_lineEdit_stem.text())
        trees_dir = os.path.dirname(self.output_lineEdit_trees.text())

        # Command to run the script
        command = [
            'python',
            script_path,
            model_path,
            img_path,
            stem_dir,
            trees_dir
        ]

        # Switch to the log tab in the QTabWidget
        self.log_widget.setCurrentIndex(1)

        # Run the process
        try:
            #print(os.getenv('PATH'))
            process = subprocess.run(command, check=True, capture_output=True, text=True)
            # Display the output in the QPlainTextEdit
            output_text = process.stdout + process.stderr
            print(process.stderr)
            self.output_log.setPlainText(output_text)

            print("Process output:", process.stdout)
        except subprocess.CalledProcessError as e:
            print("Error running the process:", e)
            error_output = e.output if e.output else e.stderr
            print("Process output (if any):", error_output)
            self.output_log.setPlainText("Error running the process:\n" + e.output)



from PyQt5.QtCore import Qt, QThread, pyqtSignal


class RunProcessThread(QThread):
    update_signal = pyqtSignal(str)

    def __init__(self, command):
        super(RunProcessThread, self).__init__()
        self.command = command

    def run(self):
        try:
            process = subprocess.run(self.command, check=True, capture_output=True, text=True, stderr=subprocess.PIPE)
            output = process.stdout
            if process.stderr:
                output += "\nError output:\n" + process.stderr
            self.update_signal.emit(output)
        except subprocess.CalledProcessError as e:
            error_output = e.output
            if e.stderr:
                error_output += "\nError output:\n" + e.stderr
            self.update_signal.emit(f"Error running the process: {e}\n{error_output}")




#/home/mmawad/miniconda3/envs/WINMOL_Analyzer/bin/python.exe
# standalone/WINMOL_Analyzer.py
# /home/mmawad/repos/WINMOL_Analyzer/standalone/model/UNet_SpecDS_UNet_Mask-RCNN_512_Spruce_2_model_2023-02-27_061925.hdf5
# /home/mmawad/repos/WINMOL_Analyzer/standalone/input/test.tiff
# /home/mmawad/repos/WINMOL_Analyzer/standalone/predict/
# /home/mmawad/repos/WINMOL_Analyzer/standalone/output/
